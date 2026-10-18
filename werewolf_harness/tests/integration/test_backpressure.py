"""Overload protection end to end: queued games in Postgres, FIFO start when the
load drops, refusal past the queue, restart, and the HTTP surface (202 / 503)."""
from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from werewolf_harness.app.runtime.admission import Overloaded
from werewolf_harness.app.runtime.agents import LocalExecutor
from werewolf_harness.app.runtime.load import LoadSample
from werewolf_harness.app.runtime.runtime import Runtime
from werewolf_harness.app.runtime.table import TableConfig

from .helpers import until
from .test_api import _clean, _guest


def record_launches(rt: Runtime) -> list:
    """The tables rt starts, in order. A mock game can finish and leave rt.actors before a
    poll sees it there, so the order (and the table) is taken at launch instead."""
    launched, launch = [], rt._launch

    async def recording(table):
        launched.append(table)
        return await launch(table)
    rt._launch = recording
    return launched


class LoadDial(LocalExecutor):
    """A real executor whose reported load is set by the test."""

    def __init__(self):
        super().__init__(4)
        self.heavy = False

    def load_sample(self) -> LoadSample:
        s = super().load_sample()
        if self.heavy:
            return LoadSample(depth={"online": 40}, oldest_wait_s=30.0, inflight=4, capacity=4, turn_p50_ms=5000)
        return s


def _settings(settings, **kw):
    return settings.model_copy(update={"load_tick_s": 0.05, "overload_hold_s": 0.2, "admission_probe_s": 0.0,
                                       "overload_per_game_tau_s": 0.1,
                                       "max_queued_games": 3, **kw})


CFG = TableConfig(variant="8p", models=["mock"])


async def test_calm_node_admits_at_once(settings, store, redis):
    rt = Runtime(_settings(settings), store, redis, executor=LoadDial())
    await rt.start(scan=False)
    d, a, info = await rt.request(CFG, seed=1)
    assert d == "admit" and a is not None and info == {}
    assert (await store.get_game(a.id))["state"] == "running"
    await rt.shutdown()


async def test_overloaded_node_queues_then_starts_fifo_when_load_drops(settings, store, redis):
    ex = LoadDial()
    rt = Runtime(_settings(settings), store, redis, executor=ex)
    await rt.start(scan=False)
    running = await rt.create(CFG, seed=1)                  # a game already in flight
    ex.heavy = True
    await rt.refresh_load()
    assert rt.monitor.overloaded
    queued = []
    for i in range(3):
        d, a, info = await rt.request(CFG, seed=10 + i)
        assert d == "queue" and a is None and info["position"] == i + 1 and info["eta_s"] >= 1
        assert (await store.get_game(info["game_id"]))["state"] == "queued"
        queued.append(info["game_id"])
    with pytest.raises(Overloaded) as e:                    # the line is full
        await rt.request(CFG, seed=99)
    assert 5 <= e.value.retry_after_s <= 600
    # running games are never paused: the one in flight keeps committing while new ones wait
    v0 = len(running.t.journal)
    await until(lambda: len(running.t.journal) > v0 + 3, 20)
    assert all(g not in rt.actors for g in queued)
    lv = rt.load_view()
    assert lv["overloaded"] and lv["admission"]["queued_games"] == 3 and lv["level"] == "overloaded"

    launched = record_launches(rt)
    ex.heavy = False                                        # the provider recovers
    await until(lambda: len(launched) == len(queued), 20)
    assert [t.id for t in launched] == queued               # first come, first started
    assert rt.admission.waits and len(rt.admission) == 0
    for g in queued:                                        # each one really plays (it may be over already)
        for _ in range(1000):
            if await store.journal(g):
                break
            await asyncio.sleep(0.02)
        assert await store.journal(g) and (await store.get_game(g))["state"] in ("running", "over")
    await rt.shutdown()


async def test_queued_games_survive_a_restart_in_order(settings, store, redis):
    ex = LoadDial()
    ex.heavy = True
    s = _settings(settings)
    rt = Runtime(s, store, redis, executor=ex)
    await rt.start(scan=False)
    await rt.refresh_load()
    ids = [(await rt.request(CFG, seed=20 + i))[2]["game_id"] for i in range(2)]
    await rt.shutdown()
    ex2 = LoadDial()
    ex2.heavy = True
    rt2 = Runtime(s, store, redis, executor=ex2)
    launched = record_launches(rt2)
    await rt2.start(scan=True)                              # recovery must not adopt a queued game
    assert [rt2.admission.position(g) for g in ids] == [1, 2] and not rt2.actors
    ex2.heavy = False
    await until(lambda: len(launched) == 2, 20)
    assert [t.id for t in launched] == ids
    await rt2.shutdown()


async def test_a_queued_game_keeps_its_engine_version_across_a_restart(settings, store, redis):
    """The rules are pinned when the game is queued, not when it finally starts: a node
    restarted on newer code still starts it under the version it was created with."""
    from werewolf_harness.envs import BASELINE_ENGINE_VERSION
    ex = LoadDial()
    ex.heavy = True
    s = _settings(settings)
    rt = Runtime(s, store, redis, executor=ex)
    await rt.start(scan=False)
    await rt.refresh_load()
    old = TableConfig(variant="8p", models=["mock"], engine_version=BASELINE_ENGINE_VERSION)
    gid = (await rt.request(old, seed=31))[2]["game_id"]
    assert (await store.get_game(gid))["engine_version"] == BASELINE_ENGINE_VERSION
    await rt.shutdown()
    ex2 = LoadDial()
    rt2 = Runtime(s, store, redis, executor=ex2)
    launched = record_launches(rt2)
    await rt2.start(scan=True)
    await until(lambda: len(launched) == 1, 20)
    assert launched[0].id == gid and launched[0].engine_version == BASELINE_ENGINE_VERSION
    assert launched[0].env.rules.tie_revote is False
    await rt2.shutdown()


async def test_admission_off_starts_everything(settings, store, redis):
    ex = LoadDial()
    ex.heavy = True
    rt = Runtime(_settings(settings, admission=False), store, redis, executor=ex)
    await rt.start(scan=False)
    d, a, _ = await rt.request(CFG, seed=3)
    assert d == "admit" and a is not None
    await rt.shutdown()


# ------------------------------------------------------------------ HTTP
@pytest.fixture
def dial_client(_migrated, settings):
    made = []

    def make(**over):
        asyncio.run(_clean())
        from werewolf_harness.app.api.main import create_app
        ex = LoadDial()
        app = create_app(_settings(settings, **over), migrate_on_start=False, env_keys={}, executor=ex)
        c = TestClient(app)
        c.__enter__()
        made.append(c)
        return c, ex

    yield make
    for c in made:
        c.__exit__(None, None, None)


def test_api_queues_with_202_and_refuses_with_503_retry_after(dial_client):
    c, ex = dial_client(max_queued_games=1)
    h, _ = _guest(c)
    body = {"variant": "8p", "models": ["mock"], "human_seats": [3], "turn_seconds": 30}
    r = c.post("/api/games", json=body, headers=h)
    assert r.status_code == 201 and r.json()["state"] == "waiting"
    load = c.get("/api/load").json()
    assert {"overloaded", "level", "pressure", "parts", "depth", "inflight", "capacity", "latency_p50_ms",
            "admission"} <= set(load)
    assert load["level"] == "ok"

    ex.heavy = True
    t0 = time.time()
    while not c.get("/api/load").json()["overloaded"]:
        assert time.time() - t0 < 10
        time.sleep(0.05)
    r = c.post("/api/games", json=body, headers=h)
    assert r.status_code == 202, r.text
    q = r.json()
    assert q["state"] == "queued" and q["queue"]["position"] == 1 and q["queue"]["eta_s"] >= 1
    gid = q["id"]
    assert c.get(f"/api/games/{gid}/queue").json()["queue"]["position"] == 1
    listed = {g["id"]: g for g in c.get("/api/games").json()["games"]}
    assert listed[gid]["state"] == "queued" and listed[gid]["queue"]["position"] == 1

    r = c.post("/api/games", json=body, headers=h)          # the line (1) is full
    assert r.status_code == 503
    assert int(r.headers["Retry-After"]) >= 5 and r.json()["retry_after"] == int(r.headers["Retry-After"])
    assert c.get("/readyz").json()["load"]["level"] == "overloaded"   # still ready: running games need it

    ex.heavy = False
    t0 = time.time()
    while c.get(f"/api/games/{gid}/queue").json()["state"] == "queued":
        assert time.time() - t0 < 10
        time.sleep(0.05)
    assert c.get(f"/api/games/{gid}/queue").json()["state"] == "waiting"   # admitted; a person starts it
    assert "wh_admissions_total" in c.get("/metrics").text


# ------------------------------------------------------------------ Redis Streams executor
async def test_streams_load_sample_and_stale_replies_free_the_games_slot(settings, store, redis):
    from werewolf_harness.app.llm.queue import StreamExecutor, Worker

    from ..test_backpressure import _job
    ex = StreamExecutor(redis, settings, per_game_asks=2)
    await ex.start()
    w = Worker(settings, store, redis, None, consumer="w-bp", concurrency=3)
    await w.ensure_groups()
    for seat in range(1, 6):
        ex.submit(_job("nogame", seat=seat), lambda r: None)   # a game the worker will not find: stale
    await asyncio.sleep(0.2)
    s = await ex.load_sample()
    assert s.depth["online"] == 2 and s.depth["held"] == 3     # 2 in the stream, 3 held on the node
    assert s.oldest_wait_s >= 0.1 and s.capacity == 1          # no worker has announced itself yet
    wt = asyncio.create_task(w.run())
    await until(lambda: not ex.held and not ex.out, 10)        # stale replies came back and freed the slots
    assert w.stale == 5
    s = await ex.load_sample()
    assert s.queued == 0 and s.capacity == 3                   # the worker announced its concurrency
    await w.stop()
    wt.cancel()
    await ex.close()
