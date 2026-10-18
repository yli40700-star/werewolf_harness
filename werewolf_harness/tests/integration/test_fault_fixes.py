"""Regressions for what the fault-injection run found (faults/report.md, docs/adr/0013)."""
from __future__ import annotations

import asyncio
import json
import time

import pytest
from sqlalchemy import text
from starlette.websockets import WebSocketDisconnect

from werewolf_harness.app.llm.gateway import CallRecord, Hub
from werewolf_harness.app.llm.queue import GROUP, StreamExecutor, Worker, jobs_key
from werewolf_harness.app.runtime.agents import AgentJob

from .conftest import REDIS_URL
from .helpers import until
from .test_api import _guest, _new_game, client_factory  # noqa: F401, F811 -- the fixture


async def test_a_worker_outlives_a_redis_that_comes_back_empty(settings, store, redis):
    """Redis restarted without persistence: no streams, no consumer group. The worker used
    to die on the first read (NOGROUP / connection reset); it now waits, makes the group
    again and goes on taking jobs."""
    w = Worker(settings, store, redis, None, consumer="w")
    t = asyncio.create_task(w.run())
    await asyncio.sleep(0.3)
    await redis.flushdb()                                    # everything gone, the group too
    await until(lambda: w.redis_errors >= 1, 10)
    await redis.xadd(jobs_key("online"), {"game_id": "gone", "seat": 1, "version": 0, "task": "speak",
                                          "memory": "", "reply_to": "n", "enq": time.time()})
    await until(lambda: w.stale >= 1, 10)                    # read and handled: a job for no game is dropped
    assert not t.done()
    await w.stop()
    t.cancel()


async def test_a_running_job_stays_fresh_so_only_a_dead_workers_job_is_claimed(settings, store, redis):
    w1 = Worker(settings, store, redis, None, consumer="w1", claim_idle_ms=600)
    await w1.ensure_groups()
    await redis.xadd(jobs_key("online"), {"game_id": "g", "seat": 1, "version": 0, "task": "speak", "memory": "",
                                          "reply_to": "n", "enq": time.time()})
    got = await redis.xreadgroup(GROUP, "w1", {jobs_key("online"): ">"}, count=1)
    eid = got[0][1][0][0]
    w1.held[eid] = "online"                                  # w1 is running it
    keep = asyncio.create_task(w1._keepalive_loop())
    await asyncio.sleep(1.5)
    claimed = await redis.xautoclaim(jobs_key("online"), GROUP, "w2", 600, start_id="0-0", count=10)
    assert claimed[1] == []                                  # alive and running: nobody takes it
    keep.cancel()                                            # w1 dies
    await asyncio.sleep(0.8)
    claimed = await redis.xautoclaim(jobs_key("online"), GROUP, "w2", 600, start_id="0-0", count=10)
    assert [e for e, _f in claimed[1]] == [eid]              # silent past the idle time: claimed


def _job(gid: str, seat: int, version: int) -> AgentJob:
    return AgentJob(gid, seat, version, "vote", {}, None, None, make_player=lambda on_step: None)


async def test_cancelling_a_job_gives_its_place_in_the_stream_back(settings, redis):
    """A cancelled job's place was held until the worker's reply; when Redis lost the job
    no reply ever came and the game's next asks waited agent_timeout_s on the node."""
    ex = StreamExecutor(redis, settings, per_game_asks=1)
    a, b = _job("g1", 1, 5), _job("g1", 2, 5)
    ex.submit(a, lambda r: None)
    ex.submit(b, lambda r: None)
    assert list(ex.out["g1"]) == [(1, 5)] and len(ex.held["g1"]) == 1
    ex.cancel(a, "turn_gone")
    assert list(ex.out["g1"]) == [(2, 5)] and "g1" not in ex.held
    await asyncio.sleep(0.05)                                # let the XADD / PUBLISH tasks run


async def test_a_worker_writes_its_model_calls_to_postgres(settings, store, redis):
    hub = Hub(REDIS_URL, sim_sleep=False)
    w = Worker(settings, store, redis, hub, consumer="w", store_calls=True)
    hub.sink.put(CallRecord("gx", 3, 7, "mock", "mock", "ok", input_tokens=10, output_tokens=2, latency_ms=5))
    assert await w.flush_calls() == 1
    async with store.engine.connect() as c:
        n = await c.scalar(text("SELECT count(*) FROM llm_calls WHERE game_id = 'gx' AND seq = 7"))
    assert n == 1


@pytest.mark.timeout(60)
def test_a_websocket_on_a_node_that_lost_the_game_is_closed_so_the_client_moves(client_factory):  # noqa: F811
    c = client_factory()
    h, _ = _guest(c)
    gid = _new_game(c, h, turn_seconds=None)
    rt = c.app.state.svc.rt
    with c.websocket_connect(f"/ws/games/{gid}") as ws:
        assert ws.receive_json()["op"] == "hello"
        c.portal.call(rt.get(gid).stop, "lease_lost")        # another node adopted it
        with pytest.raises(WebSocketDisconnect) as e:
            for _ in range(200):
                ws.receive_json()
        assert e.value.code == 4409


def test_an_online_seat_retries_a_failed_model_call_with_backoff():
    """The gateway tries each model of the chain once; with the seat's retries switched off
    one 503 from the only model made the turn the default action."""
    from werewolf_harness.app.runtime.table import Table, TableConfig
    t = Table("gr", 1, TableConfig(variant="8p", models=["mock"]), hub=Hub(REDIS_URL, sim_sleep=False))
    p = t.make_player(1, task="speak", version=0)
    assert p.policy.max_transport_retries == 2 and p.policy.backoff_s == 1.0


class _FlakyRedis:
    """xadd fails `n` times (a Redis blip), then works."""

    def __init__(self, n: int):
        self.n, self.sent = n, []

    async def xadd(self, key, fields, **kw):
        if self.n > 0:
            self.n -= 1
            raise ConnectionError("Connection reset by peer")
        self.sent.append((key, fields))


async def test_a_job_queued_during_a_redis_blip_is_sent_when_redis_answers_again(settings):
    """The job used to be XADDed fire-and-forget: one failed write and the turn waited for
    its deadline, then took the default action."""
    r = _FlakyRedis(3)
    ex = StreamExecutor(r, settings, per_game_asks=4)
    ex.submit(_job("g2", 1, 3), lambda res: None)
    await until(lambda: len(r.sent) == 1, 10)
    assert ex.send_retries == 3


@pytest.mark.timeout(60)
def test_a_client_that_connects_just_after_the_node_let_go_is_moved_at_once(client_factory):  # noqa: F811
    """The actor stopped (lease lost) but is still listed for a moment: a client that
    connects then used to wait a whole heartbeat (25 s) on a bell nobody rings, long
    enough for the game to end elsewhere."""
    c = client_factory()
    h, _ = _guest(c)
    gid = _new_game(c, h, turn_seconds=None)
    rt = c.app.state.svc.rt
    a = rt.get(gid)
    c.portal.call(a.stop, "lease_lost")
    rt.actors[gid] = a                                       # not yet dropped by the renew loop
    t0 = time.monotonic()
    with c.websocket_connect(f"/ws/games/{gid}") as ws:
        assert ws.receive_json()["op"] == "hello"
        with pytest.raises(WebSocketDisconnect) as e:
            for _ in range(200):
                ws.receive_json()
        assert e.value.code == 4409
    assert time.monotonic() - t0 < 5
    r = c.get(f"/api/games/{gid}/events?drafts=1")           # SSE: ends instead of waiting 15 s
    assert r.status_code == 200 and time.monotonic() - t0 < 10


async def test_the_rate_limit_lets_requests_through_while_redis_is_away():
    """Every /api/ request goes through it: an error here was a 500 for every request
    right after a Redis restart (a pooled connection whose socket had died)."""
    from werewolf_harness.app.api.limits import SlidingWindow

    class _Dead:
        def register_script(self, _src):
            async def run(**_kw):
                raise RuntimeError("unable to perform operation on <TCPTransport closed=True>; the handler is closed")
            return run

    lim = SlidingWindow(_Dead())
    assert await lim.hit("ip:1.2.3.4", 10, 60_000) == (True, 0)
    assert lim.failed_open == 1


async def test_a_dead_lettered_job_tells_its_node(settings, store, redis):
    """A job delivered too often goes to llm:dead. The node used to hear nothing and the
    turn waited out its whole deadline before the default action."""
    from werewolf_harness.app.llm.queue import results_key
    w = Worker(settings, store, redis, None, consumer="w", claim_idle_ms=100, max_deliveries=1)
    await w.ensure_groups()
    await redis.xadd(jobs_key("online"), {"game_id": "gd", "seat": 2, "version": 9, "task": "vote", "memory": "",
                                          "reply_to": "n1", "enq": time.time()})
    await redis.xreadgroup(GROUP, "gone", {jobs_key("online"): ">"}, count=1)   # taken by a worker that died
    await asyncio.sleep(0.3)
    t = asyncio.create_task(w._reclaim_loop())
    await until(lambda: w.dead >= 1, 10)
    t.cancel()
    got = await redis.xrange(results_key("n1"))
    assert [json.loads(f.get(b"r", f.get("r"))) for _e, f in got] == [{"game_id": "gd", "seat": 2, "version": 9, "dead": True}]


async def test_the_node_asks_again_once_when_its_job_was_dead_lettered(settings, redis):
    from werewolf_harness.app.llm.queue import results_key
    ex = StreamExecutor(redis, settings, per_game_asks=4)
    await ex.start()
    job = _job("g3", 4, 7)
    ex.submit(job, lambda r: None)
    dead = {"r": json.dumps({"game_id": "g3", "seat": 4, "version": 7, "dead": True})}
    for _ in range(2):                                       # the second death is not asked again
        await asyncio.sleep(0.2)
        await redis.xadd(results_key(settings.node_id), dead)
    await asyncio.sleep(1.5)
    jobs = await redis.xrange(jobs_key("online"))
    assert len([f for _e, f in jobs if f.get(b"game_id", f.get("game_id")) in (b"g3", "g3")]) == 2
    assert ex.reasks == 1
    await ex.close()
