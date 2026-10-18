"""P2: the runtime -- single writer, journal-first, recoverable, fenced."""
from __future__ import annotations

import asyncio

from werewolf_harness.app.runtime.runtime import Runtime
from werewolf_harness.app.runtime.table import PUBLIC_TASKS, Table, TableConfig

from .helpers import CountingExecutor, GatedExecutor, until


def _rt(settings, store, redis, node="node-a", **kw):
    s = settings.model_copy(update={"node_id": node})
    return Runtime(s, store, redis, **kw)


async def test_an_all_ai_game_runs_to_the_end_and_the_database_agrees(settings, store, redis):
    rt = _rt(settings, store, redis)
    await rt.start(scan=False)
    a = await rt.create(TableConfig(variant="8p", models=["mock"]), seed=11)
    await until(lambda: a.t.state == "over", 60)
    rows = await store.journal(a.id)
    assert [(r.actor, r.task) for r in rows] == [(r.actor, r.task) for r in a.t.journal]
    assert [r.seq for r in rows] == list(range(len(rows)))
    assert (await store.get_game(a.id))["state"] == "over"
    # world == f(seed, journal): a table rebuilt from the database is the same world
    t2 = Table(a.id, 11, TableConfig(variant="8p", models=["mock"]))
    Runtime.replay(t2, list(a.t.journal))
    assert t2.fingerprint() == a.t.fingerprint() and t2.turn == a.t.turn
    await rt.shutdown()


async def test_model_memory_is_saved_with_each_step(settings, store, redis):
    rt = _rt(settings, store, redis)
    await rt.start(scan=False)
    a = await rt.create(TableConfig(variant="8p", models=["mock"]), seed=12)
    await until(lambda: len(a.t.journal) > 20, 60)
    mem = await store.agent_memory(a.id)
    assert mem, "model seats wrote no memory"
    seat, (belief, note) = next(iter(mem.items()))
    assert isinstance(belief, dict) and belief
    await rt.shutdown()


async def test_a_human_turn_accepts_once_and_says_why_otherwise(settings, store, redis):
    rt = _rt(settings, store, redis)
    await rt.start(scan=False)
    a = await rt.create(TableConfig(variant="8p", models=["mock"], human_seats=[3], turn_seconds=30), seed=5)
    assert a.t.state == "waiting"
    await a.start()
    await until(lambda: a.t.pending_seat == 3, 60)
    p = a.t.pending
    assert (await a.submit(4, p.seq, p.spec.fallback))["status"] == "not_a_human_seat"
    assert (await a.submit(3, p.seq + 7, p.spec.fallback))["status"] == "stale"
    assert (await a.submit(3, p.seq, {"name": "no_such_tool", "args": {}}))["status"] == "illegal"
    ok = await a.submit(3, p.seq, p.spec.fallback, request_id="click-1")
    assert ok["status"] == "accepted"
    again = await a.submit(3, p.seq, p.spec.fallback, request_id="click-1")      # double click / retried POST
    assert again["status"] == "duplicate" and again["seq"] == p.seq
    assert sum(1 for r in await store.journal(a.id) if r.request_id == "click-1") == 1
    await rt.shutdown()


async def test_a_human_who_runs_out_of_clock_gets_the_fallback(settings, store, redis):
    rt = _rt(settings, store, redis)
    await rt.start(scan=False)
    a = await rt.create(TableConfig(variant="8p", models=["mock"], human_seats=[3], turn_seconds=0.3), seed=5)
    await a.start()
    await until(lambda: any(r.actor == 3 for r in a.t.journal), 60)
    mine = [r for r in a.t.journal if r.actor == 3]
    assert mine[0].forced is True
    await rt.shutdown()


async def test_a_late_model_answer_is_dropped_after_its_deadline(settings, store, redis):
    ex = CountingExecutor(delay=0.8)
    rt = _rt(settings.model_copy(update={"agent_timeout_s": 0.3}), store, redis, executor=ex)
    await rt.start(scan=False)
    a = await rt.create(TableConfig(variant="8p", models=["mock"]), seed=3)
    await until(lambda: a.t.stale_results >= 2, 30)
    assert all(r.forced for r in a.t.journal)      # every step so far was the clock's, not the model's
    versions = [r.seq for r in await store.journal(a.id)]
    assert versions == list(range(len(versions)))   # and nothing was written twice
    await rt.shutdown(drain=False)


async def test_crash_recovery_replays_without_asking_models_again(settings, store, redis):
    """Node A dies mid-game (no drain: its lease just expires). Node B adopts the
    game, and no committed turn is sent to a model a second time."""
    ex_a = CountingExecutor()
    a_rt = _rt(settings, store, redis, "node-a", executor=ex_a)
    await a_rt.start(scan=False)
    game = await a_rt.create(TableConfig(variant="12p", models=["mock"]), seed=21)
    await until(lambda: len(game.t.journal) >= 30, 60)
    # hard crash: stop the loops without touching the lease or the database
    for t in a_rt._tasks:
        t.cancel()
    game.task.cancel()
    crashed_at = len(game.t.journal)
    memory_before = await store.agent_memory(game.id)

    ex_b = CountingExecutor()
    b_rt = _rt(settings, store, redis, "node-b", executor=ex_b)
    await b_rt.start(scan=True)
    await until(lambda: game.id in b_rt.actors, 10)          # after the lease TTL
    b = b_rt.actors[game.id]
    assert b.t.memory == memory_before                          # the AI is not amnesiac
    assert b.t.recoveries == 1
    await until(lambda: b.t.state == "over", 60)

    replayed = [v for (g, v, _s) in ex_b.jobs if g == game.id and v < crashed_at]
    assert replayed == [], f"committed turns were asked again: {replayed}"
    rows = await store.journal(game.id)
    assert [r.seq for r in rows] == list(range(len(rows)))
    assert (await store.get_game(game.id))["owner_node"] == "node-b"
    await b_rt.shutdown()
    await a_rt.executor.close()


async def test_a_paused_node_that_wakes_up_cannot_write(settings, store, redis):
    """The split-brain case: A is frozen past its lease, B takes over, A wakes up
    holding a model answer. The fencing token makes A's write fail and A stops."""
    gated = GatedExecutor()
    a_rt = _rt(settings, store, redis, "node-a", executor=gated)
    await a_rt.start(scan=False)
    game = await a_rt.create(TableConfig(variant="8p", models=["mock"]), seed=8)
    await until(lambda: len(gated.held) >= 1, 10)        # A has an answer in hand but is "frozen"
    for t in a_rt._tasks:                                   # frozen: stops renewing
        t.cancel()

    b_rt = _rt(settings, store, redis, "node-b", executor=CountingExecutor())
    await b_rt.start(scan=True)
    await until(lambda: game.id in b_rt.actors, 10)
    b = b_rt.actors[game.id]
    await until(lambda: len(b.t.journal) >= 3, 30)

    gated.release()                                         # A wakes up and tries to commit
    await until(lambda: game.stopped is not None, 10)
    assert game.stopped == "fenced"
    assert len(game.t.journal) == 0                         # A's world never advanced
    rows = await store.journal(game.id)
    assert [r.seq for r in rows] == list(range(len(rows)))
    await b_rt.shutdown()
    await a_rt.executor.close()


async def test_drain_hands_games_over_without_waiting_for_the_ttl(settings, store, redis):
    a_rt = _rt(settings.model_copy(update={"lease_ttl_ms": 60_000}), store, redis, "node-a")
    await a_rt.start(scan=False)
    game = await a_rt.create(TableConfig(variant="8p", models=["mock"], human_seats=[1], turn_seconds=None), seed=2)
    await game.start()
    await a_rt.shutdown(drain=True)
    b_rt = _rt(settings, store, redis, "node-b")
    await b_rt.start(scan=True)
    assert game.id in b_rt.actors                           # adopted on the first scan, not after 60 s
    await b_rt.shutdown()


async def test_events_never_name_a_night_actor_or_carry_a_thought(settings, store, redis):
    rt = _rt(settings, store, redis)
    await rt.start(scan=False)
    a = await rt.create(TableConfig(variant="8p", models=["mock"]), seed=13)
    await until(lambda: a.t.state == "over", 60)
    for e in a.t.events:
        d = e.to_dict()
        if d["kind"] in ("phase", "turn") and d.get("task") not in PUBLIC_TASKS and d.get("task") is not None:
            assert d["task"] == "hidden" and d.get("actor") is None, d
        if d["kind"] == "turn" and isinstance(d.get("action"), dict):
            assert "thought" not in (d["action"].get("args") or {}), d
    await rt.shutdown()


async def test_a_stale_request_for_a_finished_game_reads_from_the_database(settings, store, redis):
    rt = _rt(settings, store, redis)
    await rt.start(scan=False)
    a = await rt.create(TableConfig(variant="8p", models=["mock"]), seed=14)
    await until(lambda: a.t.state == "over", 60)
    fp = a.t.fingerprint()
    other = _rt(settings, store, redis, "node-c")
    t = await other.load_readonly(a.id)
    assert t is not None and t.state == "over" and t.fingerprint() == fp
    await rt.shutdown()
    await asyncio.sleep(0)
