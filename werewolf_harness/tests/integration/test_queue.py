"""P4: model turns through Redis Streams and separate workers."""
from __future__ import annotations

import asyncio
import time

from sqlalchemy import text

from werewolf_harness.app.llm.gateway import Hub
from werewolf_harness.app.llm.queue import GROUP, StreamExecutor, Worker, jobs_key
from werewolf_harness.app.runtime.table import TableConfig

from .conftest import REDIS_URL
from .helpers import until
from .test_runtime import _rt


async def test_a_game_played_by_a_separate_worker(settings, store, redis):
    """The worker has no Postgres: the turn comes in the job, the model-call records go
    back over Redis and the API side (the runtime's CallRelay) stores them."""
    ex = StreamExecutor(redis, settings)
    rt = _rt(settings, store, redis, executor=ex)
    await rt.start(scan=False)
    w = Worker(settings, redis, Hub(REDIS_URL, sim_sleep=False), consumer="w1", concurrency=4)
    assert not hasattr(w, "store")
    wt = asyncio.create_task(w.run())
    a = await rt.create(TableConfig(variant="8p", models=["mock"]), seed=51)
    await until(lambda: a.t.state == "over", 90)
    assert w.done == sum(1 for r in a.t.journal if a.t.seat_kind(r.actor) != "human")
    assert a.t.memory                                   # memory made the round trip through the queue
    assert a.t.traces                                    # and the ReAct trace came back with the result
    await w.flush_calls()
    await rt.flush_calls()                              # (the relay's own loop may be the one that writes them)
    await until(lambda: rt.relay.written >= w.calls_sent > 0, 10)
    async with store.engine.connect() as c:
        n = await c.scalar(text("SELECT count(*) FROM llm_calls WHERE game_id = :g"), {"g": a.id})
    assert n >= w.done and n == w.calls_sent and rt.relay.written == n
    await w.stop()
    wt.cancel()
    await rt.shutdown()


async def test_a_worker_that_dies_mid_job_loses_nothing(settings, store, redis):
    """w1 takes a job and dies before acknowledging it; w2 claims it after it sits idle."""
    ex = StreamExecutor(redis, settings)
    rt = _rt(settings, store, redis, executor=ex)
    await rt.start(scan=False)
    a = await rt.create(TableConfig(variant="8p", models=["mock"]), seed=52)
    await asyncio.sleep(0.3)
    w1 = Worker(settings, redis, None, consumer="w1")
    await w1.ensure_groups()
    got = await redis.xreadgroup(GROUP, "w1", {jobs_key("online"): ">"}, count=1)   # take it... and die
    assert got
    w2 = Worker(settings, redis, Hub(REDIS_URL, sim_sleep=False), consumer="w2", claim_idle_ms=500)
    t2 = asyncio.create_task(w2.run())
    await until(lambda: a.t.state == "over", 90)
    assert len(a.t.journal) > 1 and not a.t.journal[0].forced   # the first turn was the model's, not the clock's
    await w2.stop()
    t2.cancel()
    await rt.shutdown()


async def test_a_waiting_person_goes_before_offline_work(settings, store, redis):
    w = Worker(settings, redis, None, consumer="w")
    await w.ensure_groups()
    for i in range(5):
        await redis.xadd(jobs_key("offline"), {"game_id": f"x{i}", "seat": 1, "version": 0, "task": "speak",
                                               "memory": "", "reply_to": "n", "enq": time.time()})
    await redis.xadd(jobs_key("human"), {"game_id": "h", "seat": 1, "version": 0, "task": "speak",
                                         "memory": "", "reply_to": "n", "enq": time.time()})
    first = await w._next()
    assert first[0] == "human"


async def test_jobs_of_a_game_that_stopped_are_not_run(settings, store, redis):
    """The worker no longer looks the game up. A game that stops closes its asks
    (llm:closed:{job_id}), so a worker that starts afterwards -- and so never heard the
    cancel on pub/sub -- drops those jobs without calling a model."""
    ex = StreamExecutor(redis, settings)
    rt = _rt(settings, store, redis, executor=ex)
    await rt.start(scan=False)
    a = await rt.create(TableConfig(variant="8p", models=["mock"]), seed=53)
    await until(lambda: ex.out.get(a.id), 10)
    queued = sum([await redis.xlen(jobs_key(lane)) for lane in ("human", "online", "offline")])
    assert queued >= 1
    await rt.shutdown(drain=False)                       # the actor stops: every open ask is cancelled
    await asyncio.sleep(0.2)
    w = Worker(settings, redis, Hub(REDIS_URL, sim_sleep=False), consumer="w")
    wt = asyncio.create_task(w.run())
    await until(lambda: w.stale >= queued, 10)
    assert w.done == 0 and w.calls_sent == 0
    await w.stop()
    wt.cancel()


async def test_a_second_delivery_of_an_answered_job_is_not_run_again(settings, store, redis):
    """At-least-once: the same job twice in the stream. The first run closes it, the copy is dropped."""
    ex = StreamExecutor(redis, settings)
    rt = _rt(settings, store, redis, executor=ex)
    await rt.start(scan=False)
    a = await rt.create(TableConfig(variant="8p", models=["mock"]), seed=54)
    await until(lambda: ex.out.get(a.id), 10)
    eid, fields = (await redis.xrange(jobs_key("online"), count=1))[0]
    w = Worker(settings, redis, Hub(REDIS_URL, sim_sleep=False), consumer="w")
    wt = asyncio.create_task(w.run())
    await until(lambda: w.done >= 1, 30)
    await until(lambda: len(a.t.journal) >= 1, 30)
    done = w.done
    stale = w.stale
    await redis.xadd(jobs_key("online"), fields)          # the first job, delivered again
    await until(lambda: w.stale > stale, 10)
    assert await redis.exists(f"llm:closed:{fields['job_id']}")
    await w.stop()
    wt.cancel()
    await rt.shutdown(drain=False)
    assert w.done >= done
