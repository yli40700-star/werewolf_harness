"""P4: model turns through Redis Streams and separate workers."""
from __future__ import annotations

import asyncio
import json
import time

from werewolf_harness.app.llm.gateway import Hub
from werewolf_harness.app.llm.queue import GROUP, StreamExecutor, Worker, jobs_key
from werewolf_harness.app.runtime.table import TableConfig

from .conftest import REDIS_URL
from .helpers import until
from .test_runtime import _rt


async def test_a_game_played_by_a_separate_worker(settings, store, redis):
    ex = StreamExecutor(redis, settings)
    rt = _rt(settings, store, redis, executor=ex)
    await rt.start(scan=False)
    w = Worker(settings, store, redis, Hub(REDIS_URL, sim_sleep=False), consumer="w1", concurrency=4)
    wt = asyncio.create_task(w.run())
    a = await rt.create(TableConfig(variant="8p", models=["mock"]), seed=51)
    await until(lambda: a.t.state == "over", 90)
    assert w.done == sum(1 for r in a.t.journal if a.t.seat_kind(r.actor) != "human")
    assert a.t.memory                                   # memory made the round trip through the queue
    assert a.t.traces                                    # and the ReAct trace came back with the result
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
    w1 = Worker(settings, store, redis, None, consumer="w1")
    await w1.ensure_groups()
    got = await redis.xreadgroup(GROUP, "w1", {jobs_key("online"): ">"}, count=1)   # take it... and die
    assert got
    w2 = Worker(settings, store, redis, Hub(REDIS_URL, sim_sleep=False), consumer="w2", claim_idle_ms=500)
    t2 = asyncio.create_task(w2.run())
    await until(lambda: a.t.state == "over", 90)
    assert len(a.t.journal) > 1 and not a.t.journal[0].forced   # the first turn was the model's, not the clock's
    await w2.stop()
    t2.cancel()
    await rt.shutdown()


async def test_a_waiting_person_goes_before_offline_work(settings, store, redis):
    w = Worker(settings, store, redis, None, consumer="w")
    await w.ensure_groups()
    for i in range(5):
        await redis.xadd(jobs_key("offline"), {"game_id": f"x{i}", "seat": 1, "version": 0, "task": "speak",
                                               "memory": "", "reply_to": "n", "enq": time.time()})
    await redis.xadd(jobs_key("human"), {"game_id": "h", "seat": 1, "version": 0, "task": "speak",
                                         "memory": "", "reply_to": "n", "enq": time.time()})
    first = await w._next()
    assert first[0] == "human"


async def test_a_job_for_a_world_that_moved_on_is_dropped(settings, store, redis):
    rt = _rt(settings, store, redis)
    await rt.start(scan=False)
    a = await rt.create(TableConfig(variant="8p", models=["mock"]), seed=53)
    await until(lambda: len(a.t.journal) >= 5, 30)
    w = Worker(settings, store, redis, None, consumer="w")
    reply = await w._run({"game_id": a.id, "seat": "1", "version": "0", "task": "night", "memory": "",
                          "reply_to": "n", "enq": str(time.time())})
    assert reply is None and w.stale == 1
    await rt.shutdown()
    assert json.dumps({})
