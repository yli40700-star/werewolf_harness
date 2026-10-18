"""Numbers for the model layer (docs/numbers.md). Simulated providers; needs Postgres + Redis.

1. priority lanes: 3 tables with a person waiting + 24 offline tables, 4 worker
   slots, 150 ms simulated model latency. Queue wait of the person's turns with
   lanes vs with everything in one FIFO lane.
2. outage drill: 10 games whose model fails 50% of calls (30% timeouts, 20% 5xx),
   with and without a fallback model: games finished, turns decided by the
   harness instead of a model, and fallbacks taken.

    python -m werewolf_harness.scripts.bench_llm
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics as st
import time

from redis.asyncio import Redis

from ..app.config import Settings
from ..app.llm.gateway import SIM_PROFILES, Hub
from ..app.llm.queue import StreamExecutor, Worker
from ..app.runtime.runtime import Runtime
from ..app.runtime.table import TableConfig
from ..app.storage import Store, migrate

DB = os.environ.get("BENCH_DATABASE_URL", "postgresql+asyncpg://postgres@127.0.0.1:5432/wh_bench")
RD = os.environ.get("BENCH_REDIS_URL", "redis://127.0.0.1:6379/13")


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p * len(xs)))], 1) if xs else None


async def fresh():
    await migrate.adowngrade(DB)
    await migrate.aupgrade(DB)
    r = Redis.from_url(RD)
    await r.flushdb()
    return Store.connect(DB), r


async def lanes(priority: bool) -> dict:
    SIM_PROFILES["sim-bench"] = {"median_ms": 150, "sigma": 0.3}
    store, r = await fresh()
    s = Settings(database_url=DB, redis_url=RD, node_id="bench", agent_timeout_s=120)
    rt = Runtime(s, store, r, executor=StreamExecutor(r, s))
    await rt.start(scan=False)
    w = Worker(s, store, r, Hub(RD, default_tpm=10**9), consumer="w", concurrency=4)
    wt = asyncio.create_task(w.run())
    human = [await rt.create(TableConfig(variant="8p", models=["sim-bench"], lane="human" if priority else "online"),
                             seed=100 + i) for i in range(3)]
    offline = [await rt.create(TableConfig(variant="8p", models=["sim-bench"], lane="offline" if priority else "online"),
                               seed=200 + i) for i in range(24)]
    t0 = time.monotonic()
    while time.monotonic() - t0 < 25 and not all(g.t.state == "over" for g in human):
        await asyncio.sleep(0.2)
    # with lanes: the person's turns only; one FIFO lane: every turn waits alike, the person's included
    waits = w.waits["human"] if priority else w.waits["online"]
    res = {"human_wait_p50_ms": pct(waits, 0.5), "human_wait_p95_ms": pct(waits, 0.95),
           "human_tables_finished": sum(g.t.state == "over" for g in human),
           "offline_turns_done": sum(len(g.t.journal) for g in offline)}
    await w.stop()
    wt.cancel()
    await rt.shutdown(drain=False)
    await store.close()
    await r.aclose()
    return res


async def outage(fallback: bool) -> dict:
    store, r = await fresh()
    s = Settings(database_url=DB, redis_url=RD, node_id="bench", agent_timeout_s=60)
    hub = Hub(RD, sim_sleep=False, breaker_kw={"min_calls": 10, "threshold": 0.8, "cooldown_s": 5})
    rt = Runtime(s, store, r, hub=hub)
    await rt.start(scan=False)
    games = [await rt.create(TableConfig(variant="8p", models=["sim-flaky"],
                                         fallbacks=["sim-fast"] if fallback else []), seed=300 + i) for i in range(10)]
    t0 = time.monotonic()
    while time.monotonic() - t0 < 120 and not all(g.t.state in ("over", "aborted", "crashed") for g in games):
        await asyncio.sleep(0.2)
    await rt.flush_calls()
    stats = await store.call_stats()
    res = {"games": len(games), "finished": sum(g.t.state == "over" for g in games),
           "turns": sum(len(g.t.journal) for g in games),
           "turns_by_harness_pct": round(100 * sum(r.forced for g in games for r in g.t.journal)
                                         / max(1, sum(len(g.t.journal) for g in games)), 1),
           "model_requests": stats["calls"], "failed_requests": stats["failed"],
           "answered_by_fallback_model": stats["fallbacks"]}
    await rt.shutdown()
    await store.close()
    await r.aclose()
    return res


async def main() -> None:
    out = {"lanes_priority": await lanes(True), "lanes_single_fifo": await lanes(False),
           "outage_no_fallback": await outage(False), "outage_with_fallback": await outage(True)}
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
    _ = st
