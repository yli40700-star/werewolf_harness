"""Numbers for the runtime (docs/numbers.md). Mock models; needs Postgres + Redis.

    python -m werewolf_harness.scripts.bench_runtime

1. adoption latency: how long a node takes to take over a game (lease + fence +
   read journal + replay + restore memory), at ~30 steps and at game end.
2. idle cost of waiting tables: N games each blocked on a person, old server
   (one thread per game) vs the actor runtime (one coroutine per game).
"""
from __future__ import annotations

import asyncio
import os
import resource
import statistics as st
import threading
import time

from ..app.config import Settings
from ..app.runtime.runtime import Runtime
from ..app.runtime.table import TableConfig
from ..app.storage import Store

DB = os.environ.get("BENCH_DATABASE_URL", "postgresql+asyncpg://postgres@127.0.0.1:5432/wh_bench")
RD = os.environ.get("BENCH_REDIS_URL", "redis://127.0.0.1:6379/13")


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


async def until(cond, timeout=60):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            raise TimeoutError
        await asyncio.sleep(0.01)


async def adoption(samples: int = 20) -> dict:
    from redis.asyncio import Redis
    from sqlalchemy import text

    from ..app.storage import migrate
    await migrate.adowngrade(DB)
    await migrate.aupgrade(DB)
    store = Store.connect(DB)
    r = Redis.from_url(RD)
    await r.flushdb()
    s = Settings(database_url=DB, redis_url=RD, lease_ttl_ms=60_000)
    out = {}
    for label, cut in (("~30 steps", 30), ("finished-length", 10_000)):
        times, steps = [], []
        for i in range(samples):
            a = Runtime(s.model_copy(update={"node_id": "a"}), store, r)
            await a.start(scan=False)
            g = await a.create(TableConfig(variant="12p", models=["mock"]), seed=1000 + i)
            try:
                await until(lambda g=g, cut=cut: len(g.t.journal) >= cut or g.t.state == "over")
            except TimeoutError:
                pass
            await a.shutdown(drain=False)
            async with store.engine.begin() as c:        # pretend it never finished, so it is adoptable
                await c.execute(text("UPDATE games SET state='running' WHERE id=:g"), {"g": g.id})
            await r.delete(f"lease:{g.id}")
            b = Runtime(s.model_copy(update={"node_id": "b"}), store, r)
            b.wheel.start()
            t0 = time.perf_counter()
            actor = await b.adopt(g.id)
            times.append((time.perf_counter() - t0) * 1000)
            steps.append(len(actor.t.journal))
            await b.shutdown(drain=True)
        out[label] = {"p50_ms": round(st.median(times), 1), "p95_ms": round(sorted(times)[int(0.95 * len(times)) - 1], 1),
                      "steps_median": int(st.median(steps)), "samples": samples}
    await store.close()
    await r.aclose()
    return out


def idle_old(n: int) -> dict:
    from ..server.service import GameService
    base_t, base_m = threading.active_count(), rss_mb()
    svc = GameService({})
    games = [svc.create(variant="8p", model="mock", guard="L1", playbook="static", human_seats=[1],
                        turn_seconds=None, seed=i, max_tokens=10**7) for i in range(n)]
    for g in games:
        g.start()
    time.sleep(8)
    waiting = sum(1 for g in games if g.humans[1].pending is not None)
    return {"games": n, "waiting_on_human": waiting, "threads": threading.active_count() - base_t,
            "rss_mb": round(rss_mb() - base_m, 1)}


async def idle_new(n: int) -> dict:
    from redis.asyncio import Redis
    store = Store.connect(DB)
    r = Redis.from_url(RD)
    base_t, base_m = threading.active_count(), rss_mb()
    rt = Runtime(Settings(database_url=DB, redis_url=RD, node_id="idle"), store, r)
    await rt.start(scan=False)
    games = [await rt.create(TableConfig(variant="8p", models=["mock"], human_seats=[1], turn_seconds=None), seed=i)
             for i in range(n)]
    await asyncio.gather(*(g.start() for g in games))
    await asyncio.sleep(8)
    waiting = sum(1 for g in games if g.t.pending_seat == 1)
    res = {"games": n, "waiting_on_human": waiting, "threads": threading.active_count() - base_t,
           "rss_mb": round(rss_mb() - base_m, 1)}
    await rt.shutdown()
    await store.close()
    await r.aclose()
    return res


async def main() -> None:
    import json
    res = {"adoption": await adoption()}
    res["idle_new_500"] = await idle_new(500)
    res["idle_old_500"] = idle_old(500)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
