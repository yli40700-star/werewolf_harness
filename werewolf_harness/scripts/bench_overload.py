"""Overload drill: the provider slows down 10x while new games keep arriving.

    python -m werewolf_harness.scripts.bench_overload            (~90 s, needs Postgres + Redis)
    python -m werewolf_harness.scripts.bench_overload --quick    (~45 s, smaller)

Same scenario three times (docs/adr/0011): a reference run with no new games,
then protection off, then on:

    t = 0          BASE games start (8 seats, all simulated models, 0.2 s per call)
    t = warmup     the simulated provider jumps 10x, to 2 s per call (sim.CONTROL)
    t = warmup..   one new game requested every `every` s, NEW of them
    t = +slow      the provider recovers (0.2 s); queued games should start, in order

    (--fast-ms 300 --slow-ms 3000 plays the 0.3 s -> 3 s jump; same shape, needs
    a longer --slow window to see as many turns, so the default is scaled to fit
    both runs in under two minutes.)

  off   every request starts a game at once; one FIFO in front of the worker pool
  on    admission control (queue, then 503 past max_queued_games) and per-game
        fair queuing in the executor

Measured for the BASE games (the ones already running when the slowdown hit):
turn latency = model job submitted -> answer delivered to the actor, for every
turn asked after the jump. A turn still unanswered at the end counts with its
age so far (a lower bound), so a stalled game cannot hide by not finishing.
Also: commits/s of the BASE games, how many new games started / queued /
were refused, and how long the queued ones waited before they started.

Numbers are simulated (SimClient latency, MockClient decisions) and labelled so.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

from redis.asyncio import Redis

from ..app.config import Settings
from ..app.llm import sim
from ..app.llm.gateway import Hub
from ..app.runtime.admission import Overloaded
from ..app.runtime.agents import LocalExecutor
from ..app.runtime.runtime import Runtime
from ..app.runtime.table import TableConfig
from ..app.storage import Store, migrate

DB = os.environ.get("BENCH_DATABASE_URL", "postgresql+asyncpg://postgres@127.0.0.1:5432/wh_bench_overload")
RD = os.environ.get("BENCH_REDIS_URL", "redis://127.0.0.1:6379/13")
MODEL = "sim-cheap"              # 300 ms median per call (gateway.SIM_PROFILES)


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p * len(xs)))], 2) if xs else None


class TimedExecutor(LocalExecutor):
    """Records, per job, when it was submitted and when its answer came back."""

    def __init__(self, workers: int, fair: bool):
        super().__init__(workers, fair=fair)
        self.asks: list[dict] = []

    def submit(self, job, deliver):
        rec = {"game": job.game_id, "t0": time.monotonic(), "t1": None}
        self.asks.append(rec)

        def wrapped(res, rec=rec):
            rec["t1"] = time.monotonic()
            deliver(res)
        super().submit(job, wrapped)


async def ensure_db() -> None:
    import asyncpg
    name = DB.rsplit("/", 1)[1]
    base = DB.replace("postgresql+asyncpg", "postgresql").rsplit("/", 1)[0] + "/postgres"
    c = await asyncpg.connect(base)
    try:
        if not await c.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name):
            await c.execute(f'CREATE DATABASE "{name}"')
    finally:
        await c.close()


async def fresh():
    await ensure_db()
    await migrate.adowngrade(DB)
    await migrate.aupgrade(DB)
    r = Redis.from_url(RD)
    await r.flushdb()
    return Store.connect(DB), r


async def scenario(label: str, protect: bool, *, base: int, new: int, every: float, warmup: float, slow: float,
                   recover: float, workers: int, fast_ms: float, slow_ms: float) -> dict:
    store, r = await fresh()
    s = Settings(database_url=DB, redis_url=RD, node_id="bench", agent_timeout_s=600, lease_ttl_ms=120_000,
                 admission=protect, fair_queuing=protect, max_queued_games=10, agent_workers=workers)
    ex = TimedExecutor(workers, fair=protect)
    hub = Hub(RD, default_tpm=10**9)
    rt = Runtime(s, store, r, executor=ex, hub=hub)
    await rt.start(scan=False)
    sim.CONTROL.set(median_ms=fast_ms)
    cfg = TableConfig(variant="8p", models=[MODEL])
    base_ids = set()
    for i in range(base):
        _d, a, _q = await rt.request(cfg, seed=100 + i)
        base_ids.add(a.id)
    await asyncio.sleep(warmup)

    sim.CONTROL.set(median_ms=slow_ms)                  # the provider slows down 10x
    t_jump = time.monotonic()
    commits0 = sum(rt.actors[g].stats.commits for g in base_ids)
    outcome = {"admitted": 0, "queued": 0, "rejected": 0}
    queued_ids: list[str] = []
    eta_quoted: list[float] = []
    pressure_trace = []

    async def ramp():
        for i in range(new):
            try:
                d, a, q = await rt.request(cfg, seed=500 + i)
                outcome["admitted" if d == "admit" else "queued"] += 1
                if d != "admit":
                    queued_ids.append(q["game_id"])
                    eta_quoted.append(q["eta_s"])
            except Overloaded:
                outcome["rejected"] += 1
            await asyncio.sleep(every)

    async def watch():
        while True:
            await asyncio.sleep(1.0)
            v = rt.monitor.view()
            pressure_trace.append({"t": round(time.monotonic() - t_jump, 1), "p": v["pressure"],
                                   "over": v["overloaded"], "queued": v["queued_jobs"], "inflight": v["inflight"],
                                   "wait": v["oldest_wait_s"], "exp": v["expected_wait_s"], "turn_ms": v["turn_p50_ms"],
                                   "call_ms": v["latency_p50_ms"], "games": rt.active_games()})
            if os.environ.get("BENCH_TRACE"):
                print(pressure_trace[-1], flush=True)

    def base_commits() -> int:
        return sum(rt.actors[g].stats.commits if g in rt.actors else 0 for g in base_ids)

    tasks = [asyncio.create_task(ramp()), asyncio.create_task(watch())]
    await asyncio.sleep(slow / 2)
    commits_mid = base_commits()
    await asyncio.sleep(slow / 2)
    t_end = time.monotonic()
    commits1 = base_commits()

    def lat(ids):
        return [min(a["t1"] or t_end, t_end) - a["t0"] for a in ex.asks if a["game"] in ids and t_jump <= a["t0"] < t_end]

    base_lat = lat(base_ids)
    unanswered = sum(1 for a in ex.asks if a["game"] in base_ids and t_jump <= a["t0"] < t_end
                     and (a["t1"] is None or a["t1"] > t_end))
    started_new = {a["game"] for a in ex.asks if a["game"] not in base_ids}
    new_lat = lat(started_new)
    # the provider recovers: queued games should now start, oldest first
    sim.CONTROL.set(median_ms=fast_ms)
    await asyncio.sleep(recover)
    for t in tasks:
        t.cancel()
    still_queued = len(rt.admission)
    res = {
        "protection": label,
        "base_turn_p50_s": pct(base_lat, 0.5), "base_turn_p95_s": pct(base_lat, 0.95),
        "base_turns": len(base_lat), "base_unanswered_at_end": unanswered,
        "base_commits_per_s": round((commits1 - commits0) / (t_end - t_jump), 2),
        "base_commits_per_s_2nd_half": round((commits1 - commits_mid) / (slow / 2), 2),
        "new_requested": new, **outcome,
        "queued_then_started": len(rt.admission.waits), "still_queued_after_recovery": still_queued,
        "eta_quoted_p50_s": pct(eta_quoted, 0.5),
        "queue_wait_p50_s": pct(rt.admission.waits, 0.5), "queue_wait_max_s": pct(rt.admission.waits, 1.0),
        "new_games_running": len(started_new), "new_turn_p50_s": pct(new_lat, 0.5),
        "new_turn_p95_s": pct(new_lat, 0.95),
        "overload_flips": rt.monitor.h.flips,
        "pressure_trace": pressure_trace[::max(1, len(pressure_trace) // 8)],
    }
    rt.s = rt.s.model_copy(update={"recovery_scan_s": 999})
    await rt.shutdown(drain=True)
    await store.close()
    await r.aclose()
    sim.CONTROL.set(median_ms=None)
    return res


def table(rows: list[dict]) -> str:
    keys = [("base_turn_p50_s", "running games: turn p50 (s)"), ("base_turn_p95_s", "running games: turn p95 (s)"),
            ("base_unanswered_at_end", "running games: turns still unanswered at end"),
            ("base_commits_per_s", "running games: commits/s (whole slow phase)"),
            ("base_commits_per_s_2nd_half", "running games: commits/s (2nd half, queue built up)"),
            ("new_requested", "new games requested"), ("admitted", "  started at once"),
            ("queued", "  queued"), ("rejected", "  refused (503)"),
            ("queued_then_started", "  queued, then started (incl. recovery)"),
            ("still_queued_after_recovery", "  still queued after recovery"), ("eta_quoted_p50_s", "  ETA quoted when queued, p50 (s)"),
            ("queue_wait_p50_s", "  queue wait p50 (s)"),
            ("queue_wait_max_s", "  queue wait max (s)"), ("new_turn_p50_s", "new games: turn p50 (s)"),
            ("new_turn_p95_s", "new games: turn p95 (s)"), ("overload_flips", "overload state changes")]
    out = ["| metric (simulated provider) | " + " | ".join(r["protection"] for r in rows) + " |",
           "|---|" + "---|" * len(rows)]
    for k, label in keys:
        out.append(f"| {label} | " + " | ".join("—" if r.get(k) is None else str(r.get(k)) for r in rows) + " |")
    return "\n".join(out)


async def main(quick: bool, fast_ms: float | None = None, slow_ms: float | None = None,
               slow: float | None = None) -> None:
    kw = (dict(base=3, new=12, every=1.0, warmup=3, slow=16, recover=6, workers=12, fast_ms=200, slow_ms=2000)
          if quick else
          dict(base=4, new=20, every=1.0, warmup=3, slow=20, recover=14, workers=16, fast_ms=200, slow_ms=2000))
    kw.update({k: v for k, v in (("fast_ms", fast_ms), ("slow_ms", slow_ms), ("slow", slow)) if v})
    rows = []
    # reference: the same running games with nobody arriving -- what "kept their latency" means
    runs = [("no new games (reference)", True, {"new": 0, "recover": 0}),
            ("protection off", False, {}), ("protection on", True, {})]
    for label, protect, over in runs:
        t0 = time.monotonic()
        rows.append(await scenario(label, protect, **{**kw, **over}))
        rows[-1]["wall_s"] = round(time.monotonic() - t0, 1)
    print(json.dumps({"params": kw, "results": rows}, indent=1, ensure_ascii=False))
    print()
    print(table(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fast-ms", type=float)
    ap.add_argument("--slow-ms", type=float)
    ap.add_argument("--slow", type=float, help="seconds the provider stays slow")
    a = ap.parse_args()
    asyncio.run(main(a.quick, a.fast_ms, a.slow_ms, a.slow))
