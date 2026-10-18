"""Numbers for streaming drafts (docs/adr/0009). Mock models; needs Postgres + Redis.

    python -m werewolf_harness.scripts.bench_streaming [--speeches 150] [--games 40] [--crashes 10]

1. latency: tables of slow mock seats timed like a real model -- 1 s to the first
   token, 30-60 tokens/s (drawn per seat), ~400-token speeches. For every speech
   that was committed: ms from the moment the turn was asked to the first
   character a client could see (the draft), to the last streamed character, and
   to the commit -- which is when the whole speech appeared before streaming.
   p50 / p95 over all speeches.
2. invariant: many fast mock games with faults injected -- a stream that drops
   mid-speech (a transport retry), a stream that stalls until the deadline cuts it
   (timeout -> retract + cancel), and node crashes mid-draft (the game is adopted
   by another node, the open draft is retracted there and the turn asked again).
   Each game is watched by a spectator and a wolf through the same merge the
   websocket uses (drafts.outbox + DraftMirror); a game passes when no reader saw
   a draft that neither became the journal's text nor was retracted.

Real-model run (the lead): the same latency phase with a real seat, e.g.
    BENCH_MODEL=claude-cli:sonnet python -m werewolf_harness.scripts.bench_streaming --speeches 30 --games 0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics as st
import time

from ..app.config import Settings
from ..app.runtime.drafts import DraftMirror, outbox
from ..app.runtime.runtime import Runtime
from ..app.runtime.table import Table, TableConfig
from ..app.storage import Store
from ..players.providers import base as pbase
from ..players.providers.base import Delta, ProviderError
from ..players.providers.mock import MockClient

DB = os.environ.get("BENCH_DATABASE_URL", "postgresql+asyncpg://postgres@127.0.0.1:5432/wh_bench")
RD = os.environ.get("BENCH_REDIS_URL", "redis://127.0.0.1:6379/13")
MODEL = os.environ.get("BENCH_MODEL", "bench-slow")


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, max(0, int(round(p * len(xs))) - 1))], 1) if xs else float("nan")


async def until(cond, timeout=60.0):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            raise TimeoutError
        await asyncio.sleep(0.02)


class Reader:
    """One client reading the merged event + draft stream, as the websocket sends it."""

    def __init__(self, actor, seat=None):
        self.a, self.seat = actor, seat
        self.mirror = DraftMirror()
        self.cur = {"next": 0, "epoch": None, "dlast": -1, "known": {}}
        self.task = asyncio.create_task(self._run())

    def pump(self):
        for op, body in outbox(self.a.t, self.seat, self.cur):
            if op == "draft":
                self.mirror.apply(body)

    async def _run(self):
        # not just cancel(): on 3.10 asyncio.wait_for can swallow a cancel that lands as the
        # condition it waits on fires, and a bare `while True` would then go round forever
        me = asyncio.current_task()
        while self.task is me:
            await self.a.wait_any(self.cur["next"], self.cur["dlast"], timeout=0.5)
            self.pump()

    def switch(self, actor):
        self.a = actor
        self.cur["known"] = {d.draft_id: len(d.text) for d in self.mirror.drafts.values() if d.state == "open"}
        self.task = asyncio.create_task(self._run())

    def halt(self):
        """Stop reading (a dropped connection). Returns the task to await, if any."""
        t, self.task = self.task, None
        if t is not None:
            t.cancel()
        return t

    async def close(self):
        t = self.halt()
        try:
            if t is not None:
                await t
        except asyncio.CancelledError:
            pass
        self.pump()


class Faulty(MockClient):
    """A streaming mock that sometimes drops mid-speech or stalls until it is cancelled."""

    def __init__(self, seed, fail=0.0, stall=0.0, **kw):
        super().__init__(seed=seed, **kw)
        self.fail, self.stall = fail, stall
        self.rng = random.Random(seed * 31 + 7)

    def chat_stream(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0, cancel=None):
        cancel = cancel or pbase.CancelToken()
        r, n = self.rng.random(), 0
        for item in super().chat_stream(messages, tools, temperature, max_tokens, timeout, cancel=cancel):
            if isinstance(item, Delta):
                n += 1
                if n == 3 and r < self.fail:
                    raise ProviderError("connection dropped mid-stream", status=502)
                if n == 3 and r < self.fail + self.stall:
                    cancel.wait(60)
                    cancel.raise_if(n)
            yield item


def install_clients(fail: float, stall: float) -> None:
    orig = Table.client_for

    def client_for(self, model, seat):
        if model == "bench-slow":
            rng = random.Random(self.seed * 1000 + seat)
            return MockClient(seed=self.seed * 100 + seat, name=model, ttft_s=1.0,
                              tokens_per_s=rng.uniform(30.0, 60.0), chunk_tokens=4, speech_tokens=400)
        if model == "bench-faulty":
            return Faulty(self.seed * 100 + seat, fail=fail, stall=stall, ttft_s=0.0, tokens_per_s=2000.0,
                          chunk_tokens=3, speech_tokens=60, name=model)
        return orig(self, model, seat)
    Table.client_for = client_for


def settings(**kw) -> Settings:
    base = dict(database_url=DB, redis_url=RD, jwt_secret="bench-secret-that-is-at-least-32-bytes-long",
                lease_ttl_ms=1500, lease_renew_s=0.3, recovery_scan_s=0.3, agent_timeout_s=120.0)
    return Settings(**{**base, **kw})


# ---------------------------------------------------------------- 1. latency
async def latency(store, r, speeches: int, tables: int) -> dict:
    rt = Runtime(settings(node_id="lat"), store, r)
    await rt.start(scan=False)
    games = [await rt.create(TableConfig(variant="8p", models=[MODEL]), seed=500 + i) for i in range(tables)]

    def done():
        n = sum(1 for g in games for x in g.t.draft_timings if x["outcome"] == "committed")
        return n >= speeches or all(g.t.state == "over" for g in games)
    try:
        await until(done, 3600)
    except TimeoutError:
        pass
    await rt.shutdown(drain=False)
    rows = [x for g in games for x in g.t.draft_timings if x["outcome"] == "committed" and x["first_ms"]]
    first, last, full = ([x[k] for x in rows] for k in ("first_ms", "last_ms", "close_ms"))
    gain = [x["close_ms"] - x["first_ms"] for x in rows]
    return {"model": MODEL, "speeches": len(rows), "chars_median": int(st.median([x["chars"] for x in rows])) if rows else 0,
            "first_visible_ms": {"p50": pct(first, .5), "p95": pct(first, .95)},
            "last_streamed_ms": {"p50": pct(last, .5), "p95": pct(last, .95)},
            "full_text_ms_without_streaming": {"p50": pct(full, .5), "p95": pct(full, .95)},
            "earlier_by_ms": {"p50": pct(gain, .5), "p95": pct(gain, .95)}}


# ---------------------------------------------------------------- 2. invariant
def _wolf(g):
    return next(s for s, role in g.t.env._roles.items() if role == "werewolf")


def _tally(readers, out: dict) -> bool:
    ok = True
    for rd in readers:
        v = rd.mirror.violations(rd.a.t.journal)
        if v:
            ok = False
            out["examples"] = (out.get("examples") or []) + v[:2]
        for k, n in rd.mirror.counts().items():
            out["drafts"][k] = out["drafts"].get(k, 0) + n
        for d in rd.mirror.drafts.values():
            if d.reason:
                out["reasons"][d.reason] = out["reasons"].get(d.reason, 0) + 1
    return ok


async def faults(store, r, n: int, out: dict) -> None:
    rt = Runtime(settings(node_id="inv", agent_timeout_s=1.0), store, r)
    await rt.start(scan=False)
    games, readers = [], {}
    for i in range(n):
        g = await rt.create(TableConfig(variant="8p" if i % 2 else "12p", models=["bench-faulty"]), seed=900 + i)
        games.append(g)
        readers[g.id] = [Reader(g), Reader(g, _wolf(g))]
    try:
        await until(lambda: all(g.t.state == "over" for g in games), 900)
    except TimeoutError:
        pass
    for g in games:
        for rd in readers[g.id]:
            await rd.close()
        out["games"] += 1
        out["passed"] += _tally(readers[g.id], out) and not g.t.drafts and g.t.state == "over"
        for k in ("cancelled_turns", "tokens_saved", "tokens_saved_cut", "tokens_saved_skipped"):
            out["cancel"][k] = out["cancel"].get(k, 0) + g.t.draft_stats[k]
    await rt.shutdown()


async def crash_one(store, r, i: int, out: dict) -> None:
    """A node dies mid-draft without a word; another adopts the game and finishes it."""
    a_rt = Runtime(settings(node_id=f"ca{i}", agent_timeout_s=5.0), store, r)
    await a_rt.start(scan=False)
    g = await a_rt.create(TableConfig(variant="8p", models=["bench-faulty"]), seed=1700 + i)
    readers = [Reader(g), Reader(g, _wolf(g))]
    try:
        await until(lambda: any(len(d.text) > 10 for d in g.t.drafts.values()) or g.t.state == "over", 60)
    except TimeoutError:
        pass
    for rd in readers:
        rd.halt()
        rd.pump()
    for t in a_rt._tasks:
        t.cancel()
    g._close_all = lambda reason: None
    g.task.cancel()
    b_rt = Runtime(settings(node_id=f"cb{i}", agent_timeout_s=5.0), store, r)
    await b_rt.start(scan=True)
    try:
        await until(lambda: g.id in b_rt.actors, 15)
        b = b_rt.actors[g.id]
        for rd in readers:
            rd.switch(b)
        await until(lambda: b.t.state == "over", 120)
    except TimeoutError:
        pass
    for rd in readers:
        await rd.close()
    out["games"] += 1
    out["crashes"] += 1
    over = all(rd.a.t.state == "over" for rd in readers) and all(rd.a is not g for rd in readers)
    out["passed"] += _tally(readers, out) and over
    await b_rt.shutdown()
    await a_rt.executor.close()


async def invariant(store, r, n_games: int, n_crashes: int) -> dict:
    out = {"games": 0, "passed": 0, "crashes": 0, "drafts": {}, "reasons": {}, "cancel": {}}
    if n_games:
        await faults(store, r, n_games, out)
    for i in range(n_crashes):
        await crash_one(store, r, i, out)
    out["pass_rate"] = round(out["passed"] / out["games"], 4) if out["games"] else None
    c = out["cancel"]
    if c.get("cancelled_turns"):
        c["tokens_saved_per_cancel"] = round(c["tokens_saved"] / c["cancelled_turns"], 1)
    return out


async def main() -> None:
    from redis.asyncio import Redis

    from ..app.storage import migrate
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeches", type=int, default=150)
    ap.add_argument("--tables", type=int, default=10)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--crashes", type=int, default=10)
    ap.add_argument("--fail", type=float, default=0.08)
    ap.add_argument("--stall", type=float, default=0.05)
    args = ap.parse_args()
    install_clients(args.fail, args.stall)
    await migrate.adowngrade(DB)
    await migrate.aupgrade(DB)
    store = Store.connect(DB)
    r = Redis.from_url(RD, decode_responses=True)
    await r.flushdb()
    res = {}
    t0 = time.monotonic()
    if args.speeches:
        res["latency"] = await latency(store, r, args.speeches, args.tables)
    t1 = time.monotonic()
    if args.games or args.crashes:
        res["invariant"] = await invariant(store, r, args.games, args.crashes)
    res["wall_s"] = {"latency": round(t1 - t0, 1), "invariant": round(time.monotonic() - t1, 1)}
    print(json.dumps(res, ensure_ascii=False, indent=2))
    await store.close()
    await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
