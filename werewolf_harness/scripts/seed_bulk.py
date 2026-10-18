"""Fill a database with a realistic history: 100k finished games in minutes.

Playing 100k games through the engine one by one would take hours, and the point
is the *database* at that size, not the games. So:

  1. pool   play a pool of real games (mock models) through the real Runtime into
            the target database -- their journals, seats and agent memory are what
            the service itself writes, so every clone replays to the recorded outcome
  2. clone  stamp each pool game out many times with a new id, a new created_at
            (spread over --days, slowly growing traffic), its own models, human seats
            claimed by users (Zipf: a few heavy players, a long tail), and several
            llm_calls per model step; the seed stays the recorded one (world =
            f(seed, journal), so the seed travels with the journal)
  3. load   binary COPY (asyncpg copy_records_to_table), indexes built once at the end
            would be faster still but the point is to measure the schema as shipped,
            so the shipped indexes stay in place while loading

A few hundred games are left `running` (journal cut part-way, no outcome, nobody
holding the lease) and some `waiting`, so the recovery scan has work at startup.

    python -m werewolf_harness.scripts.seed_bulk --url postgresql://postgres@127.0.0.1:5432/wh_scale \
        --redis redis://127.0.0.1:6379/9 --games 100000 --running 300 [--rev head]
"""
from __future__ import annotations

import argparse
import asyncio
import bisect
import datetime as dt
import json
import random
import time

import asyncpg

MODELS = ["deepseek-chat", "gpt-4o-mini", "claude-sonnet-4", "qwen-plus", "sim", "sim-strong", "sim-cheap"]
PROVIDERS = {"deepseek-chat": "deepseek", "gpt-4o-mini": "openai", "claude-sonnet-4": "anthropic",
             "qwen-plus": "dashscope", "sim": "sim", "sim-strong": "sim", "sim-cheap": "sim", "mock": "mock"}
PRICE_IN = {"deepseek-chat": 0.27e-6, "gpt-4o-mini": 0.15e-6, "claude-sonnet-4": 3e-6, "qwen-plus": 0.4e-6,
            "sim": 0.0, "sim-strong": 0.0, "sim-cheap": 0.0, "mock": 0.0}


def asyncpg_url(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://")


def sa_url(url: str) -> str:
    return asyncpg_url(url).replace("postgresql://", "postgresql+asyncpg://", 1)


def gid_for(i: int, salt: int = 0) -> str:
    """Unique 9-char ids like the runtime's ("g" + 8 hex): an odd multiplier is a
    bijection mod 2**32, so distinct i never collide."""
    return "g%08x" % (((i + salt * 1_000_003) * 2654435761 + 0x5bd1e995) % 2**32)


# ------------------------------------------------------------------ 1. pool
async def play_pool(url: str, redis_url: str, n8: int, n12: int) -> list[str]:
    """Real games, written by the real runtime. Returns their ids."""
    from redis.asyncio import Redis

    from ..app.config import Settings
    from ..app.runtime.runtime import Runtime
    from ..app.runtime.table import TableConfig
    from ..app.storage import Store

    store = Store.connect(sa_url(url), pool_size=5)
    redis = Redis.from_url(redis_url)
    s = Settings(database_url=sa_url(url), redis_url=redis_url, node_id="seed-pool", agent_timeout_s=30)
    rt = Runtime(s, store, redis)
    await rt.start(scan=False)
    actors = []
    seeds = iter(range(9001, 99999))
    for variant, n in (("8p", n8), ("12p", n12)):
        for _ in range(n):
            actors.append(await rt.create(TableConfig(variant=variant, models=["mock"]), seed=next(seeds)))
            while sum(a.t.state not in ("over", "aborted", "crashed") for a in actors) >= 8:
                await asyncio.sleep(0.05)
    t0 = time.monotonic()
    while any(a.t.state not in ("over", "aborted", "crashed") for a in actors):
        if time.monotonic() - t0 > 600:
            raise RuntimeError("pool games did not finish")
        await asyncio.sleep(0.1)
    await asyncio.sleep(0.5)          # the last commits and the final set_state
    await rt.shutdown(drain=True)
    await store.close()
    await redis.aclose()
    return [a.id for a in actors if a.t.state == "over"]


async def load_pool(c: asyncpg.Connection, ids: list[str]) -> list[dict]:
    pool = []
    for gid in ids:
        g = await c.fetchrow("SELECT variant, seed, config, outcome FROM games WHERE id = $1", gid)
        j = await c.fetch("SELECT seq, actor, task, action, forced FROM journal WHERE game_id = $1 ORDER BY seq", gid)
        a = await c.fetch("SELECT seat, seq, belief, note FROM agent_state WHERE game_id = $1", gid)
        seats = await c.fetch("SELECT seat, kind FROM game_seats WHERE game_id = $1 ORDER BY seat", gid)
        pool.append({"variant": g["variant"], "seed": g["seed"], "config": json.loads(g["config"]),
                     "outcome": g["outcome"], "journal": [tuple(r) for r in j],
                     "agent": [tuple(r) for r in a], "n": len(seats)})
    return pool


# ------------------------------------------------------------------ 2. clone
class Clock:
    """created_at over the last `days`, traffic growing ~3x from the first day to the last."""

    def __init__(self, rng: random.Random, days: int, now: dt.datetime):
        self.rng, self.days, self.now = rng, days, now

    def created(self) -> dt.datetime:
        u = self.rng.random()
        x = (1 + 8 * u) ** 0.5 / 2 - 0.5          # inverse CDF of density ~ (1 + 2x) on [0, 1]
        return self.now - dt.timedelta(days=self.days * (1 - x), seconds=self.rng.random() * 3600)


class Zipf:
    def __init__(self, rng: random.Random, n: int, s: float = 0.9):
        self.rng = rng
        acc, self.cdf = 0.0, []
        for k in range(1, n + 1):
            acc += 1 / k ** s
            self.cdf.append(acc)

    def draw(self) -> int:
        return bisect.bisect_left(self.cdf, self.rng.random() * self.cdf[-1])


def _calls_for_step(rng, gid, seat, seq, model, at):
    """One decision call, a note-to-self on about half the steps, rare failures + fallback."""
    prov = PROVIDERS[model]
    out = []
    inp = rng.randint(2500, 9000)
    cached = int(inp * rng.uniform(0.3, 0.85))
    outp = rng.randint(40, 400)
    if rng.random() < 0.03:
        out.append((gid, seat, seq, model, prov, rng.choice(["error", "timeout"]), inp, 0, 0,
                    rng.randint(3000, 60000), 0.0, None, at))
        fb = "sim-cheap"
        out.append((gid, seat, seq, fb, "sim", "ok", inp, 0, outp, rng.randint(200, 2000), 0.0, "timeout", at))
    else:
        cost = round((inp - cached) * PRICE_IN[model] + cached * PRICE_IN[model] * 0.1 + outp * PRICE_IN[model] * 4, 6)
        out.append((gid, seat, seq, model, prov, "ok", inp, cached, outp, rng.randint(300, 9000), cost, None, at))
    if rng.random() < 0.5:
        out.append((gid, seat, seq, model, prov, "ok", rng.randint(800, 2500), 0, rng.randint(30, 150),
                    rng.randint(200, 3000), 0.0, None, at + dt.timedelta(seconds=1)))
    return out


def clone(i: int, src: dict, rng: random.Random, clock: Clock, users: list[str], zipf: Zipf,
          state: str = "over", cut: int | None = None):
    """Rows for one cloned game. `cut` keeps only the first `cut` steps (a live game)."""
    gid = gid_for(i)
    created = clock.created() if state == "over" else clock.now - dt.timedelta(minutes=rng.randint(1, 90))
    n = src["n"]
    humans = []
    r = rng.random()
    if r < 0.30:
        humans = rng.sample(range(1, n + 1), 1 if r < 0.24 else 2)
    if state == "waiting":
        humans = humans or [rng.randint(1, n)]
    # live games keep mock seats so a node that adopts them can actually drive them
    models = rng.sample(MODELS, 1 if rng.random() < 0.7 else rng.randint(2, 3)) if state == "over" else ["mock"]
    cfg = dict(src["config"], models=models, human_seats=sorted(humans))
    rota = list(models)
    random.Random(src["seed"]).shuffle(rota)
    ai = [s for s in range(1, n + 1) if s not in humans]
    seat_model = {s: rota[k % len(rota)] for k, s in enumerate(ai)}
    claimed = {h: users[zipf.draw()] for h in humans}
    creator = claimed[humans[0]] if humans else users[zipf.draw()]

    steps = src["journal"] if cut is None else src["journal"][:cut]
    if state == "waiting":
        steps = []
    jrows, crows, at = [], [], created
    for seq, actor, task, action, forced in steps:
        at = at + dt.timedelta(seconds=rng.uniform(1.5, 25))
        jrows.append((gid, seq, actor, task, action, forced, None, at))
        if actor in seat_model and not forced:
            crows.extend(_calls_for_step(rng, gid, actor, seq, seat_model[actor], at))
    last = len(steps)
    arows = [(gid, seat, seq, belief, note) for seat, seq, belief, note in src["agent"]
             if seq < last and seat in seat_model]
    ended = at + dt.timedelta(seconds=2) if state == "over" else None
    game = (gid, src["variant"], src["seed"], json.dumps(cfg, ensure_ascii=False), state,
            "node-1" if state == "over" else None, 1 if state == "over" else rng.randint(1, 3),
            src["outcome"] if state == "over" else None, None, creator, None, created, ended)
    srows = [(gid, s, "human" if s in humans else "llm", None if s in humans else seat_model[s], claimed.get(s))
             for s in range(1, n + 1)]
    return game, srows, jrows, arows, crows


# ------------------------------------------------------------------ 3. load
GAME_COLS = ["id", "variant", "seed", "config", "state", "owner_node", "fencing_token", "outcome", "error",
             "created_by", "resumed_from", "created_at", "ended_at"]
SEAT_COLS = ["game_id", "seat", "kind", "model", "user_id"]
JOURNAL_COLS = ["game_id", "seq", "actor", "task", "action", "forced", "request_id", "created_at"]
AGENT_COLS = ["game_id", "seat", "seq", "belief", "note"]
CALL_COLS = ["game_id", "seat", "seq", "model", "provider", "status", "input_tokens", "cached_tokens", "output_tokens",
             "latency_ms", "cost_usd", "fallback_reason", "created_at"]


async def _jsonb_as_text(c: asyncpg.Connection) -> None:
    """Keep jsonb as the JSON text the runtime wrote (binary COPY: a version byte, then the text)."""
    await c.set_type_codec("jsonb", encoder=lambda s: b"\x01" + s.encode(), decoder=lambda b: b[1:].decode(),
                           schema="pg_catalog", format="binary")


async def seed(url: str, redis_url: str, *, games: int, running: int, waiting: int, users: int, pool8: int,
               pool12: int, days: int, rng_seed: int, rev: str, min_steps: int = 0, batch: int = 2000) -> dict:
    from ..app.storage import migrate
    t_all = time.monotonic()
    await asyncio.to_thread(migrate.upgrade, sa_url(url), rev)
    c = await asyncpg.connect(asyncpg_url(url))
    await _jsonb_as_text(c)
    await c.execute("TRUNCATE llm_calls, agent_state, journal, game_seats, games, users CASCADE")
    for extra in ("game_archive",):
        if await c.fetchval("SELECT to_regclass($1)", extra):
            await c.execute(f"TRUNCATE {extra}")

    t = time.monotonic()
    # play extra 8p games and keep the longer ones: mock games often end in a quick
    # wolf win (~37 steps); --min-steps keeps the pool at a real game's length
    ids = await play_pool(url, redis_url, pool8 * (3 if min_steps else 1), pool12)
    pool = await load_pool(c, ids)
    long8 = [p for p in pool if p["variant"] == "8p" and len(p["journal"]) >= min_steps][:pool8]
    pool = long8 + [p for p in pool if p["variant"] != "8p"]
    await c.execute("TRUNCATE llm_calls, agent_state, journal, game_seats, games, users CASCADE")
    t_pool = time.monotonic() - t
    print(f"pool: {len(pool)} games played in {t_pool:.1f}s "
          f"(8p steps avg {sum(len(p['journal']) for p in pool if p['variant'] == '8p') / max(1, len(long8)):.0f})",
          flush=True)

    rng = random.Random(rng_seed)
    now = dt.datetime.now(dt.timezone.utc)
    clock = Clock(rng, days, now)
    uids = [f"u_{rng.getrandbits(48):012x}" for _ in range(users)]
    rng.shuffle(uids)
    await c.copy_records_to_table("users", columns=["id", "kind", "name", "created_at"], records=[
        (u, "guest" if k % 5 else "oauth", f"玩家{k}", now - dt.timedelta(days=days * rng.random())) for k, u in enumerate(uids)])
    zipf = Zipf(rng, users)
    pool8s = [p for p in pool if p["variant"] == "8p"]
    pool12s = [p for p in pool if p["variant"] == "12p"]
    counts = dict.fromkeys(("games", "game_seats", "journal", "agent_state", "llm_calls"), 0)
    t = time.monotonic()
    total = games + running + waiting
    buf: dict[str, list] = {k: [] for k in counts}

    async def flush():
        for table, cols, key in (("games", GAME_COLS, "games"), ("game_seats", SEAT_COLS, "game_seats"),
                                 ("journal", JOURNAL_COLS, "journal"), ("agent_state", AGENT_COLS, "agent_state"),
                                 ("llm_calls", CALL_COLS, "llm_calls")):
            if buf[key]:
                await c.copy_records_to_table(table, columns=cols, records=buf[key])
                counts[key] += len(buf[key])
                buf[key].clear()

    for i in range(total):
        src = rng.choice(pool12s if (pool12s and rng.random() < 0.2) else pool8s)
        if i < games:
            rows = clone(i, src, rng, clock, uids, zipf)
        elif i < games + running:
            rows = clone(i, src, rng, clock, uids, zipf, state="running",
                         cut=rng.randint(1, len(src["journal"]) - 1))
        else:
            rows = clone(i, src, rng, clock, uids, zipf, state="waiting", cut=0)
        for key, r in zip(("games", "game_seats", "journal", "agent_state", "llm_calls"), rows, strict=True):
            if key == "games":
                buf[key].append(r)
            else:
                buf[key].extend(r)
        if (i + 1) % batch == 0:
            await flush()
            if (i + 1) % (batch * 10) == 0:
                el = time.monotonic() - t
                print(f"  {i + 1}/{total} games, {counts['journal']} steps, {el:.0f}s", flush=True)
    await flush()
    t_load = time.monotonic() - t
    await c.execute("ANALYZE")
    size = await c.fetchval("SELECT pg_size_pretty(pg_database_size(current_database()))")
    await c.close()
    return {"pool_games": len(pool), "pool_s": round(t_pool, 1), "load_s": round(t_load, 1),
            "total_s": round(time.monotonic() - t_all, 1), "rows": {"users": users, **counts}, "db_size": size}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True, help="postgresql://... (the database is TRUNCATED)")
    ap.add_argument("--redis", default="redis://127.0.0.1:6379/9", help="for the pool games' leases")
    ap.add_argument("--games", type=int, default=100_000, help="finished games")
    ap.add_argument("--running", type=int, default=300, help="live games nobody holds (recovery scan work)")
    ap.add_argument("--waiting", type=int, default=60, help="games waiting for people")
    ap.add_argument("--users", type=int, default=20_000)
    ap.add_argument("--pool8", type=int, default=40, help="real 8-player games to clone from")
    ap.add_argument("--pool12", type=int, default=10, help="real 12-player games to clone from")
    ap.add_argument("--min-steps", type=int, default=45, help="keep 8p pool games at least this long")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--rev", default="head", help="migrate to this revision first (e.g. 0001 for the 'before' schema)")
    a = ap.parse_args()
    r = asyncio.run(seed(a.url, a.redis, games=a.games, running=a.running, waiting=a.waiting, users=a.users,
                         pool8=a.pool8, pool12=a.pool12, days=a.days, rng_seed=a.seed, rev=a.rev,
                         min_steps=a.min_steps))
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
