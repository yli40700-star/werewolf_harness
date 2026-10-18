"""The service's queries at scale: EXPLAIN (ANALYZE, BUFFERS) + wall-clock medians.

Run against a database filled by scripts/seed_bulk.py (never a live one: the
lease/append probes run inside transactions that are rolled back, but the
recovery-scan probe bumps fencing tokens of live games):

    python -m werewolf_harness.scripts.bench_db --url postgresql://postgres@127.0.0.1:5432/wh_scale \
        [--runs 15] [--out before.json] [--set before|after] [--e2e]

Each query is run once on a fresh connection ("first": plan + buffer cold for that
connection; the OS page cache could not be dropped in the container, so this is not
a disk-cold number -- the BUFFERS read/hit counts say how much I/O a truly cold
run would do) and then --runs times warm; the median of the warm runs is reported.

--set picks the SQL: "before" is what the code ran before phase J, "after" what it
runs now. The parameters (which user, which game) are picked once from the data so
both sets see the same inputs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
import time

import asyncpg

LIVE = "SELECT id, state, owner_node, fencing_token FROM games WHERE state IN ('waiting', 'running') ORDER BY created_at"

QUERIES: dict[str, dict[str, str]] = {
    # recovery scan / startup: which games should somebody be driving
    "recovery_scan": {"before": LIVE, "after": LIVE},
    # lease acquisition: fencing bump right after winning the lease
    "lease_bump_fencing": {
        "before": "UPDATE games SET fencing_token = fencing_token + 1, owner_node = 'bench' WHERE id = $1 RETURNING fencing_token",
        "after": "UPDATE games SET fencing_token = fencing_token + 1, owner_node = 'bench' WHERE id = $1 RETURNING fencing_token"},
    # adoption: the journal and the model memory of a live game
    "adopt_journal": {
        "before": "SELECT seq, actor, task, action, forced, request_id FROM journal WHERE game_id = $1 ORDER BY seq",
        "after": "SELECT seq, actor, task, action, forced, request_id FROM journal WHERE game_id = $1 ORDER BY seq"},
    "adopt_agent_memory": {
        "before": "SELECT DISTINCT ON (seat) seat, belief, note FROM agent_state WHERE game_id = $1 ORDER BY seat, seq DESC",
        "after": "SELECT DISTINCT ON (seat) seat, belief, note FROM agent_state WHERE game_id = $1 ORDER BY seat, seq DESC"},
    # the journal write's fencing check (FOR SHARE) -- run with the insert in a rolled-back txn
    "append_fence_check": {
        "before": "SELECT fencing_token FROM games WHERE id = $1 FOR SHARE",
        "after": "SELECT fencing_token FROM games WHERE id = $1 FOR SHARE"},
    "step_for_request": {
        "before": "SELECT seq FROM journal WHERE game_id = $1 AND request_id = $2",
        "after": "SELECT seq FROM journal WHERE game_id = $1 AND request_id = $2"},
    # lobby: newest 50
    "lobby_first_page": {
        "before": "SELECT id, variant, seed, config, state, outcome, error, owner_node, created_at, ended_at "
                  "FROM games ORDER BY created_at DESC LIMIT 50",
        "after": "SELECT id, variant, seed, config, state, outcome, error, owner_node, created_at, ended_at "
                 "FROM games ORDER BY created_at DESC, id DESC LIMIT 50"},
    # lobby page 200 (deep scroll / a crawler): OFFSET before, keyset after
    "lobby_page_200": {
        "before": "SELECT id, variant, seed, config, state, outcome, error, owner_node, created_at, ended_at "
                  "FROM games ORDER BY created_at DESC LIMIT 50 OFFSET 9950",
        "after": "SELECT id, variant, seed, config, state, outcome, error, owner_node, created_at, ended_at "
                 "FROM games WHERE (created_at, id) < ($1, $2) ORDER BY created_at DESC, id DESC LIMIT 50"},
    # "my games": the heaviest user and a typical one
    "user_history_heavy": {
        "before": "SELECT g.id, g.variant, g.state, g.outcome, s.seat, g.created_at FROM game_seats s "
                  "JOIN games g ON g.id = s.game_id WHERE s.user_id = $1 ORDER BY g.created_at DESC LIMIT 20",
        "after": "SELECT g.id, g.variant, g.state, g.outcome, s.seat, s.game_created_at AS created_at FROM game_seats s "
                 "JOIN games g ON g.id = s.game_id WHERE s.user_id = $1 "
                 "ORDER BY s.game_created_at DESC NULLS LAST, s.game_id DESC LIMIT 20"},
    "user_history_typical": {"before": "", "after": ""},     # same SQL, filled below
    # game detail / log: the game row and its whole journal
    "game_detail_row": {"before": "SELECT * FROM games WHERE id = $1", "after": "SELECT * FROM games WHERE id = $1"},
    "game_detail_journal": {
        "before": "SELECT seq, actor, task, action, forced, request_id FROM journal WHERE game_id = $1 ORDER BY seq",
        "after": "SELECT seq, actor, task, action, forced, request_id FROM journal WHERE game_id = $1 ORDER BY seq"},
    # llm usage: one game, and the dashboard (all time before, last 24h after)
    "llm_stats_game": {
        "before": "SELECT count(*), sum(input_tokens), percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) "
                  "FROM llm_calls WHERE game_id = $1",
        "after": "SELECT count(*), sum(input_tokens), percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) "
                 "FROM llm_calls WHERE game_id = $1"},
    "llm_stats_dashboard": {
        "before": "SELECT count(*), sum(input_tokens), sum(cached_tokens), sum(output_tokens), sum(cost_usd), "
                  "percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms), "
                  "count(*) FILTER (WHERE status = 'ok' AND fallback_reason IS NOT NULL), "
                  "count(*) FILTER (WHERE status <> 'ok') FROM llm_calls",
        "after": "SELECT count(*), sum(input_tokens), sum(cached_tokens), sum(output_tokens), sum(cost_usd), "
                 "percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms), "
                 "count(*) FILTER (WHERE status = 'ok' AND fallback_reason IS NOT NULL), "
                 "count(*) FILTER (WHERE status <> 'ok') FROM llm_calls WHERE created_at >= now() - interval '24 hours'"},
    # verify_replay: the newest 500 finished games (filter in Python before, in SQL after)
    "verify_pick_games": {
        "before": "SELECT id, variant, seed, config, state, outcome, error, owner_node, created_at, ended_at "
                  "FROM games ORDER BY created_at DESC LIMIT 500",
        "after": "SELECT id, seed, config, outcome FROM games WHERE state = 'over' "
                 "ORDER BY created_at DESC, id DESC LIMIT 500"},
    # archive job: the next batch of games old enough to pack
    "archive_pick_batch": {
        "before": "SELECT id FROM games WHERE state IN ('over', 'aborted', 'crashed') "
                  "AND ended_at < now() - interval '30 days' ORDER BY ended_at LIMIT 500",
        "after": "SELECT id FROM games WHERE archived_at IS NULL AND state IN ('over', 'aborted', 'crashed') "
                 "AND ended_at < now() - interval '30 days' ORDER BY ended_at LIMIT 500"},
}
QUERIES["user_history_typical"] = QUERIES["user_history_heavy"]
WRITES = {"lease_bump_fencing", "append_fence_check"}


async def pick_params(c: asyncpg.Connection) -> dict:
    """Inputs chosen once from the data, the same for both SQL sets."""
    live = await c.fetchval("SELECT id FROM games WHERE state = 'running' ORDER BY created_at LIMIT 1 OFFSET 100")
    heavy = await c.fetchval("SELECT user_id FROM game_seats WHERE user_id IS NOT NULL "
                             "GROUP BY user_id ORDER BY count(*) DESC LIMIT 1")
    typical = await c.fetchval(
        "SELECT user_id FROM (SELECT user_id, count(*) n FROM game_seats WHERE user_id IS NOT NULL GROUP BY user_id) x "
        "ORDER BY n LIMIT 1 OFFSET (SELECT count(DISTINCT user_id) / 2 FROM game_seats WHERE user_id IS NOT NULL)")
    done = await c.fetchval("SELECT id FROM games g WHERE state = 'over' AND variant = '12p' "
                            "ORDER BY created_at DESC LIMIT 1 OFFSET 777")
    old = await c.fetchval("SELECT id FROM games WHERE state = 'over' AND variant = '12p' "
                           "AND created_at < now() - interval '120 days' LIMIT 1")
    page = await c.fetchrow("SELECT created_at, id FROM games ORDER BY created_at DESC, id DESC LIMIT 1 OFFSET 9949")
    n_heavy = await c.fetchval("SELECT count(*) FROM game_seats WHERE user_id = $1", heavy)
    n_typ = await c.fetchval("SELECT count(*) FROM game_seats WHERE user_id = $1", typical)
    return {"live": live, "heavy": heavy, "typical": typical, "done": done, "old": old,
            "page_ts": page["created_at"], "page_id": page["id"], "n_heavy": n_heavy, "n_typical": n_typ}


def args_for(name: str, which: str, p: dict) -> list:
    return {
        "lease_bump_fencing": [p["live"]], "adopt_journal": [p["live"]], "adopt_agent_memory": [p["live"]],
        "append_fence_check": [p["live"]], "step_for_request": [p["live"], "r-none"],
        "lobby_page_200": [p["page_ts"], p["page_id"]] if which == "after" else [],
        "user_history_heavy": [p["heavy"]], "user_history_typical": [p["typical"]],
        "game_detail_row": [p["old"]], "game_detail_journal": [p["old"]], "llm_stats_game": [p["done"]],
    }.get(name, [])


async def _explain(c: asyncpg.Connection, sql: str, args: list) -> tuple[float, dict]:
    if name_is_write(sql):
        tr = c.transaction()
        await tr.start()
    t0 = time.perf_counter()
    rows = await c.fetch("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql, *args)
    ms = (time.perf_counter() - t0) * 1000
    if name_is_write(sql):
        await tr.rollback()
    plan = rows[0][0]
    plan = json.loads(plan) if isinstance(plan, str) else plan
    return ms, plan[0]


def name_is_write(sql: str) -> bool:
    return sql.lstrip().upper().startswith(("UPDATE", "SELECT FENCING_TOKEN"))


def one_line(node: dict) -> str:
    """The plan's shape in one line: node types (and index names) top-down."""
    out = []

    def walk(n, depth=0):
        label = n["Node Type"]
        if n.get("Index Name"):
            label += f"({n['Index Name']})"
        elif n.get("Relation Name") and "Scan" in label:
            label += f"({n['Relation Name']})"
        out.append(label)
        for ch in n.get("Plans", []):
            walk(ch, depth + 1)

    walk(node)
    return " > ".join(out)


def buffers(node: dict) -> dict:
    return {"hit": node.get("Shared Hit Blocks", 0), "read": node.get("Shared Read Blocks", 0)}


async def bench_sql(url: str, which: str, runs: int, only: list[str] | None) -> dict:
    c = await asyncpg.connect(url)
    p = await pick_params(c)
    await c.close()
    out = {"params": {k: str(v) for k, v in p.items()}, "queries": {}}
    for name, sqls in QUERIES.items():
        if only and name not in only:
            continue
        sql, args = sqls[which], args_for(name, which, p)
        c = await asyncpg.connect(url)                       # fresh connection: the "first" run
        try:
            first_ms, plan = await _explain(c, sql, args)
            first_exec = plan["Execution Time"]
            warm, exec_ms = [], []
            for _ in range(runs):
                ms, plan = await _explain(c, sql, args)
                warm.append(ms)
                exec_ms.append(plan["Execution Time"])
        except asyncpg.PostgresError as e:
            out["queries"][name] = {"error": str(e)}
            await c.close()
            continue
        await c.close()
        out["queries"][name] = {
            "first_exec_ms": round(first_exec, 3), "warm_exec_ms_p50": round(st.median(exec_ms), 3),
            "warm_roundtrip_ms_p50": round(st.median(warm), 3), "plan": one_line(plan["Plan"]),
            "buffers": buffers(plan["Plan"]), "rows": plan["Plan"].get("Actual Rows")}
        print(f"{name:24s} {st.median(exec_ms):9.3f} ms  first {first_exec:9.3f}  {one_line(plan['Plan'])}", flush=True)
    return out


async def sizes(url: str) -> dict:
    c = await asyncpg.connect(url)
    rows = await c.fetch(
        "SELECT c.relname, c.relkind, pg_total_relation_size(c.oid) tot, pg_relation_size(c.oid) heap, "
        "coalesce(pg_total_relation_size(nullif(c.reltoastrelid, 0)), 0) toast, "
        "pg_indexes_size(c.oid) idx, c.reltuples::bigint n FROM pg_class c "
        "WHERE c.relnamespace = 'public'::regnamespace AND c.relkind IN ('r', 'p') ORDER BY 3 DESC")
    idx = await c.fetch("SELECT indexrelid::regclass::text name, pg_relation_size(indexrelid) sz FROM pg_index i "
                        "JOIN pg_class c ON c.oid = i.indrelid WHERE c.relnamespace = 'public'::regnamespace ORDER BY 2 DESC")
    db = await c.fetchval("SELECT pg_database_size(current_database())")
    await c.close()
    mb = lambda b: round(b / 2**20, 1)   # noqa: E731
    return {"db_mb": mb(db),
            "tables": {r["relname"]: {"total_mb": mb(r["tot"]), "heap_mb": mb(r["heap"]), "toast_mb": mb(r["toast"]),
                                      "index_mb": mb(r["idx"]), "rows": r["n"]} for r in rows},
            "indexes": {r["name"]: mb(r["sz"]) for r in idx}}


# ------------------------------------------------------------------ end to end (the service's own code)
async def e2e(url: str, redis_url: str, runs: int) -> dict:
    """What the service does, timed through its own Store/Runtime code paths."""
    from redis.asyncio import Redis

    from ..app.config import Settings
    from ..app.runtime.runtime import Runtime
    from ..app.storage import Store
    from ..arena import Record
    from ..scripts.verify_replay import verify

    sa = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    store = Store.connect(sa, pool_size=10)
    redis = Redis.from_url(redis_url)
    c = await asyncpg.connect(url)
    p = await pick_params(c)
    users = [r[0] for r in await c.fetch(
        "SELECT user_id FROM game_seats WHERE user_id IS NOT NULL GROUP BY user_id ORDER BY random() LIMIT 200")]
    games = [r[0] for r in await c.fetch(
        "SELECT id FROM games WHERE state = 'over' ORDER BY random() LIMIT 200")]
    await c.close()
    res: dict = {}

    def pct(xs, q):
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(q * len(xs)))], 3)

    async def timed(fn, n):
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            await fn()
            out.append((time.perf_counter() - t0) * 1000)
        return out

    lob = await timed(lambda: store.recent_games(50), runs * 4)
    res["lobby_list_ms"] = {"p50": pct(lob, 0.5), "p95": pct(lob, 0.95)}
    uh = []
    for u in users:
        uh += await timed(lambda u=u: store.user_games(u), 1)
    res["user_history_ms"] = {"p50": pct(uh, 0.5), "p95": pct(uh, 0.95), "p99": pct(uh, 0.99), "users": len(users)}
    heavy = await timed(lambda: store.user_games(p["heavy"]), runs)
    res["user_history_heaviest_ms"] = {"p50": pct(heavy, 0.5), "games": p["n_heavy"]}

    rp, load, hot, arch = [], [], [], []
    for gid in games:
        t0 = time.perf_counter()
        row = await store.get_game(gid)
        try:                                                  # the read path load_readonly takes
            rows = await store.journal(gid, archived=row.get("archived_at") is not None)
        except TypeError:                                     # the code before phase J
            rows = await store.journal(gid)
        t1 = time.perf_counter()
        t = Runtime.table_from_row(row)                       # under the game's own engine version
        Runtime.replay(t, [Record(r.actor, r.task, r.action, r.forced) for r in rows])
        t2 = time.perf_counter()
        load.append((t1 - t0) * 1000)
        rp.append((t2 - t0) * 1000)
        (arch if row.get("archived_at") else hot).append(((t1 - t0) * 1000, (t2 - t0) * 1000))
    res["single_game_replay_ms"] = {"load_p50": pct(load, 0.5), "load_p95": pct(load, 0.95),
                                    "total_p50": pct(rp, 0.5), "total_p95": pct(rp, 0.95), "games": len(games)}
    for name, xs in (("hot", hot), ("archived", arch)):
        if xs:
            res["single_game_replay_ms"][name] = {"n": len(xs), "load_p50": pct([a for a, _ in xs], 0.5),
                                                  "total_p50": pct([b for _, b in xs], 0.5)}

    t0 = time.perf_counter()
    v = await verify(store, 500)
    res["verify_replay_500_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    res["verify_replay_500"] = {"checked": v["checked"], "consistent_rate": v["consistent_rate"]}

    # startup: the first recovery scan adopts every live game nobody holds. A gated
    # executor (asks are never answered) keeps the adopted games from writing, so the
    # probe can be repeated on the same data.
    from ..app.runtime.agents import LocalExecutor

    class Never(LocalExecutor):
        def submit(self, job, deliver):
            pass

    scans = []
    for i in range(3):
        await redis.flushdb()
        s = Settings(database_url=sa, redis_url=redis_url, node_id=f"bench-{i}", lease_ttl_ms=60_000)
        rt = Runtime(s, store, redis, executor=Never(1))
        rt.wheel.start()
        t0 = time.perf_counter()
        n = await rt.scan()
        scans.append(((time.perf_counter() - t0) * 1000, n))
        await rt.shutdown(drain=True)
    res["recovery_scan_ms"] = {"p50": round(st.median(x for x, _ in scans), 1), "adopted": scans[0][1],
                               "runs": [round(x, 1) for x, _ in scans]}
    t0 = time.perf_counter()
    for _ in range(runs):
        await store.live_games()
    res["live_games_query_ms"] = round((time.perf_counter() - t0) * 1000 / runs, 3)
    await redis.flushdb()
    await store.close()
    await redis.aclose()
    for k, v in res.items():
        print(f"{k:28s} {v}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--redis", default="redis://127.0.0.1:6379/9")
    ap.add_argument("--set", default="after", choices=["before", "after"])
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--e2e", action="store_true", help="also time the service's own code paths")
    ap.add_argument("--no-sql", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    url = a.url.replace("postgresql+asyncpg://", "postgresql://")
    out = {"set": a.set, "sizes": asyncio.run(sizes(url))}
    if not a.no_sql:
        out.update(asyncio.run(bench_sql(url, a.set, a.runs, a.only)))
    if a.e2e:
        out["e2e"] = asyncio.run(e2e(url, a.redis, a.runs))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
