"""Move finished games out of the hot tables: one packed journal per game.

    python -m werewolf_harness.scripts.archive_games --days 30 [--batch 500] [--max N] [--vacuum]

A finished game is read rarely (its log page, verify_replay) and never written, yet
its ~50-120 journal rows and one agent_state row per model step (each a ~1.5 KB
belief table) sit in the same tables and indexes as the live games. This job packs
every game that ended more than --days ago into game_archive -- the journal as one
jsonb array (TOAST-compressed), model memory cut to each seat's last entry -- and
deletes the hot rows (Store.archive_batch: one statement per batch).

Reads do not change: Store.journal() / journals() / agent_memory() fall back to the
archive, so replay, /api/games/{id}/log and verify_replay work on archived games.

Space: DELETE leaves dead tuples; VACUUM (--vacuum) makes the space reusable by new
games, which is the steady state for a nightly job. Giving it back to the OS needs a
rewrite (VACUUM FULL / pg_repack) once, after the first big archive.

Safe to run while the service is up (SKIP LOCKED; batches are short transactions)
and safe to run twice at once.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

from sqlalchemy import text

from ..app.config import get_settings
from ..app.storage import Store


async def archive(store: Store, days: float, batch: int = 500, max_games: int | None = None,
                  vacuum: bool = False, log=print) -> dict:
    t0, n, batches = time.monotonic(), 0, 0
    while max_games is None or n < max_games:
        ids = await store.archive_batch(days, batch if max_games is None else min(batch, max_games - n))
        if not ids:
            break
        n += len(ids)
        batches += 1
        if batches % 20 == 0:
            log(f"  已归档 {n} 局，{time.monotonic() - t0:.0f}s")
    took = time.monotonic() - t0
    vac = None
    if vacuum and n:
        t1 = time.monotonic()
        async with store.engine.connect() as c:
            c = await c.execution_options(isolation_level="AUTOCOMMIT")
            for tbl in ("journal", "agent_state", "games", "game_archive"):
                await c.execute(text(f"VACUUM (ANALYZE) {tbl}"))
        vac = round(time.monotonic() - t1, 1)
    return {"archived": n, "batches": batches, "seconds": round(took, 1),
            "games_per_s": round(n / took, 1) if took else None, "vacuum_s": vac}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=float, default=30, help="archive games that ended more than this many days ago")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--max", type=int, default=None, help="stop after this many games")
    ap.add_argument("--vacuum", action="store_true", help="VACUUM (ANALYZE) the touched tables afterwards")
    ap.add_argument("--url", default=None, help="defaults to DATABASE_URL")
    a = ap.parse_args()
    store = Store.connect(a.url or get_settings().database_url, pool_size=2)
    try:
        r = await archive(store, a.days, a.batch, a.max, a.vacuum)
    finally:
        await store.close()
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
