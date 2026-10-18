"""Determinism check: rebuild every finished game from (seed, journal) and compare.

A game whose replay does not end where the live game ended has a source of
nondeterminism in the engine (dict order, a clock, an unseeded random). Run it
nightly; any mismatch is a bug, not noise.

    python -m werewolf_harness.scripts.verify_replay [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import json

from ..app.config import get_settings
from ..app.runtime.runtime import Runtime
from ..app.storage import Store
from ..arena import Record, Terminal


async def verify(store: Store, limit: int = 500, chunk: int = 100) -> dict:
    """The newest `limit` finished games (picked in SQL -- filtering a page of recent
    games in Python checked fewer than asked once live games crowded the top), with
    their journals fetched `chunk` games per query (archived ones unpacked)."""
    games = await store.finished_games(limit)
    journals: dict = {}
    for i in range(0, len(games), chunk):
        journals.update(await store.journals([g["id"] for g in games[i:i + chunk]]))
    bad = []
    for g in games:
        rows = journals[g["id"]]
        try:
            t = Runtime.table_from_row(g)                 # the engine version the game was created with
            Runtime.replay(t, [Record(r.actor, r.task, r.action, r.forced) for r in rows])
            ok = isinstance(t.turn, Terminal) and json.loads(json.dumps(t.turn.outcome, default=str)) == \
                json.loads(json.dumps(g["outcome"], default=str))
        except Exception as exc:                  # noqa: BLE001
            ok, t = False, None
            bad.append({"id": g["id"], "error": str(exc)})
            continue
        if not ok:
            bad.append({"id": g["id"], "error": "outcome differs"})
    n = len(games)
    return {"checked": n, "mismatches": bad, "consistent_rate": round((n - len(bad)) / n, 4) if n else None}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    a = ap.parse_args()
    store = Store.connect(get_settings().database_url)
    print(json.dumps(await verify(store, a.limit), ensure_ascii=False, indent=2))
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
