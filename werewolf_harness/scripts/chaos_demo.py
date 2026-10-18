"""kill -9 a node in the middle of a game and watch another node pick it up.

Starts two API processes sharing Postgres and Redis, opens an all-AI 12-seat
table on node 1 (simulated models, ~300 ms per call), lets it play, kills node 1
with SIGKILL (no drain, no goodbye), and times how node 2 adopts the game.

    python -m werewolf_harness.scripts.chaos_demo

Checks at the end:
  * the journal has no gap and no duplicate (seq 0..n-1)
  * no model call was made for an already-committed turn after the kill
  * the game finishes on node 2
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

DB = os.environ.get("CHAOS_DATABASE_URL", "postgresql+asyncpg://postgres@127.0.0.1:5432/wh_bench")
RD = os.environ.get("CHAOS_REDIS_URL", "redis://127.0.0.1:6379/12")
TTL_MS = 5000


def node(n: int, port: int) -> subprocess.Popen:
    env = {**os.environ, "DATABASE_URL": DB, "REDIS_URL": RD, "NODE_ID": f"node-{n}", "LEASE_TTL_MS": str(TTL_MS),
           "LEASE_RENEW_S": "1", "RECOVERY_SCAN_S": "1", "JWT_SECRET": "chaos-demo-secret-chaos-demo-secret!",
           "RATE_LIMIT_PER_MIN": "100000", "LOG_LEVEL": "WARNING"}
    return subprocess.Popen([sys.executable, "-m", "uvicorn", "werewolf_harness.app.api.main:app", "--port", str(port),
                             "--log-level", "warning"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def call(port: int, path: str, body: dict | None = None, token: str | None = None) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def wait_ready(port: int, timeout: float = 30) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if call(port, "/readyz").get("ok"):
                return
        except Exception:                          # noqa: BLE001
            time.sleep(0.2)
    raise SystemExit(f"node on {port} did not come up")


async def reset() -> None:
    from redis.asyncio import Redis

    from ..app.storage import migrate
    await migrate.adowngrade(DB)
    await migrate.aupgrade(DB)
    r = Redis.from_url(RD)
    await r.flushdb()
    await r.aclose()


async def audit(gid: str, crashed_at: int, t_kill: float) -> dict:
    from sqlalchemy import text

    from ..app.storage import Store
    s = Store.connect(DB)
    rows = await s.journal(gid)
    async with s.engine.connect() as c:
        again = await c.scalar(text("SELECT count(*) FROM llm_calls WHERE game_id = :g AND seq < :v "
                                    "AND created_at > to_timestamp(:t)"), {"g": gid, "v": crashed_at, "t": t_kill})
        owner = await c.scalar(text("SELECT owner_node FROM games WHERE id = :g"), {"g": gid})
    await s.close()
    return {"journal_rows": len(rows), "seq_contiguous": [r.seq for r in rows] == list(range(len(rows))),
            "model_calls_for_committed_turns_after_kill": int(again or 0), "owner_now": owner}


def main() -> None:
    asyncio.run(reset())
    n1, n2 = node(1, 8411), node(2, 8412)
    try:
        wait_ready(8411)
        wait_ready(8412)
        tok = call(8411, "/api/auth/guest", {})["token"]
        g = call(8411, "/api/games", {"variant": "12p", "models": ["sim-cheap"], "human_seats": [], "seed": 77}, tok)
        gid = g["id"]
        print(f"game {gid} on node-1")
        while call(8411, f"/api/games/{gid}")["game"]["version"] < 20:
            time.sleep(0.2)
        crashed_at = call(8411, f"/api/games/{gid}")["game"]["version"]
        t_kill = time.time()
        n1.send_signal(signal.SIGKILL)
        n1.wait()
        print(f"t=0.00s  kill -9 node-1 at step {crashed_at}")
        adopted = moved = None
        while True:
            try:
                s = call(8412, f"/api/games/{gid}")["game"]
            except Exception:                      # noqa: BLE001
                time.sleep(0.1)
                continue
            now = time.time() - t_kill
            if adopted is None and s["health"]["recoveries"] >= 1:
                adopted = now
                print(f"t={now:.2f}s  node-2 adopted it (lease TTL {TTL_MS / 1000:.0f}s + scan 1s), at step {s['version']}")
            if adopted is not None and moved is None and s["version"] > crashed_at:
                moved = now
                print(f"t={now:.2f}s  first new step committed by node-2")
            if s["state"] == "over":
                print(f"t={now:.2f}s  game over on node-2 after {s['version']} steps")
                break
            if now > 180:
                raise SystemExit("did not finish")
            time.sleep(0.1)
        res = asyncio.run(audit(gid, crashed_at, t_kill))
        res.update({"crashed_at_step": crashed_at, "adopted_after_s": round(adopted, 2),
                    "first_new_step_after_s": round(moved, 2)})
        print(json.dumps(res, indent=2))
        assert res["seq_contiguous"] and res["model_calls_for_committed_turns_after_kill"] == 0
    finally:
        for p in (n1, n2):
            if p.poll() is None:
                p.terminate()
                p.wait(10)


if __name__ == "__main__":
    main()
