"""WebSocket load test: N tables, each with one bot player and 7 simulated-model seats.

Each bot does what a person's browser does: guest login, create a table, claim
its seat, start, hold a WebSocket (heartbeats included), and whenever an event
says its turn may have come, GET its seat state and act through the socket.

Measured client-side:
  push_ms    event created on the server (its ts) -> received by the bot  (same host clock)
  get_ms     GET /api/games/{id}?seat=1
  ack_ms     act sent over the socket -> ack received (includes the journal commit)

    python -m werewolf_harness.loadtest.bots --url http://127.0.0.1:8402 --tables 200 --seconds 60
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics as st
import time
import uuid

import httpx
import websockets


class Stats:
    def __init__(self):
        self.push, self.get, self.ack = [], [], []
        self.errors: dict[str, int] = {}
        self.acts = 0
        self.tables_started = 0
        self.tables_over = 0
        self.queued_polls = 0

    def err(self, k: str) -> None:
        self.errors[k] = self.errors.get(k, 0) + 1


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p * len(xs)))], 1) if xs else None


async def table(i: int, url: str, model: str, stats: Stats, stop: float) -> None:
    async with httpx.AsyncClient(base_url=url, timeout=20) as c:
        try:
            tok = (await c.post("/api/auth/guest", json={})).json()["token"]
            h = {"Authorization": f"Bearer {tok}"}
            resp = await c.post("/api/games", json={"variant": "8p", "models": [model], "human_seats": [1],
                                                    "turn_seconds": 60, "seed": 1000 + i}, headers=h)
            if resp.status_code == 503:              # overload protection refused it (docs/adr/0011)
                stats.err("refused_503")
                return
            g = resp.json()
            gid = g["id"]
            while g.get("state") == "queued" and time.time() < stop:
                stats.queued_polls += 1              # in line: wait until the node starts it
                await asyncio.sleep(1.0)
                g = (await c.get(f"/api/games/{gid}/queue")).json()
            st_ = (await c.post(f"/api/games/{gid}/join", json={"seat": 1}, headers=h)).json()["seat_token"]
            sh = {"Authorization": f"Bearer {st_}"}
            await c.post(f"/api/games/{gid}/start", headers=sh)
            stats.tables_started += 1
        except Exception as e:                      # noqa: BLE001
            stats.err(f"setup:{type(e).__name__}")
            return
        ws_url = url.replace("http", "ws") + f"/ws/games/{gid}?token={st_}"
        try:
            async with websockets.connect(ws_url, max_size=2**22) as ws:
                hello = json.loads(await ws.recv())
                inbox: asyncio.Queue = asyncio.Queue()
                pending: dict[str, float] = {}
                done = asyncio.Event()

                async def reader():
                    # timestamp on arrival, before any of our own work, so push_ms is the server's
                    async for raw in ws:
                        now = time.time()
                        m = json.loads(raw)
                        if m["op"] == "event":
                            stats.push.append((now - m["ts"]) * 1000)
                            if m["kind"] == "end":
                                stats.tables_over += 1
                                done.set()
                                return
                            inbox.put_nowait(m)
                        elif m["op"] == "ack":
                            t0 = pending.pop(m["request_id"], None)
                            if t0:
                                stats.ack.append((time.perf_counter() - t0) * 1000)
                            if m["result"]["status"] == "accepted":
                                stats.acts += 1
                            elif m["result"]["status"] not in ("duplicate", "stale", "not_your_turn"):
                                stats.err(f"act:{m['result']['status']}")

                async def beater():
                    await asyncio.sleep(random.random() * hello["heartbeat_s"])
                    while True:
                        await ws.send(json.dumps({"op": "ping", "t": time.time()}))
                        await asyncio.sleep(hello["heartbeat_s"])

                async def player():
                    last_seq = -1
                    while True:
                        await inbox.get()
                        while not inbox.empty():          # coalesce: only the latest state matters
                            inbox.get_nowait()
                        t0 = time.perf_counter()
                        s = (await c.get(f"/api/games/{gid}?seat=1", headers=sh)).json()
                        stats.get.append((time.perf_counter() - t0) * 1000)
                        me = s.get("me") or {}
                        if me.get("waiting") and me["turn"]["seq"] != last_seq:
                            t = me["turn"]
                            last_seq = t["seq"]
                            rid = uuid.uuid4().hex
                            pending[rid] = time.perf_counter()
                            await ws.send(json.dumps({"op": "act", "seq": t["seq"], "action": t["fallback"],
                                                      "request_id": rid}))

                tasks = [asyncio.create_task(x()) for x in (reader, beater, player)]
                try:
                    await asyncio.wait_for(done.wait(), timeout=max(1.0, stop - time.time()))
                except asyncio.TimeoutError:
                    pass
                for t in tasks:
                    t.cancel()
        except Exception as e:                      # noqa: BLE001
            stats.err(f"ws:{type(e).__name__}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8402")
    ap.add_argument("--tables", type=int, default=100)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--model", default="sim-cheap")
    ap.add_argument("--ramp", type=float, default=10.0, help="seconds over which tables are opened")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    stats = Stats()
    stop = time.time() + a.seconds
    tasks = []
    for i in range(a.tables):
        tasks.append(asyncio.create_task(table(i, a.url, a.model, stats, stop)))
        await asyncio.sleep(a.ramp / a.tables)
    await asyncio.gather(*tasks)
    res = {"tables": a.tables, "started": stats.tables_started, "finished": stats.tables_over, "acts": stats.acts,
           "push_ms": {"p50": pct(stats.push, .5), "p95": pct(stats.push, .95), "p99": pct(stats.push, .99), "n": len(stats.push)},
           "get_ms": {"p50": pct(stats.get, .5), "p95": pct(stats.get, .95), "p99": pct(stats.get, .99), "n": len(stats.get)},
           "ack_ms": {"p50": pct(stats.ack, .5), "p95": pct(stats.ack, .95), "p99": pct(stats.ack, .99), "n": len(stats.ack)},
           "errors": stats.errors}
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:  # noqa: ASYNC230 -- once, at the end
            json.dump(res, f, indent=2)
    _ = st


if __name__ == "__main__":
    asyncio.run(main())
