"""Spectators that behave like the web client: one WebSocket and one SSE reader per game.

Each keeps the resume cursor a browser keeps (last event seq, draft epoch, last
dseq, the drafts it holds) and, when its connection ends -- the node died, or
closed it -- reconnects to whichever node answers for the game, resuming from
that cursor. Everything it is shown goes into a DraftMirror, the same reader
model web/app.js implements, so the draft invariant is checked on what a person
watching would actually have seen.

A node that does not drive the game answers 4404 (WebSocket) / 409 (SSE); the
web client then polls GET /api/games/{id}, so a reader that finds no node does
too, and a game that ended while it was between nodes is seen ended that way
(its drafts from the dead epoch void, as a draft_sync would make them).

A reader only reconnects when its connection ends. If a node keeps a socket open
for a game it no longer drives, the reader waits there -- as a browser would --
and the check at the end reports it.
"""
from __future__ import annotations

import asyncio
import json
import time

import httpx
import websockets

from ..app.runtime.drafts import DraftMirror


class Watcher:
    kind = "?"

    def __init__(self, cluster, gid: str):
        self.cluster, self.gid = cluster, gid
        self.mirror = DraftMirror()
        self.next = 0                       # next event seq wanted
        self.epoch: int | None = None
        self.dlast = -1
        self.ended: str | None = None       # "end" | "error"
        self.connects = 0
        self.gaps = 0                       # an event seq past the one expected
        self.last_msg = time.monotonic()
        self.task: asyncio.Task | None = None
        self._pref = 0
        self.trail: list[str] = []          # where it connected and how each session ended (for a failed check)
        self.t0 = time.monotonic()

    def start(self) -> None:
        self.task = asyncio.create_task(self._loop())

    async def close(self) -> None:
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):    # noqa: BLE001
                pass

    def _event(self, e: dict) -> None:
        seq = int(e["seq"])
        if seq > self.next:
            self.gaps += 1
        if seq >= self.next:
            self.next = seq + 1
        if e.get("kind") == "end":
            self.ended = "end"
        elif e.get("kind") == "error":
            self.ended = "error"

    def _draft(self, m: dict) -> None:
        self.mirror.apply(m)
        if "epoch" in m:
            self.epoch = int(m["epoch"])
        if "dseq" in m and not m.get("sync"):
            self.dlast = max(self.dlast, int(m["dseq"]))

    def known(self) -> dict:
        return {d.draft_id: len(d.text) for d in self.mirror.drafts.values() if d.state == "open" and d.text}

    async def _loop(self) -> None:
        while self.ended is None:
            nodes = self.cluster.up_nodes()
            if not nodes:
                await asyncio.sleep(0.2)
                continue
            n = nodes[self._pref % len(nodes)]
            try:
                served = await self._session(n)
                why = "closed"
            except (OSError, websockets.WebSocketException, httpx.HTTPError, asyncio.TimeoutError, ValueError) as exc:
                served, why = False, type(exc).__name__ + (f" {exc.rcvd.code}" if getattr(exc, "rcvd", None) else "")
            if served:
                self.trail.append(f"{time.monotonic() - self.t0:.1f}s {n.name} {why} next={self.next} epoch={self.epoch}")
            if not served:
                self._pref += 1                            # not there: try the next node
                if self._pref % max(1, len(nodes)) == 0 and await self._poll(n):
                    return
                await asyncio.sleep(0.15)

    async def _poll(self, n) -> bool:
        """No node streams it: ask any node for its state, as the web client's polling does."""
        try:
            st = ((await self.cluster.get(n, f"/api/games/{self.gid}")).get("game") or {}).get("state")
        except Exception:                                  # noqa: BLE001
            return False
        if st in ("over", "aborted", "crashed"):
            self.mirror._new_epoch(-1, "sync")
            self.ended = "poll"
            self.trail.append(f"{time.monotonic() - self.t0:.1f}s {n.name} poll: {st}")
            return True
        return False

    async def _session(self, n) -> bool:
        raise NotImplementedError

    def report(self) -> dict:
        return {"kind": self.kind, "ended": self.ended, "connects": self.connects, "gaps": self.gaps,
                "trail": self.trail[-8:] if self.ended is None else [],
                "drafts": self.mirror.counts(), "seen_tasks": sorted(self.mirror.seen_tasks)}


class WSWatcher(Watcher):
    kind = "ws"

    async def _session(self, n) -> bool:
        q = f"last_seq={self.next - 1}&dseq={self.dlast}" + (f"&epoch={self.epoch}" if self.epoch is not None else "")
        async with websockets.connect(f"ws://127.0.0.1:{n.port}/ws/games/{self.gid}?{q}", open_timeout=3,
                                      close_timeout=1, max_size=None) as ws:
            hello = json.loads(await asyncio.wait_for(ws.recv(), 3))
            if hello.get("op") != "hello":
                return False
            self.connects += 1
            if self.known():
                await ws.send(json.dumps({"op": "resume", "last_seq": self.next - 1, "epoch": self.epoch,
                                          "dseq": self.dlast, "drafts": self.known()}))
            pinger = asyncio.create_task(self._ping(ws))
            try:
                async for raw in ws:
                    m = json.loads(raw)
                    self.last_msg = time.monotonic()
                    if m.get("op") == "event":
                        self._event(m)
                    elif m.get("op") == "draft":
                        self._draft(m)
                    if self.ended:
                        return True
            except websockets.ConnectionClosed:
                pass
            finally:
                pinger.cancel()
        return True

    async def _ping(self, ws) -> None:
        while True:
            await asyncio.sleep(5)
            await ws.send(json.dumps({"op": "ping", "t": time.time()}))


class SSEWatcher(Watcher):
    kind = "sse"

    async def _session(self, n) -> bool:
        q = f"after={self.next}&drafts=1&dseq={self.dlast}" + (f"&epoch={self.epoch}" if self.epoch is not None else "")
        url = f"http://127.0.0.1:{n.port}/api/games/{self.gid}/events?{q}"
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, read=40.0)) as c:
            async with c.stream("GET", url) as r:
                if r.status_code != 200:
                    return False
                self.connects += 1
                kind = None
                async for line in r.aiter_lines():
                    if line.startswith("event:"):
                        kind = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        m = json.loads(line.split(":", 1)[1])
                        self.last_msg = time.monotonic()
                        if kind == "draft":
                            self._draft(m)
                        else:
                            self._event(m)
                        kind = None
                        if self.ended:
                            return True
        return True
