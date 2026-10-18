"""A TCP fault proxy in plain asyncio (no toxiproxy): every byte between the
service processes and Postgres / Redis goes through one of these.

    latency(ms, jitter)   each chunk waits ms +- jitter before it is forwarded
    loss(p)               each chunk, with probability p, is held 200-600 ms first --
                          what a lost segment costs a TCP stream (the retransmit),
                          since bytes are never really lost inside TCP
    blackhole(True)       forward nothing (connections stay open, reads hang): a partition
    cut()                 reset every open connection now (SO_LINGER 0 -> RST)
    refuse(True)          accept and immediately reset new connections: the server is "down"

Faults apply to connections already open and to new ones. `clear()` restores all.
"""
from __future__ import annotations

import asyncio
import random
import socket
import struct


class FaultProxy:
    def __init__(self, target_host: str, target_port: int, listen_port: int = 0, name: str = ""):
        self.target = (target_host, target_port)
        self.listen_port = listen_port
        self.name = name
        self.server: asyncio.base_events.Server | None = None
        self.conns: set[tuple[asyncio.StreamWriter, asyncio.StreamWriter]] = set()
        self._rng = random.Random()
        self.clear()
        self.stats = {"accepted": 0, "cut": 0, "refused": 0, "bytes": 0}

    def clear(self) -> None:
        self.lat_ms, self.jitter_ms, self.loss_p = 0.0, 0.0, 0.0
        self.hole = asyncio.Event()                        # set = blackhole on
        self.refusing = False

    # ------------------------------------------------------------ controls
    def latency(self, ms: float, jitter: float = 0.0) -> None:
        self.lat_ms, self.jitter_ms = ms, jitter

    def loss(self, p: float) -> None:
        self.loss_p = p

    def blackhole(self, on: bool) -> None:
        if on:
            self.hole.set()
        else:
            self.hole.clear()

    def refuse(self, on: bool) -> None:
        self.refusing = on

    def cut(self) -> int:
        n = 0
        for a, b in list(self.conns):
            for w in (a, b):
                _reset(w)
            n += 1
        self.conns.clear()
        self.stats["cut"] += n
        return n

    # ------------------------------------------------------------ plumbing
    @property
    def port(self) -> int:
        assert self.server is not None
        return self.server.sockets[0].getsockname()[1]

    async def start(self) -> FaultProxy:
        self.server = await asyncio.start_server(self._accept, "127.0.0.1", self.listen_port)
        return self

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
            self.cut()
            await self.server.wait_closed()

    async def _accept(self, cr: asyncio.StreamReader, cw: asyncio.StreamWriter) -> None:
        if self.refusing:
            self.stats["refused"] += 1
            _reset(cw)
            return
        try:
            ur, uw = await asyncio.open_connection(*self.target)
        except OSError:
            _reset(cw)
            return
        self.stats["accepted"] += 1
        pair = (cw, uw)
        self.conns.add(pair)
        try:
            await asyncio.gather(self._pipe(cr, uw), self._pipe(ur, cw))
        finally:
            self.conns.discard(pair)
            for w in pair:
                try:
                    w.close()
                except Exception:                          # noqa: BLE001
                    pass

    async def _pipe(self, r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        try:
            while True:
                data = await r.read(65536)
                if not data:
                    break
                while self.hole.is_set():
                    await asyncio.sleep(0.05)
                d = self.lat_ms + (self._rng.uniform(-self.jitter_ms, self.jitter_ms) if self.jitter_ms else 0)
                if self.loss_p and self._rng.random() < self.loss_p:
                    d += self._rng.uniform(200, 600)
                if d > 0:
                    await asyncio.sleep(d / 1000)
                w.write(data)
                self.stats["bytes"] += len(data)
                await w.drain()
        except (ConnectionError, OSError, asyncio.IncompleteReadError):
            pass
        finally:
            try:
                w.close()
            except Exception:                              # noqa: BLE001
                pass


def _reset(w: asyncio.StreamWriter) -> None:
    """Close with RST, not FIN: the peer sees 'connection reset', as when a server dies."""
    sock = w.get_extra_info("socket")
    try:
        if sock is not None:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        w.transport.abort()
    except Exception:                                      # noqa: BLE001
        pass
