"""A hashed timing wheel (the Netty/Kafka shape) on one asyncio task.

Every turn in every game has a clock -- a person's speaking time, a model call's
deadline -- so a node with a thousand tables holds a few thousand pending
timeouts that are mostly cancelled before they fire. A heap (asyncio's own
call_later) makes each insert and cancel O(log n); the wheel makes them O(1) and
pays with resolution: a timeout fires within one tick of its deadline.

    slots:  ring of `size` buckets; the cursor advances one bucket per `tick`
    rounds: a delay longer than one lap stores how many laps are left

Cancelling only flags the entry; the bucket drops it when the cursor passes.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable


class Timeout:
    __slots__ = ("callback", "rounds", "cancelled", "deadline")

    def __init__(self, callback: Callable[[], None], rounds: int, deadline: float):
        self.callback = callback
        self.rounds = rounds
        self.deadline = deadline
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class TimingWheel:
    def __init__(self, tick: float = 0.1, size: int = 512):
        self.tick = tick
        self.size = size
        self.slots: list[list[Timeout]] = [[] for _ in range(size)]
        self.cursor = 0
        self.pending = 0
        self._task: asyncio.Task | None = None
        self._started_at = 0.0
        self._ticks = 0

    def start(self) -> None:
        if self._task is None:
            self._started_at = time.monotonic()
            self._task = asyncio.get_running_loop().create_task(self._run(), name="timing-wheel")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def schedule(self, delay: float, callback: Callable[[], None]) -> Timeout:
        ticks = max(1, int(round(delay / self.tick)))
        rounds, offset = divmod(ticks, self.size)
        if offset == 0:                       # exactly N laps: fire at the end of the last lap, not one lap early
            rounds, offset = rounds - 1, self.size
        slot = (self.cursor + offset) % self.size
        t = Timeout(callback, rounds, time.monotonic() + delay)
        self.slots[slot].append(t)
        self.pending += 1
        return t

    def _advance(self) -> None:
        self.cursor = (self.cursor + 1) % self.size
        bucket = self.slots[self.cursor]
        keep: list[Timeout] = []
        for t in bucket:
            if t.cancelled:
                self.pending -= 1
            elif t.rounds > 0:
                t.rounds -= 1
                keep.append(t)
            else:
                self.pending -= 1
                try:
                    t.callback()
                except Exception:              # noqa: BLE001 -- one bad callback must not stop every clock
                    pass
        self.slots[self.cursor] = keep

    async def _run(self) -> None:
        # Tick against the wall clock, not by sleeping `tick` each time: a slow
        # tick (a busy loop) catches up instead of drifting every deadline late.
        while True:
            self._ticks += 1
            target = self._started_at + self._ticks * self.tick
            await asyncio.sleep(max(0.0, target - time.monotonic()))
            self._advance()
