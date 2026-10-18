"""Admission control: a new game starts only while the model layer has room.

Running games are never paused or slowed on purpose: they were promised a
partner that answers. What the node can still choose is whether to take on
*more* work, so overload protection lives entirely at the door:

    POST /api/games ──▶ decide(...) ──┬─ admit   start now                     201
                                      ├─ queue   state "queued", FIFO position  202
                                      └─ reject  waiting queue full             503 + Retry-After

    Runtime load tick (every load_tick_s): while not overloaded and queue not
    empty, start the head of the queue -- one per tick, so a burst of starts
    does not push the node straight back over the threshold before the load
    signal has had time to see the first one.

Slow start in the "busy" band (pressure between the exit and enter thresholds):
a game started now shows its full weight on the load signal only a minute later
(its 8-seat votes come on day 1), so a node that kept admitting until the signal
crossed 1.0 would already have taken in a burst too big for it. While busy, at
most one game starts per `probe_s` -- directly or from the queue -- and the rest
queue. While calm, games start at once; while overloaded, none do.

FIFO, strictly: while anyone is queued a newcomer queues behind them even if
the node happens to be calm at that instant, or a lucky refresh would jump the
line. A queued game is a real row in Postgres (state 'queued', no lease), so a
restarted node puts its own queued games back in line in creation order.

ETA. Games leave the system at about `active / mean_game_s` per second (each
running game frees its slot when it ends), so position N waits about
N * mean_game_s / active. mean_game_s is an EWMA over games that ended here,
seeded with a typical game length; it is a hint for a person, not a promise.
"""
from __future__ import annotations

import collections
import math
import time
from dataclasses import dataclass, field
from typing import Any

ADMIT, QUEUE, REJECT = "admit", "queue", "reject"


def decide(*, overloaded: bool, queued: int, max_queue: int, active: int, max_active: int = 0,
           busy: bool = False, since_last_start_s: float = float("inf"), probe_s: float = 0.0) -> str:
    """Pure decision for one creation request."""
    full = max_active > 0 and active >= max_active
    probing = busy and since_last_start_s < probe_s
    if not overloaded and not full and not probing and queued == 0:
        return ADMIT
    if queued >= max_queue:
        return REJECT
    return QUEUE


@dataclass
class Ticket:
    game_id: str
    table: Any                    # runtime.table.Table, built but not started
    enqueued: float = field(default_factory=time.time)


class Overloaded(Exception):
    """The waiting queue is full: refuse, and say when to come back."""

    def __init__(self, retry_after_s: int):
        super().__init__(f"overloaded; retry in {retry_after_s}s")
        self.retry_after_s = retry_after_s


class Admission:
    def __init__(self, *, max_queue: int = 50, max_active: int = 0, mean_game_s: float = 120.0,
                 probe_s: float = 5.0):
        self.max_queue = max_queue
        self.max_active = max_active
        self.probe_s = probe_s
        self.last_start = 0.0                     # monotonic time a game last started (either door)
        self.mean_game_s = mean_game_s
        self.q: collections.OrderedDict[str, Ticket] = collections.OrderedDict()
        self.waits: list[float] = []             # seconds each admitted-from-queue game waited (bench reads it)
        self.counts = {ADMIT: 0, QUEUE: 0, REJECT: 0}

    def __len__(self) -> int:
        return len(self.q)

    def __contains__(self, gid: str) -> bool:
        return gid in self.q

    def _since_start(self) -> float:
        return time.monotonic() - self.last_start if self.last_start else float("inf")

    def decide(self, *, overloaded: bool, active: int, busy: bool = False) -> str:
        d = decide(overloaded=overloaded, queued=len(self.q), max_queue=self.max_queue, active=active,
                   max_active=self.max_active, busy=busy, since_last_start_s=self._since_start(),
                   probe_s=self.probe_s)
        self.counts[d] += 1
        if d == ADMIT:
            self.started()
        from ..obs import metrics
        metrics.ADMISSIONS.labels(decision=d).inc()
        return d

    def started(self) -> None:
        self.last_start = time.monotonic()

    def can_start_next(self, *, overloaded: bool, active: int, busy: bool = False) -> bool:
        full = self.max_active > 0 and active >= self.max_active
        probing = busy and self._since_start() < self.probe_s
        return bool(self.q) and not overloaded and not full and not probing

    def push(self, t: Ticket) -> int:
        self.q[t.game_id] = t
        return len(self.q)

    def pop(self) -> Ticket:
        _gid, t = self.q.popitem(last=False)
        self.started()
        wait = time.time() - t.enqueued
        self.waits.append(wait)
        from ..obs import metrics
        metrics.ADMISSION_WAIT_S.observe(wait)
        return t

    def position(self, gid: str) -> int | None:
        for i, k in enumerate(self.q, 1):
            if k == gid:
                return i
        return None

    # ------------------------------------------------------------ estimates
    def game_ended(self, duration_s: float) -> None:
        if duration_s > 0:
            self.mean_game_s = 0.8 * self.mean_game_s + 0.2 * duration_s

    def eta_s(self, position: int, active: int) -> int:
        return max(1, math.ceil(position * self.mean_game_s / max(1, active)))

    def retry_after_s(self, active: int) -> int:
        """For a refusal: when the tail of today's queue should be moving."""
        return min(600, max(5, self.eta_s(len(self.q) + 1, active)))

    def info(self, gid: str, active: int) -> dict | None:
        pos = self.position(gid)
        if pos is None:
            return None
        t = self.q[gid]
        return {"position": pos, "eta_s": self.eta_s(pos, active), "waited_s": round(time.time() - t.enqueued, 1),
                "length": len(self.q)}
