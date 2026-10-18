"""Circuit breaker per provider deployment.

    closed     calls go through; failures are counted in a sliding window
    open       failure rate >= threshold (with at least min_calls in the window):
               calls are refused at once (the gateway moves to the next model)
               for `cooldown_s`
    half-open  after the cooldown, up to `probes` calls are let through;
               all succeed -> closed; any fails -> open again

Refusing fast is the point: a provider that is down should cost a game one
decision, not a 90-second timeout on every seat in turn.
"""
from __future__ import annotations

import threading
import time
from collections import deque


class Breaker:
    def __init__(self, name: str, *, window_s: float = 30.0, min_calls: int = 5, threshold: float = 0.5,
                 cooldown_s: float = 15.0, probes: int = 2, clock=time.monotonic):
        self.name = name
        self.window_s, self.min_calls, self.threshold = window_s, min_calls, threshold
        self.cooldown_s, self.probes = cooldown_s, probes
        self.clock = clock
        self.state = "closed"
        self.opened_at = 0.0
        self._probe_inflight = 0
        self._probe_ok = 0
        self._events: deque[tuple[float, bool]] = deque()
        self._fails = 0                  # failures currently in the window (kept incrementally: O(1) per call)
        self._lock = threading.Lock()
        self.transitions: list[tuple[str, str]] = []

    def _move(self, to: str) -> None:
        self.transitions.append((self.state, to))
        self.state = to

    def allow(self) -> bool:
        with self._lock:
            now = self.clock()
            if self.state == "open":
                if now - self.opened_at < self.cooldown_s:
                    return False
                self._move("half_open")
                self._probe_inflight = self._probe_ok = 0
            if self.state == "half_open":
                if self._probe_inflight >= self.probes:
                    return False
                self._probe_inflight += 1
            return True

    def record(self, ok: bool) -> None:
        with self._lock:
            now = self.clock()
            if self.state == "half_open":
                if not ok:
                    self._move("open")
                    self.opened_at = now
                    return
                self._probe_ok += 1
                if self._probe_ok >= self.probes:
                    self._move("closed")
                    self._events.clear()
                    self._fails = 0
                return
            self._events.append((now, ok))
            self._fails += 0 if ok else 1
            while self._events and now - self._events[0][0] > self.window_s:
                _, good = self._events.popleft()
                self._fails -= 0 if good else 1
            n = len(self._events)
            fails = self._fails
            if self.state == "closed" and n >= self.min_calls and fails / n >= self.threshold:
                self._move("open")
                self.opened_at = now
