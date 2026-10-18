"""The model seats of a fault run: free, deterministic in what they decide, steerable in how they fail.

`faults-mock` is MockClient (seeded choices, streamed speeches) with two additions:

* a control file (env FAULTS_CONTROL, JSON) the harness rewrites while a scenario
  runs -- {"error_rate": 0.5, "timeout_rate": 0.2, "slow_ms": 3000} -- read at
  most every 100 ms, so every process (API nodes, workers) changes behaviour
  within a tick of the harness deciding a window has started;
* a call log (env FAULTS_CALL_LOG, JSONL, one line per call, O_APPEND from every
  process) with game, seat, the version the turn was asked at and the start
  time -- what the "no model call for a committed turn" invariant is checked on.

Only the harness sets FAULTS_CONTROL; without it the name is not offered anywhere.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time

from ..players.providers.base import Cancelled, CancelToken, ProviderError
from ..players.providers.mock import _TASK, MockClient

MODEL = "faults-mock"
# fast enough that a game is ~20-30 s, slow enough that a kill lands mid-turn and mid-speech
PROFILE = {"sleep_ms": 120, "ttft_s": 0.08, "tokens_per_s": 600.0, "chunk_tokens": 6, "speech_tokens": 150}
_NOTE = "write a note to yourself"


def enabled() -> bool:
    return bool(os.environ.get("FAULTS_CONTROL"))


class _Control:
    """The control file, cached for 100 ms (every model call reads it)."""

    def __init__(self) -> None:
        self.path = os.environ.get("FAULTS_CONTROL", "")
        self._at = 0.0
        self._val: dict = {}
        self._lock = threading.Lock()

    def get(self) -> dict:
        now = time.monotonic()
        with self._lock:
            if now - self._at > 0.1:
                self._at = now
                try:
                    with open(self.path, encoding="utf-8") as f:
                        self._val = json.load(f) or {}
                except (OSError, ValueError):
                    pass                                   # mid-rewrite or absent: keep the last good one
            return self._val


_CONTROL = _Control()
_rng = random.Random()                                     # failures are not seeded: they hit whatever is running


class FaultMockClient(MockClient):
    name = MODEL

    def __init__(self, seed: int = 0):
        super().__init__(seed=seed, name=MODEL, **PROFILE)
        self.ctx: dict = {}

    def bind_call(self, game_id: str | None, seat: int | None, seq: int | None) -> None:
        self.ctx = {"g": game_id, "s": seat, "v": seq}

    def _log(self, messages, cancel_or_none) -> None:
        path = os.environ.get("FAULTS_CALL_LOG")
        if not path:
            return
        last = (messages[-1].get("content") or "") if messages else ""
        prompt = "\n".join(m.get("content") or "" for m in messages)
        m = _TASK.search(prompt)
        row = {**self.ctx, "t": time.time(), "pid": os.getpid(),
               "kind": "note" if _NOTE in last else "step",
               "step": sum(1 for x in messages if x.get("role") == "assistant"),
               "task": m.group(1) if m else None}
        line = json.dumps(row) + "\n"
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)

    def _faults(self, cancel: CancelToken | None) -> None:
        c = _CONTROL.get()
        slow = float(c.get("slow_ms") or 0) / 1000
        if slow:
            if cancel is not None and cancel.wait(slow):
                raise Cancelled(cancel.reason or "cancelled", 0)
            if cancel is None:
                time.sleep(slow)
        x = _rng.random()
        if x < float(c.get("error_rate") or 0):
            raise ProviderError("faults-mock: injected 503", status=503, hint="注入的上游故障")
        if x < float(c.get("error_rate") or 0) + float(c.get("timeout_rate") or 0):
            # what a provider timeout looks like to the gateway (classify(): no status, "timed out")
            wait = float(c.get("timeout_s") or 1.0)
            if cancel is not None and cancel.wait(wait):
                raise Cancelled(cancel.reason or "cancelled", 0)
            if cancel is None:
                time.sleep(wait)
            raise ProviderError("faults-mock: request timed out (injected)")

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0):
        self._log(messages, None)
        self._faults(None)
        return super().chat(messages, tools, temperature, max_tokens, timeout)

    def chat_stream(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0,
                    cancel: CancelToken | None = None):
        self._log(messages, cancel)
        self._faults(cancel)
        yield from super().chat_stream(messages, tools, temperature, max_tokens, timeout, cancel=cancel)
