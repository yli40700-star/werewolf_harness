"""A seat a person sits in.

It implements the same one method a model seat does. The environment cannot
tell them apart -- it hands both the same view and the same list of legal
actions, and takes back a value of the same shape. Everything that differs is
inside `act`: instead of a prompt and a completion, a queue and a clock.

Submissions are idempotent per turn. A retried POST, a double-click and a
redelivered message all carry the same sequence number, and only the first one
is taken; the rest are told they are duplicates. Without that, "the same action
twice" is not a repeated write, it is a different world.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..arena import ActionSpec


@dataclass
class Pending:
    """What the seat is being asked, in a shape a browser can render."""
    seq: int
    task: str
    view: dict
    spec: ActionSpec
    deadline: float | None
    feedback: str | None = None

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "task": self.task,
            "view": self.view,
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "params": {
                        k: {"type": p.type, "description": p.description, "required": p.required,
                            "choices": list(p.choices) if p.choices is not None else None,
                            "labels": dict(p.labels) if p.labels else None,
                            "max_len": p.max_len, "range": list(p.range) if p.range else None}
                        for k, p in t.params.items()
                    },
                }
                for t in self.spec.tools
            ],
            "fallback": self.spec.fallback,
            "feedback": self.feedback,
            "deadline": self.deadline,
            "seconds_left": None if self.deadline is None else max(0, round(self.deadline - time.time())),
        }


class HumanPlayer:
    """Blocks the turn until someone submits, or the clock runs out.

    A timeout is not a crash and not a random move: the environment's own
    fallback is used -- the quietest legal action it defines -- and the turn is
    marked so the record never reads as a decision the person made.
    """

    is_human = True

    def __init__(self, seat: int, turn_seconds: float | None = 300.0, on_change=None):
        self.seat = seat
        self.turn_seconds = turn_seconds
        self.on_change = on_change or (lambda: None)
        self.pending: Pending | None = None
        self.history: list[dict] = []
        self._inbox: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._seq = 0
        self._settled: dict[int, Any] = {}
        self.last_forced = False        # the driver reads this: a timeout is not a decision

    # ---------------------------------------------------------- the contract
    def act(self, view: dict, task: str, spec: ActionSpec, feedback: str | None) -> Any:
        with self._lock:
            seq = self._seq
            deadline = None if self.turn_seconds is None else time.time() + self.turn_seconds
            self.pending = Pending(seq, task, view, spec, deadline, feedback)
        self.on_change()

        forced, action = False, None
        self.last_forced = False
        try:
            while True:
                wait = None if deadline is None else max(0.0, deadline - time.time())
                got_seq, action = self._inbox.get(timeout=wait)
                if got_seq == seq:
                    break                                   # anything else is a stale submission
        except queue.Empty:
            action, forced = spec.fallback, True
            self.last_forced = True

        with self._lock:
            self._settled[seq] = action
            self._seq += 1
            self.pending = None
        self.history.append({"seq": seq, "task": task, "action": action, "forced": forced,
                             "fallback_used": "timeout" if forced else None})
        self.on_change()
        return action

    # ---------------------------------------------------------- submissions
    def submit(self, seq: int, action: Any) -> dict:
        """Take one submission. Returns what happened, never raises on a repeat."""
        with self._lock:
            if seq in self._settled:
                return {"status": "duplicate", "action": self._settled[seq]}
            pending = self.pending
            if pending is None:
                return {"status": "not_your_turn"}
            if seq != pending.seq:
                return {"status": "stale", "expected": pending.seq}
            reason = pending.spec.check(action)
            if reason:
                return {"status": "illegal", "reason": reason}
        self._inbox.put((seq, action))
        return {"status": "accepted"}

    # ---------------------------------------------------------- for the page
    def snapshot(self) -> dict:
        with self._lock:
            p = self.pending
        return {"seat": self.seat, "waiting": p is not None, "turn": p.to_dict() if p else None,
                "turn_seconds": self.turn_seconds,
                "forced": sum(1 for h in self.history if h["forced"]), "turns": len(self.history)}

    @property
    def fallback_used(self):
        return self.history[-1]["fallback_used"] if self.history else None
