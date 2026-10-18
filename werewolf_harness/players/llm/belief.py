"""Structured long-term memory of one seat.

Three layers, kept apart on purpose:

  working memory  -- the current ReAct chain, discarded at the end of the turn
  short-term      -- this round's observations, live in the context window
  long-term       -- this module: one entry per player, carried across rounds

It lives inside the Player, not the environment: a belief was never an action,
so the world (``f(seed, journal)``) does not contain it and replay cannot
re-derive it. That is also why ``snapshot``/``restore`` exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BeliefEntry:
    player_id: int
    suspicion: float
    guessed_role: str | None = None
    reason: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    updated_at_round: int = 0

    def to_dict(self) -> dict:
        return {"player_id": self.player_id, "suspicion": self.suspicion, "guessed_role": self.guessed_role,
                "reason": self.reason, "evidence_refs": list(self.evidence_refs), "updated_at_round": self.updated_at_round}

    @classmethod
    def from_dict(cls, d: dict) -> "BeliefEntry":
        return cls(player_id=int(d["player_id"]), suspicion=float(d["suspicion"]), guessed_role=d.get("guessed_role"),
                   reason=d.get("reason", ""), evidence_refs=list(d.get("evidence_refs", [])),
                   updated_at_round=int(d.get("updated_at_round", 0)))


class BeliefState:
    """player_id -> BeliefEntry, with snapshotting for the log."""

    def __init__(self, owner: int, players: list[int]):
        self.owner = owner
        self.entries: dict[int, BeliefEntry] = {
            p: BeliefEntry(player_id=p, suspicion=0.5, reason="no information yet") for p in players if p != owner
        }

    def get(self, player_id: int) -> BeliefEntry | None:
        return self.entries.get(player_id)

    def update(self, player_id: int, suspicion: float, reason: str, round_no: int,
               evidence_refs: list[str] | None = None, guessed_role: str | None = None) -> BeliefEntry:
        entry = self.entries.get(player_id)
        if entry is None:
            entry = BeliefEntry(player_id=player_id, suspicion=suspicion)
            self.entries[player_id] = entry
        entry.suspicion = max(0.0, min(1.0, float(suspicion)))
        entry.reason = reason
        entry.updated_at_round = round_no
        if evidence_refs is not None:
            entry.evidence_refs = list(evidence_refs)
        if guessed_role is not None:
            entry.guessed_role = guessed_role
        return entry

    def most_suspected(self, among: list[int]) -> int | None:
        candidates = [(self.entries[p].suspicion, -p) for p in among if p in self.entries]
        return -max(candidates)[1] if candidates else None

    def snapshot(self) -> dict:
        return {str(p): e.to_dict() for p, e in sorted(self.entries.items())}

    def restore(self, snapshot: dict) -> None:
        for key, raw in (snapshot or {}).items():
            pid = int(key)
            if pid != self.owner:
                self.entries[pid] = BeliefEntry.from_dict(raw)

    def summarise(self, alive: list[int]) -> str:
        rows = []
        for p in sorted(alive):
            e = self.entries.get(p)
            if e is None:
                continue
            role = f", guess={e.guessed_role}" if e.guessed_role else ""
            rows.append(f"  player {p}: suspicion={e.suspicion:.2f}{role} (r{e.updated_at_round}: {e.reason[:90]})")
        return "\n".join(rows) if rows else "  (no beliefs recorded yet)"

    @staticmethod
    def diff(before: dict, after: dict) -> list[dict]:
        changes = []
        for pid, a in after.items():
            b = before.get(pid)
            if b is None:
                changes.append({"player_id": int(pid), "from": None, "to": a["suspicion"]})
            elif abs(b["suspicion"] - a["suspicion"]) > 1e-9:
                changes.append({"player_id": int(pid), "from": b["suspicion"], "to": a["suspicion"], "reason": a.get("reason", "")})
        return changes
