"""A reader's copy of the draft stream, and the invariant checked against it.

`DraftMirror` applies draft messages exactly the way web/app.js does: text at
an offset (idempotent, so a replayed or synced message is harmless), commit,
retract, and -- on a `draft_reset` / `draft_sync` naming a new epoch -- void
everything held from an older one. Tests and the streaming benchmark feed it
what a client would have received and then ask `violations(journal)`:

    every draft the reader was shown ended committed or retracted, and a
    committed draft's final text is the journal's text for that turn, exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .table import Table


def outbox(t: Table, seat: int | None, cur: dict) -> list[tuple[str, dict]]:
    """Everything a reader has not seen yet, in the order it happened; moves the cursor past it.

    `cur` = {"next": next event seq, "epoch": draft epoch the reader is on (None: none
    yet), "dlast": last dseq it has, "known": {draft_id: chars it holds}}.

    Two sequences are merged: journal-derived events (`seq`, the resume cursor that
    survives recovery) and draft events (`dseq`, live only, per epoch). A draft event
    records how many events existed when it was emitted, which is where it goes in the
    merge -- so a draft_commit is always delivered before the turn it became.

    A reader on another epoch (it last saw a dead incarnation, or none) or one whose
    dseq has been trimmed out of the log gets a `draft_sync` instead: void what you hold
    from other epochs, here is every open draft (from the offset you said you have),
    and the outcome of each draft you named that is no longer open.
    """
    out: list[tuple[str, dict]] = []
    devs = t.draft_events(cur["dlast"], seat) if cur.get("epoch") == t.draft_epoch else None
    if devs is None:
        out.append(("draft", {"kind": "draft_sync", "epoch": t.draft_epoch, "dseq": t.dseq - 1}))
        out += [("draft", m) for m in t.draft_sync(seat, cur.get("known"))]
        cur.update(epoch=t.draft_epoch, known={})
        devs = []
    cur["dlast"] = t.dseq - 1
    i = 0
    for e in t.events[cur["next"]:]:
        while i < len(devs) and devs[i].at <= e.seq:
            out.append(("draft", devs[i].to_dict(t.draft_epoch)))
            i += 1
        out.append(("event", e.to_dict()))
        cur["next"] = e.seq + 1
    out += [("draft", d.to_dict(t.draft_epoch)) for d in devs[i:]]
    return out


@dataclass
class Shown:
    draft_id: str
    seat: int
    seq: int
    task: str
    epoch: int
    text: str = ""
    state: str = "open"              # open | committed | retracted
    final: str | None = None
    n: int | None = None
    reason: str | None = None
    gaps: int = 0


@dataclass
class DraftMirror:
    drafts: dict[str, Shown] = field(default_factory=dict)
    epoch: int | None = None
    messages: int = 0
    seen_tasks: set = field(default_factory=set)

    def apply(self, m: dict) -> None:
        self.messages += 1
        kind = m.get("kind")
        if kind in ("draft_reset", "draft_sync"):
            self._new_epoch(int(m.get("epoch", 0)), "recovered" if kind == "draft_reset" else "sync")
            return
        did = m.get("draft_id")
        if not did:
            return
        d = self.drafts.get(did)
        if kind == "draft":
            if d is None:
                if int(m.get("offset", 0)) != 0:
                    return
                d = self.drafts[did] = Shown(did, m["seat"], m["seq"], m.get("task", ""), int(m.get("epoch", 0)))
                self.seen_tasks.add(d.task)
            if d.state != "open":
                return
            off = int(m.get("offset", 0))
            if off > len(d.text):
                d.gaps += 1
                return
            d.text = d.text[:off] + m.get("delta", "")
        elif kind == "draft_commit" and d is not None:
            d.state, d.final, d.n = "committed", m.get("text"), m.get("n")
        elif kind == "draft_retract" and d is not None and d.state == "open":
            d.state, d.reason = "retracted", m.get("reason")

    def _new_epoch(self, epoch: int, why: str) -> None:
        """Anything still open from another incarnation died with it."""
        for d in self.drafts.values():
            if d.state == "open" and d.epoch != epoch:
                d.state, d.reason = "retracted", why
        self.epoch = epoch

    def violations(self, journal, *, final: bool = True) -> list[str]:
        """What breaks the invariant. With final=False open drafts are allowed (a game still running)."""
        out = []
        for d in self.drafts.values():
            if not d.text:
                continue
            if d.state == "open":
                if final:
                    out.append(f"{d.draft_id}: shown and never closed")
            elif d.state == "committed":
                rec = journal[d.n - 1] if d.n and 0 < d.n <= len(journal) else None
                said = (rec.action.get("args") or {}).get("content") if rec is not None and isinstance(rec.action, dict) else None
                if rec is None or rec.actor != d.seat or said != d.final:
                    out.append(f"{d.draft_id}: committed text is not the journal's (n={d.n})")
            if d.gaps:
                out.append(f"{d.draft_id}: {d.gaps} gap(s) in the stream")
        return out

    def counts(self) -> dict:
        c = {"shown": 0, "committed": 0, "retracted": 0, "open": 0}
        for d in self.drafts.values():
            if d.text:
                c["shown"] += 1
                c[d.state] += 1
        return c
