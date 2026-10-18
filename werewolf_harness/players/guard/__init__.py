"""The guard stack: three layers plus evidence enforcement.

Split into layers so the ablation can turn them on one at a time and the cost
of each is measurable independently:

    L1  isolate   structural fencing of everything written by others
    L2  filter    detect and strip forged structure / instruction overrides
    L3  verify    output-side checks before an action reaches the world
    E   evidence  no attribution without a prior lookup

Every layer both blocks attacks and, on benign strong persuasion, produces
false positives. The stack therefore records what it did on every call: the
same counters feed the security numbers and the over-defense numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import evidence as evidence_mod
from . import filter as filter_mod
from . import isolate as isolate_mod
from . import verify as verify_mod

LAYERS = ("L1", "L2", "L3")


@dataclass
class GuardStats:
    incoming_scanned: int = 0
    incoming_with_detections: int = 0
    detections: list[dict] = field(default_factory=list)
    terminal_blocks: list[dict] = field(default_factory=list)
    evidence_blocks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "incoming_scanned": self.incoming_scanned,
            "incoming_with_detections": self.incoming_with_detections,
            "detections": self.detections,
            "terminal_blocks": self.terminal_blocks,
            "evidence_blocks": self.evidence_blocks,
        }


class GuardStack:
    def __init__(
        self,
        layers: tuple[str, ...] | list[str] = (),
        evidence_forced: bool = False,
        filter_categories: set[str] | None = None,
    ):
        unknown = set(layers) - set(LAYERS)
        if unknown:
            raise ValueError(f"unknown guard layer(s): {sorted(unknown)}")
        self.layers = tuple(layers)
        self.evidence_forced = evidence_forced
        self.filter_categories = filter_categories
        self.stats = GuardStats()

    # ---- configuration -------------------------------------------------

    @property
    def l1(self) -> bool:
        return "L1" in self.layers

    @property
    def l2(self) -> bool:
        return "L2" in self.layers

    @property
    def l3(self) -> bool:
        return "L3" in self.layers

    def label(self) -> str:
        parts = list(self.layers) or ["none"]
        if self.evidence_forced:
            parts.append("E")
        return "+".join(parts)

    def system_preamble(self) -> str:
        return isolate_mod.PREAMBLE if self.l1 else ""

    # ---- inbound -------------------------------------------------------

    def clean_incoming(
        self,
        text: str,
        *,
        source: str,
        kind: str,
        round_no: int | None = None,
    ) -> tuple[str, list[dict]]:
        """Run untrusted text through L2 then L1.

        Order matters: filter first so the fence never contains live markers,
        wrap second so what the model sees is always labelled.
        """
        self.stats.incoming_scanned += 1
        detections: list[dict] = []
        out = text

        if self.l2:
            # Sentence-level removal, not a visible stub: the stub is kept for
            # the log only. See filter.strip_segments for why.
            out, dets, log_text = filter_mod.strip_segments(out, self.filter_categories)
            detections = [
                d.to_dict() | {"source": source, "kind": kind, "redacted_text": log_text}
                for d in dets
            ]
        else:
            # Detection without action: needed so "no guard" runs still record
            # where the payloads were, which is what axis 1 is measured against.
            detections = [
                d.to_dict() | {"source": source, "kind": kind, "observed_only": True}
                for d in filter_mod.scan(out, self.filter_categories)
            ]

        if detections:
            self.stats.incoming_with_detections += 1
            self.stats.detections.extend(detections)

        if self.l1:
            out = isolate_mod.wrap(out, source=source, kind=kind, round_no=round_no)

        return out, detections

    # ---- outbound ------------------------------------------------------

    def verify_terminal(
        self,
        action: str,
        args: dict,
        *,
        own_speech_this_round: str | None,
        injected_directives: set[int],
        belief_evidence: dict[int, list[str]],
        player_id: int,
        round_no: int,
    ) -> verify_mod.Verdict:
        if not self.l3:
            return verify_mod.Verdict.ok()
        verdict = verify_mod.verify_terminal(
            action,
            args,
            own_speech_this_round=own_speech_this_round,
            injected_directives=injected_directives,
            belief_evidence=belief_evidence,
        )
        if verdict.blocked:
            self.stats.terminal_blocks.append(
                {
                    "player_id": player_id,
                    "round": round_no,
                    "action": action,
                    "args": args,
                    "check": verdict.check,
                    "reason": verdict.reason,
                }
            )
        return verdict

    def check_evidence(
        self,
        content: str,
        *,
        speaker: int,
        current_round: int,
        queried_players: set[int],
        queried_vote_rounds: set[int],
    ) -> evidence_mod.EvidenceVerdict:
        if not self.evidence_forced:
            return evidence_mod.EvidenceVerdict.ok(
                sorted(evidence_mod.referenced_players(content) - {speaker})
            )
        verdict = evidence_mod.check_speech(
            content,
            speaker=speaker,
            current_round=current_round,
            queried_players=queried_players,
            queried_vote_rounds=queried_vote_rounds,
        )
        if verdict.blocked:
            self.stats.evidence_blocks.append(
                {"player_id": speaker, "round": current_round, "reason": verdict.reason}
            )
        return verdict


__all__ = [
    "GuardStack",
    "GuardStats",
    "LAYERS",
    "evidence_mod",
    "filter_mod",
    "isolate_mod",
    "verify_mod",
]
