"""Evidence enforcement.

Rule: you may not attribute a statement to a player unless you looked it up
first. Concretely, if a speech refers to what player N said in an earlier
round, the turn's trace must contain a successful `query_history(N, r)` for
some earlier round r; otherwise the speech is blocked and regenerated.

This exists because the context deliberately withholds earlier rounds' speech
text (only the current round is in the window), so "player 5 said he was a
villager" is either grounded in a lookup or invented. That makes the rule a
clean single-variable ablation: forced evidence should cut fabricated citations
at the cost of extra tool calls per turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# "player 5 said", "5 said", "5号说过", "5 号昨天说"
_ATTRIBUTION = [
    re.compile(r"(?i)\bplayer\s*#?(\d)\b[^.\n]{0,30}?\b(said|claimed|told us|stated|admitted|argued)\b"),
    re.compile(r"(?i)\b(\d)\s*(?:said|claimed|stated)\b"),
    re.compile(r"#?(\d)\s*号?[^。\n]{0,10}?(说过|说了|声称|承认|表示|讲过)"),
    re.compile(r"(?i)\b(?:as|like)\s+player\s*#?(\d)\s+(?:said|put it)\b"),
]

# Statements about a past round's ballot need query_votes instead.
_VOTE_CLAIM = re.compile(
    r"(?i)\b(?:player\s*)?#?(\d)\s*(?:号)?\s*(?:voted|投了|投给)\b"
)


@dataclass
class EvidenceVerdict:
    blocked: bool
    reason: str | None = None
    referenced: list[int] = None  # players the speech attributed statements to

    @staticmethod
    def ok(referenced: list[int] | None = None) -> "EvidenceVerdict":
        return EvidenceVerdict(False, None, referenced or [])


def referenced_players(text: str) -> set[int]:
    """Which players this text attributes a past statement to."""
    found = set()
    for rx in _ATTRIBUTION:
        for m in rx.finditer(text or ""):
            found.add(int(m.group(1)))
    return found


def referenced_ballots(text: str) -> set[int]:
    return {int(m.group(1)) for m in _VOTE_CLAIM.finditer(text or "")}


def check_speech(
    content: str,
    *,
    speaker: int,
    current_round: int,
    queried_players: set[int],
    queried_vote_rounds: set[int],
) -> EvidenceVerdict:
    """Block a speech that cites what it never looked up.

    Only *past* rounds are enforced: the current round's speeches are already
    in the context window, so quoting them needs no lookup.
    """
    refs = referenced_players(content) - {speaker}
    missing = sorted(p for p in refs if p not in queried_players)
    if missing and current_round > 1:
        plural = "s" if len(missing) > 1 else ""
        calls = ", ".join(f"query_history({p}, <round>)" for p in missing)
        return EvidenceVerdict(
            True,
            f"you attributed a past statement to player{plural} "
            f"{', '.join(map(str, missing))} without checking the record; "
            f"call {calls} first, then say it",
            referenced=sorted(refs),
        )

    ballot_refs = referenced_ballots(content)
    if ballot_refs and current_round > 1 and not queried_vote_rounds:
        return EvidenceVerdict(
            True,
            "you described how someone voted without reading the ballot; "
            "call query_votes(<round>) first",
            referenced=sorted(refs),
        )

    return EvidenceVerdict.ok(sorted(refs))


def unsupported_citations(
    content: str,
    *,
    speaker: int,
    lookups: dict[int, str],
) -> list[dict]:
    """Axis 6 (hallucination): citations whose content does not appear in what
    the agent actually retrieved.

    `lookups` maps player id -> the concatenated text returned by
    query_history for that player this turn.
    """
    problems = []
    for pid in sorted(referenced_players(content) - {speaker}):
        retrieved = lookups.get(pid)
        if retrieved is None:
            problems.append({"player_id": pid, "kind": "no_lookup"})
            continue
        quoted = _quoted_fragments(content)
        for frag in quoted:
            if len(frag) > 8 and frag.lower() not in retrieved.lower():
                problems.append(
                    {"player_id": pid, "kind": "quote_not_found", "quote": frag[:120]}
                )
    return problems


def _quoted_fragments(text: str) -> list[str]:
    return re.findall(r"[\"'“」]([^\"'“”」]{6,160})[\"'”」]", text or "")
