"""Model cascade: a cheap model by default, a strong one when it is worth paying for.

Every turn used to go to the same model, so the only cost lever was "a cheaper
model everywhere" -- and a cheap model's failures (a malformed reply, a tool
that does not exist, a speech the guard refuses) are paid for twice: the wasted
call, and the retry that often fails the same way. A cascade spends the strong
model only where it buys something:

  (a) on failure, inside the turn: the cheap model's reply did not pass a gate
      (schema, whitelist, the ActionSpec, an output-side guard, an unsupported
      citation) or the turn ran out of steps. The retry goes to the strong model
      with the SAME transcript -- the cheap model's bad reply and the guard's
      reason included -- so the strong model sees exactly what went wrong.
  (b) decisive turns, by policy: the turns where one bad choice moves the game
      (a ballot, a night kill/check/potion, the hunter's shot, the badge) start
      on the strong model.
  (c) low confidence (optional): a speech that names no seat at all is the
      cheapest model's tell -- it says nothing anyone can act on. It is asked
      again, on the strong model, with the transcript as it was before that reply.

Escalation is sticky for the rest of the turn and never goes back down: a turn
that needed the strong model once is not handed back to the model that failed it.
The note to the next turn always goes to the cheap model -- it is bookkeeping,
and the turn's transcript already holds whatever the strong model decided.

This module is the policy only: small, explicit, serialisable (it travels in a
game log's settings and in TableConfig.cascade). The mechanics live in the loop
(players/llm/loop.py): it asks ``start()`` once per turn and ``escalates_on()``
at every failure, and records each call's model and each escalation's reason.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Every reason a turn may move to the strong model, by name. The loop only ever
# passes these (a guard block as "guard:<check>", which matches "guard").
FAIL_REASONS = (
    "schema",                # no usable tool call / an info tool's arguments fail its schema
    "unknown_tool",          # a tool this turn does not offer
    "illegal_args",          # a terminal (or info tool) whose arguments the ActionSpec refuses
    "guard",                 # an output-side guard block: guard:evidence, guard:say_do_consistency, ...
    "unsupported_citation",  # a speech citing what the seat never looked up
    "forced",                # out of ReAct steps: the forced terminal goes to the strong model
    "transport",             # the cheap model did not answer at all (after its own retries)
)
POLICY_REASONS = ("decisive", "low_confidence")
REASONS = FAIL_REASONS + POLICY_REASONS

# The turns where one choice moves the game. Task names or terminal tool names
# (a night turn's task is "night" for every role; its tools say whose night it is).
DEFAULT_DECISIVE = ("vote", "night", "hunter_shoot", "badge")
MODES = ("off", "on-fail", "decisive", "both")

SPEECH_TERMINALS = ("speak", "campaign_run", "last_words")
_SEAT_REF = re.compile(r"(?<!\d)\d{1,2}\s*号|(?i:\bplayer\s*#?\d{1,2}\b)|(?<!\d)#\d{1,2}\b")


def cites_nothing(content: str) -> bool:
    """A speech that names no seat: no "3号", no "player 3", no "#3".

    Deliberately crude. It is a trigger for asking again, not a judgement of the
    speech, so the only cost of a false positive is one strong call."""
    return not _SEAT_REF.search(content or "")


@dataclass(frozen=True)
class CascadePolicy:
    on_fail: bool = True
    decisive: frozenset = field(default_factory=frozenset)
    low_confidence: bool = False
    fail_reasons: frozenset = frozenset(FAIL_REASONS)

    # ------------------------------------------------------------ construction
    @classmethod
    def from_mode(cls, mode: str, *, decisive=None, low_confidence: bool = False,
                  fail_reasons=None) -> CascadePolicy | None:
        """`off` -> None (no cascade); `on-fail`, `decisive`, `both` as named."""
        if mode in (None, "", "off"):
            return None
        if mode not in MODES:
            raise ValueError(f"cascade mode {mode!r}: one of {MODES}")
        dec = frozenset(decisive if decisive is not None else DEFAULT_DECISIVE) if mode in ("decisive", "both") else frozenset()
        fr = frozenset(fail_reasons) if fail_reasons is not None else frozenset(FAIL_REASONS)
        unknown = fr - set(FAIL_REASONS)
        if unknown:
            raise ValueError(f"unknown cascade fail reasons {sorted(unknown)}; known: {FAIL_REASONS}")
        return cls(on_fail=mode in ("on-fail", "both"), decisive=dec, low_confidence=low_confidence, fail_reasons=fr)

    @classmethod
    def from_dict(cls, d: dict | None) -> CascadePolicy | None:
        if not d:
            return None
        return cls(on_fail=bool(d.get("on_fail", True)), decisive=frozenset(d.get("decisive") or ()),
                   low_confidence=bool(d.get("low_confidence", False)),
                   fail_reasons=frozenset(d.get("fail_reasons") or FAIL_REASONS))

    def to_dict(self) -> dict:
        # sorted: the dict goes into logs and cassette headers, which are compared byte for byte
        return {"on_fail": self.on_fail, "decisive": sorted(self.decisive), "low_confidence": self.low_confidence,
                "fail_reasons": sorted(self.fail_reasons)}

    @property
    def mode(self) -> str:
        return {(True, True): "both", (True, False): "on-fail", (False, True): "decisive"}.get(
            (self.on_fail, bool(self.decisive)), "off")

    # ------------------------------------------------------------ decisions
    def start(self, task: str, tool_names) -> tuple[str, str | None]:
        """Which tier a turn starts on, and why: ("strong", "decisive") or ("cheap", None)."""
        if self.decisive and (task in self.decisive or any(n in self.decisive for n in tool_names)):
            return "strong", "decisive"
        return "cheap", None

    def escalates_on(self, reason: str) -> bool:
        """Whether a cheap-tier failure (or signal) with this reason moves the turn to the strong model."""
        if reason == "low_confidence":
            return self.low_confidence
        if reason == "decisive":
            return bool(self.decisive)
        return self.on_fail and reason.split(":", 1)[0] in self.fail_reasons

    def low_confidence_reply(self, name: str, args: dict) -> bool:
        """Signal (c): a speech-like terminal that names no seat."""
        return self.low_confidence and name in SPEECH_TERMINALS and cites_nothing(str(args.get("content", "")))


def reason_family(reason: str) -> str:
    """'guard:evidence' -> 'guard'; for grouping escalation rates."""
    return reason.split(":", 1)[0]
