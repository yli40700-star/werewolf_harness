"""Werewolf, written on the Arena protocol.

Everything below is game content. The framework (``arena/``) contributes the
five-method contract, the driver loop and replay; nothing here is visible to it
except through ``reset / observe / action_space / step / rules_text``.

Design rules this file follows (see FRAMEWORK notes in README):

* the world lives in private fields; ``step`` is the only writer;
* ``observe`` is the only information exit -- a seat gets a fresh dict with
  the public record plus its own private facts and nothing else;
* every terminal action is a ``ToolSpec`` whose legal arguments are computed
  from state, so the list the model reads is the list ``step`` checks;
* every death goes through ``_kill``; every phase change through ``_settle``;
  "who is next" is derived by ``_next`` from state, never remembered;
* rules that could plausibly be played another way are parameters on the
  ``Variant`` and the rules text is rendered from them.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..arena import ActionSpec, Param, Terminal, ToolSpec, Turn
from ..players.strategies.playbook import (load_playbook, plan_for, plan_labels,
                                           strategy_ids, strategy_labels)

# ---------------------------------------------------------------- variants

MAX_ROUNDS = 6
SHERIFF_VOTE_WEIGHT = 1.5
SPEECH_MAX_CHARS = 1200
PACK_SAY_MAX_CHARS = 600
REVIEW_SAY_MAX_CHARS = 800
THOUGHT_MAX_CHARS = 500      # the reasoning that rides along with a terminal action

ROLE_ORDER = ("werewolf", "seer", "witch", "hunter", "guard", "villager")
_PLURAL = {"werewolf": "werewolves", "witch": "witches"}


# ---------------------------------------------------------------- engine versions
#
# world = f(seed, journal) holds only for a fixed ``step``. The version names that
# function: a game is created under one version and is replayed under that same
# version forever (crash recovery, audit, the review page), whatever code is deployed.
#
# What counts. ENGINE_VERSION moves when, for some (seed, journal), the world after
# replay could differ: the deal in ``reset``, what ``step`` does, who ``_next`` asks,
# and which actions ``action_space`` makes LEGAL (``step`` refuses an illegal one, so
# legality is step behaviour). It does NOT move for what a seat is SHOWN -- ``observe``,
# tool descriptions, ``rules_text``, ``open_turns``. Views are not in the journal; they
# are recomputed from the world on every read, so an old game recovered under new view
# code is the same world, rendered the new way. The precedent is VIEW_REVISION 2
# (sealed ballots, review rounds written at once): it changed what a voter sees while
# the vote runs and which turns the driver may ask early, and every journal written
# before it still replays step for step -- so it was a view revision, not an engine one.
# VIEW_REVISION is recorded beside the engine version for provenance (it explains
# why the same journal produced different model behaviour) but never selects code.
#
# How. A step-relevant change is one entry in RULE_CHANGES: a flag and the version that
# introduced it. ``Rules.at(v)`` turns a version into flags, and the engine branches on
# the flag (``if self.rules.tie_revote:``), never on a version number and never by
# copying the engine. Retiring a version (raising MIN_ENGINE_VERSION) is what lets a
# branch be deleted; see docs/adr/0010-engine-versions.md for the support window.

@dataclass(frozen=True)
class RuleChange:
    flag: str             # the attribute on Rules the engine branches on
    introduced: int       # first engine version with the new behaviour
    summary: str


RULE_CHANGES: tuple[RuleChange, ...] = (
    RuleChange("tie_revote", 2,
               "平票 PK: a tied day vote is re-run once between the tied players (they do not "
               "vote); a second tie exiles nobody. Before v2 a tie exiled nobody at once."),
)

BASELINE_ENGINE_VERSION = 1      # the rules every unversioned journal was written under
MIN_ENGINE_VERSION = 1           # the oldest version this code can still replay
ENGINE_VERSION = max([BASELINE_ENGINE_VERSION] + [c.introduced for c in RULE_CHANGES])
VIEW_REVISION = 2                # informational: see above


class UnsupportedEngineVersion(ValueError):
    """A game recorded under a version this code cannot reproduce: too old (its
    branches were retired) or too new (written by a newer node -- a rolling deploy)."""


@dataclass(frozen=True)
class Rules:
    """The step-relevant rule set of one engine version, as flags."""
    version: int
    tie_revote: bool = False

    @classmethod
    def at(cls, version: int) -> Rules:
        if not isinstance(version, int) or isinstance(version, bool):
            raise UnsupportedEngineVersion(f"engine version must be an int, got {version!r}")
        if version < MIN_ENGINE_VERSION or version > ENGINE_VERSION:
            raise UnsupportedEngineVersion(
                f"engine v{version} is outside what this code replays (v{MIN_ENGINE_VERSION}..v{ENGINE_VERSION})")
        return cls(version, **{c.flag: version >= c.introduced for c in RULE_CHANGES})

    def changed_since(self, older: int) -> list[RuleChange]:
        """The rule changes between ``older`` and this version: what a diverging replay can blame."""
        return [c for c in RULE_CHANGES if older < c.introduced <= self.version]


def engine_version_of(record: dict | None) -> int:
    """The version a stored game was made with; a game from before versioning is the baseline."""
    v = (record or {}).get("engine_version")
    return BASELINE_ENGINE_VERSION if v is None else int(v)


@dataclass(frozen=True)
class Variant:
    name: str
    role_counts: dict[str, int]
    sheriff: bool = False
    last_words: bool = False
    pack_planning: bool = False
    # the hard cap on the pack's night discussion, counted in ROUNDS (one round = every
    # living wolf gets one turn). The wolves normally stop earlier by all passing; the cap
    # is what stops a pack that keeps talking. Every extra round costs one model call per
    # living wolf, so this number is a token budget as much as a rule.
    chat_passes: int = 3
    # the hard cap on the post-game review (复盘), counted in ROUNDS (one round = every
    # seat, living and dead, gets one turn). 0 turns the review off entirely: no review
    # phase, no review turns, and no review sentence in the rules text. Two rounds is the
    # default -- one to say what you saw and got wrong, one to answer what others said --
    # because every round is one model call per SEAT (12p = 12 calls a round, so the
    # default costs about as much as one day of speeches).
    review_rounds: int = 2
    # rule parameters: each one is both an engine branch and a rules sentence
    guard_self: bool = True            # the guard may protect himself
    guard_witch_overlap_kills: bool = False   # 同守同救: True = they cancel and the player dies
    witch_self_save: bool = False      # the witch may use the antidote on herself
    hunter_shoots_on: frozenset = frozenset({"werewolf", "vote", "witch", "hunter"})
    # what a tied day vote does is an engine rule, not a variant knob: see Rules.tie_revote

    @property
    def num_players(self) -> int:
        return sum(self.role_counts.values())

    def __post_init__(self):
        if self.num_players <= 0:
            raise ValueError("a variant needs at least one seat")


VARIANTS: dict[str, Variant] = {
    "8p": Variant("8p", {"werewolf": 2, "seer": 1, "witch": 1, "villager": 4}),
    "8p+sheriff": Variant("8p+sheriff", {"werewolf": 2, "seer": 1, "witch": 1, "villager": 4},
                          sheriff=True, last_words=True),
    "12p": Variant("12p", {"werewolf": 4, "seer": 1, "witch": 1, "hunter": 1, "guard": 1, "villager": 4},
                   sheriff=True, last_words=True, pack_planning=True),
    "15p": Variant("15p", {"werewolf": 5, "seer": 1, "witch": 1, "hunter": 1, "guard": 1, "villager": 6},
                   sheriff=True, last_words=True, pack_planning=True),
}

NIGHT_ACTIONS: dict[str, tuple[str, ...]] = {
    "seer": ("night_check", "night_skip"),
    "werewolf": ("night_kill", "night_skip"),
    "witch": ("night_save", "night_poison", "night_skip"),
    "guard": ("night_protect", "night_skip"),
    "villager": (),
    "hunter": (),
}
NIGHT_PRIORITY = {"seer": 10, "guard": 20, "werewolf": 30, "witch": 40}
# Night roles whose choice reads nothing another night actor writes: the seer's check
# and the guard's shield do not depend on the knife, and nothing they do is shown to
# anyone else before dawn. The wolves read each other's intents and the witch reads
# the pack's choice, so they wait their turn.
INDEPENDENT_AT_NIGHT = frozenset({"seer", "guard"})

TASK_TERMINALS: dict[str, tuple[str, ...]] = {
    "night": ("night_check", "night_kill", "night_save", "night_poison", "night_protect", "night_skip"),
    "pack_plan": ("pack_plan",),
    "pack_chat": ("pack_say", "pack_pass"),
    "speak": ("speak", "self_detonate"),
    "vote": ("vote", "self_detonate"),
    "campaign": ("campaign_run", "campaign_pass", "self_detonate"),
    "campaign_vote": ("campaign_vote",),
    "last_words": ("last_words",),
    "hunter_shoot": ("hunter_shoot", "hunter_hold"),
    "badge": ("badge_transfer", "badge_tear"),
    "review": ("review_say", "review_pass"),
}


def team_of(role: str) -> str:
    return "werewolf" if role == "werewolf" else "village"


# ---------------------------------------------------------------- records

@dataclass(frozen=True)
class Speech:
    round: int
    player_id: int
    order: int
    content: str
    kind: str = "speech"          # speech | campaign | last_words

    def to_dict(self) -> dict:
        return {"round": self.round, "player_id": self.player_id, "order": self.order,
                "content": self.content, "kind": self.kind}


@dataclass(frozen=True)
class Death:
    round: int
    player_id: int
    cause: str                    # werewolf | witch | vote | hunter | detonate


# ---------------------------------------------------------------- the environment

class Werewolf:
    """Seats are 1..N. Tasks are the keys of TASK_TERMINALS."""

    def __init__(self, variant: str | Variant = "12p", playbook_mode: str = "static",
                 engine_version: int | None = None):
        """``engine_version`` selects the rules a game was created under (None = the
        current ones). A replay must pass the version the game recorded."""
        self.variant = VARIANTS[variant] if isinstance(variant, str) else variant
        if playbook_mode not in ("none", "static", "dynamic"):
            raise ValueError(f"unknown playbook mode: {playbook_mode!r}")
        self.playbook_mode = playbook_mode
        self.rules = Rules.at(ENGINE_VERSION if engine_version is None else engine_version)
        self.engine_version = self.rules.version
        self._playbook = load_playbook()

    # ================================================================ reset
    def reset(self, seed: int) -> Turn | Terminal:
        v = self.variant
        rng = random.Random(seed)
        seats = list(range(1, v.num_players + 1))
        names: list[str] = []
        for role, n in v.role_counts.items():
            names.extend([role] * n)
        rng.shuffle(names)
        self._roles: dict[int, str] = dict(zip(seats, names))
        self._order = seats
        self._fallback_order = seats[:]
        rng.shuffle(self._fallback_order)

        # frozenset, not set: every read below (in / sorted / difference) works the
        # same, but `self._alive.discard(x)` -- the natural thing to type when adding
        # a new way to die -- is now an AttributeError instead of a silent bypass of
        # the death funnel. Rebinding it is what _kill does, and only _kill.
        self._alive: frozenset[int] = frozenset(seats)
        self._round = 1
        self._phase = "night"                       # night | pack_chat | pack_plan | campaign | campaign_vote | day_speech | day_vote | interrupt | review | over
        self._deaths: list[Death] = []
        self._speeches: list[Speech] = []
        self._votes: dict[int, dict[int, int | None]] = {}
        self._exiles: dict[int, int | None] = {}
        # 平票 PK (rules.tie_revote): the first ballot of a tied day, and who stands in the revote
        self._tie_ballots: dict[int, dict[int, int | None]] = {}
        self._revote: list[int] = []         # non-empty while a revote is running
        self._action_log: dict[int, list[dict]] = {s: [] for s in seats}

        # night
        self._night_queue: list[int] = []
        self._night_done: set[int] = set()
        self._wolf_intents: dict[int, int] = {}
        self._wolf_order: list[int] = []
        self._guarded: int | None = None
        self._guard_last: int | None = None
        self._saved: int | None = None
        self._poisoned: int | None = None
        self._antidote = True
        self._poison = True
        self._seer_checks: list[dict] = []
        self._night_victims: dict[int, list[int]] = {}
        self._wasted_knives: list[dict] = []
        self._night_records: list[dict] = []

        # pack
        self._pack_chat: list[dict] = []
        self._chat_cursor = 0        # index into the living wolves within the current round
        self._chat_round = 0         # rounds of discussion already finished
        self._chat_quiet = True      # nobody has spoken yet in the round now running
        self._chat_done = False      # the discussion is over (all passed, or the cap is reached)
        self._pack_plan: str | None = None
        self._pack_roles: dict[int, str] = {}
        self._pack_vote_target: int | None = None
        self._pack_planned = False
        self._planned_round = 0
        self._plan_history: list[dict] = []

        # sheriff
        self._sheriff: int | None = None
        self._sheriff_settled = False
        self._candidates: list[int] = []
        self._sheriff_votes: dict[int, int | None] = {}
        self._sheriff_cursor = 0
        self._badge_transfers: list[dict] = []

        # day
        self._speech_cursor = 0
        self._day_voters: list[int] = []
        self._vote_weights: dict[int, float] = {}
        self._detonations: list[dict] = []
        self._shots: list[dict] = []

        # interrupts
        self._pending_hunter: int | None = None
        self._pending_badge: int | None = None
        self._pending_last_words: list[int] = []
        self._last_words_given: set[int] = set()
        self._settle_to: str | None = None

        # the result, frozen the moment a win condition fires. `_finish` is the only
        # writer and it writes once; every later read (including every review turn) is a
        # read of this same dict, which is what makes the review unable to change it.
        self._winner: str | None = None
        self._outcome: dict | None = None

        # review (复盘): after the result is fixed, before the terminal
        self._review_chat: list[dict] = []
        self._review_cursor = 0      # index into the seats within the current round
        self._review_round = 0       # rounds of review already finished
        self._review_quiet = True    # nobody has spoken yet in the round now running
        self._review_done = False    # the review is over (all passed, or the cap is reached)

        self._begin_night()
        return self._next()

    # ================================================================ observe
    def observe(self, actor: int) -> dict[str, Any]:
        if actor not in self._roles:
            raise ValueError(f"no such player: {actor}")
        role = self._roles[actor]
        order = self._speech_order()
        v: dict[str, Any] = {
            "you": actor,
            "role": role,
            "team": team_of(role),
            "alive": actor in self._alive,
            "round": self._round,
            "phase": self._phase,
            "num_players": self.variant.num_players,
            "rules": self.rules_text(),
            "public": {
                "alive": sorted(self._alive),
                "dead": [{"player_id": d.player_id, "round": d.round, "cause": self._public_cause(d.cause)}
                         for d in self._deaths],
                "speech_order": order,
                "speeches": [s.to_dict() for s in self._speeches],
                "votes": {r: dict(b) for r, b in self._votes.items() if self._ballots_open(r)},
                "vote_counts": {r: dict(Counter(t for t in b.values() if t is not None))
                                for r, b in self._votes.items() if self._ballots_open(r)},
                "exiles": dict(self._exiles),
                "sheriff": self._sheriff,
                "sheriff_candidates": list(self._candidates),
                "badge_transfers": list(self._badge_transfers),
                "shots": list(self._shots),
                "detonations": [{"round": d["round"], "player_id": d["player_id"]} for d in self._detonations],
            },
            "speech_position": (order.index(actor) + 1) if actor in order else None,
            "speakers_total": len(order),
            "private": {"my_actions": list(self._action_log[actor][-14:])},
        }
        if self.rules.tie_revote:              # only under the rule, so older games' views do not move
            v["public"]["tie_votes"] = {r: dict(b) for r, b in self._tie_ballots.items()}
            v["public"]["revote_candidates"] = list(self._revote)
        priv = v["private"]
        if role == "werewolf":
            priv["fellow_wolves"] = [s for s in self._team("werewolf") if s in self._alive and s != actor]
            priv["pack_plan"] = self._pack_plan
            priv["pack_roles"] = dict(self._pack_roles)
            priv["my_tactic"] = self._pack_roles.get(actor)
            priv["staffable_plans"] = [p.id for p in plan_for(len(self._living_wolves()))]
            priv["pack_vote_target"] = self._pack_vote_target
            priv["wasted_knives"] = list(self._wasted_knives)
            priv["pack_chat"] = list(self._pack_chat[-12:])
            if self._phase == "pack_chat":
                priv["pack_chat_round"] = self._chat_round + 1
                priv["pack_chat_rounds_max"] = self._chat_round_cap()
        if role == "seer":
            priv["checks"] = list(self._seer_checks)
        if role == "witch":
            priv["antidote_available"] = self._antidote
            priv["poison_available"] = self._poison
        if self._phase == "night" and actor in self._open_night_actors():
            v["night"] = self._night_options(actor)
        if self._phase == "review":
            v["review"] = self._review_handout()
        return v

    def _review_handout(self) -> dict[str, Any]:
        """THE one deliberate hole in the isolation guarantee, and it is a hole in time.

        While ``_phase == "review"`` -- and only then, which is to say only after
        ``_finish`` has already frozen the result -- every seat is handed the whole truth
        of the game: every role, every night action, the seer's checks, the wolves' private
        channel and their plans. Nothing here can be read one turn earlier, because the
        phase is not ``review`` one turn earlier; and nothing read here can matter to the
        result, because the result is already a frozen dict.
        """
        return {
            "winner": self._winner,
            "rounds_played": self._round,
            "roles": dict(self._roles),
            "teams": {s: team_of(r) for s, r in self._roles.items()},
            "survivors": sorted(self._alive),
            "deaths": [{"round": d.round, "player_id": d.player_id, "cause": d.cause} for d in self._deaths],
            "night_actions": [dict(r) for r in self._night_records],
            "seer_checks": [dict(c) for c in self._seer_checks],
            "wasted_knives": [dict(k) for k in self._wasted_knives],
            "pack_chat": [dict(e) for e in self._pack_chat],
            "pack_plans": [{**h, "roles": dict(h["roles"]), "changed": dict(h["changed"])} for h in self._plan_history],
            "pack_vote_target": self._pack_vote_target,
            "detonations": [dict(d) for d in self._detonations],
            "shots": [dict(x) for x in self._shots],
            "badge_transfers": [dict(x) for x in self._badge_transfers],
            "sheriff": self._sheriff,
            "speeches": [sp.to_dict() for sp in self._speeches],
            "votes": {r: dict(b) for r, b in self._votes.items()},
            "exiles": dict(self._exiles),
            # this round's remarks stay hidden until the round is over: every seat writes
            # its round at once, so each one answers what was said before, never beside it
            "said": [dict(e) for e in self._review_chat if e["round"] <= self._review_round],
            "round": self._review_round + 1,
            "rounds_max": self._review_cap(),
            "seats": list(self._review_seats()),
        }

    # ================================================================ concurrency
    def open_turns(self) -> list[Turn]:
        """Every turn that may be decided right now, the one ``step`` expects next first.

        Optional part of the arena contract. A turn is open when nothing committed
        before it, in journal order, can change what its actor sees or may do --
        so a driver may ask all of them at once and still commit them one by one,
        in the order ``step`` asks for them. The journal, and with it replay, is
        exactly what the one-at-a-time loop would have written.

          day vote / badge vote  every voter: ballots are sealed until the last one is in
          review                 every seat left in this round: a round is written at once
          night                  the seer and the guard, the first wolf still to name a
                                 kill, and the witch once the pack has chosen
          everything else        nothing: a speech answers the speech before it

        Pure read: never moves the world.
        """
        if self._phase == "day_vote":
            cast = self._votes.get(self._round, {})
            return [Turn(s, "vote") for s in self._day_voters if s in self._alive and s not in cast]
        if self._phase == "campaign_vote":
            return [Turn(s, "campaign_vote") for s in self._electorate() if s not in self._sheriff_votes]
        if self._phase == "night":
            return [Turn(s, "night") for s in self._open_night_actors()]
        if self._phase == "review":
            if self._review_speaker() is None:
                return []
            return [Turn(s, "review") for s in self._review_seats()[self._review_cursor:]]
        return []

    def _ballots_open(self, rnd: int) -> bool:
        """A day's ballots are public once that vote is over, never while it is running."""
        return not (self._phase == "day_vote" and rnd == self._round)

    def _open_night_actors(self) -> list[int]:
        if self._phase != "night":
            return []
        pending = [s for s in self._night_queue if s not in self._night_done and s in self._alive]
        wolves_left = [s for s in pending if self._roles[s] == "werewolf"]
        out = []
        for s in pending:
            role = self._roles[s]
            if role in INDEPENDENT_AT_NIGHT \
                    or (role == "werewolf" and s == wolves_left[0]) \
                    or (role not in ("werewolf",) and not wolves_left):
                out.append(s)
        return out

    # ================================================================ action_space
    def action_space(self, actor: int, task: str) -> ActionSpec:
        tools = tuple(self._tool(name, actor) for name in self._terminals_for(actor, task))
        fallback = self._fallback(actor, task, tools)
        return ActionSpec("tool", tools=tools, fallback=fallback)

    def _terminals_for(self, actor: int, task: str) -> list[str]:
        role = self._roles[actor]
        if task == "night":
            return list(self._night_allowed(actor))
        names = list(TASK_TERMINALS[task])
        if "self_detonate" in names and role != "werewolf":
            names.remove("self_detonate")
        return names

    # -- one ToolSpec per terminal, legal arguments computed from state ----
    # Every terminal carries the reasoning that chose it. With tool_choice=required
    # a model returns a call and NO text, so the reasoning has nowhere else to go --
    # the trace showed a list of tool names and nothing about why. Optional, not
    # required: a fallback has no reasoning to give, and a turn must never be
    # refused for want of commentary.
    THOUGHT = Param("str", "为什么选这一步：你现在怀疑谁、依据是什么、这一手想达成什么。"
                           "写给复盘看的，桌上其他人看不到",
                    required=False, max_len=THOUGHT_MAX_CHARS)

    def _tool(self, name: str, actor: int) -> ToolSpec:
        alive = sorted(self._alive)
        others = [s for s in alive if s != actor]
        reason = Param("str", "why, in one or two sentences", max_len=400, non_empty=True)
        text = lambda desc, n=SPEECH_MAX_CHARS: Param("str", desc, max_len=n, non_empty=True)

        if name == "night_check":
            return ToolSpec(name, "SEER ONLY, at night: learn whether a player is a werewolf. ENDS YOUR TURN.",
                            {"target_id": Param("int", "a living player you have not checked", choices=tuple(self._seer_targets(actor))),
                             "thought": self.THOUGHT})
        if name == "night_kill":
            return ToolSpec(name, "WEREWOLF ONLY, at night: name the player the pack should kill. Your packmates name one too; the majority decides. ENDS YOUR TURN.",
                            {"target_id": Param("int", "a living non-wolf", choices=tuple(self._kill_targets())),
                             "thought": self.THOUGHT})
        if name == "night_save":
            return ToolSpec(name, "WITCH ONLY, at night: spend the antidote on tonight's victim. ENDS YOUR TURN.",
                            {"target_id": Param("int", "tonight's victim", choices=tuple(self._save_targets(actor))),
                             "thought": self.THOUGHT})
        if name == "night_poison":
            return ToolSpec(name, "WITCH ONLY, at night: spend the poison on a living player who is not already tonight's victim. ENDS YOUR TURN.",
                            {"target_id": Param("int", "a living player", choices=tuple(self._poison_targets(actor))),
                             "thought": self.THOUGHT})
        if name == "night_protect":
            return ToolSpec(name, "GUARD ONLY, at night: protect a living player from the wolves. You may never protect the same player two nights running. ENDS YOUR TURN.",
                            {"target_id": Param("int", "a living player, not last night's charge", choices=tuple(self._guard_targets(actor))),
                             "thought": self.THOUGHT})
        if name == "night_skip":
            return ToolSpec(name, "At night: do nothing this night. ENDS YOUR TURN.", {"thought": self.THOUGHT})
        if name == "speak":
            return ToolSpec(name, "Say something to the table. ENDS YOUR TURN.", {"content": text("what you say out loud"), "thought": self.THOUGHT})
        if name == "vote" and self._revote:
            return ToolSpec(name, "平票 PK revote: choose which of the tied players is exiled. ENDS YOUR TURN.",
                            {"target_id": Param("int", "one of the tied players", choices=tuple(self._revote)),
                             "thought": self.THOUGHT})
        if name == "vote":
            return ToolSpec(name, "Cast your vote to exile a player. ENDS YOUR TURN.",
                            {"target_id": Param("int", "a living player, not yourself", choices=tuple(others)),
                             "thought": self.THOUGHT})
        if name == "self_detonate":
            return ToolSpec(name, "WEREWOLF ONLY: reveal that you are a werewolf. The day stops immediately -- no vote, no exile, no last words for you, and if you hold the badge it is destroyed. Night follows, with you out of the game and the pack one wolf smaller. ENDS YOUR TURN.",
                            {"reason": reason,
                             "thought": self.THOUGHT})
        if name == "pack_say":
            return ToolSpec(name, "WEREWOLF ONLY: speak in the pack's private night channel. Only living wolves hear this. Name the tactic you want if you have one in mind. ENDS YOUR TURN.",
                            {"content": Param("str", "what you say to your packmates", max_len=PACK_SAY_MAX_CHARS, non_empty=True),
                             "tactic": Param("str", "the tactic id you are asking for", required=False,
                                             choices=tuple(strategy_ids()), labels=strategy_labels()),
                             "thought": self.THOUGHT})
        if name == "pack_pass":
            return ToolSpec(name, f"WEREWOLF ONLY: say nothing more in the pack's night channel -- 我没什么补充了。The discussion ends as soon as every living wolf passes in the same round, and in any case after {self._chat_round_cap()} rounds. If a packmate says something new afterwards you get another turn and may speak then. ENDS YOUR TURN.", {"thought": self.THOUGHT})
        if name == "review_say":
            return ToolSpec(name, "复盘 (post-game review): say something to the whole table about the game that has just finished. "
                            "Everyone hears it, living and dead, and the result of the game is already settled and cannot move. ENDS YOUR TURN.",
                            {"content": Param("str", "what you want to say about the game just played",
                                              max_len=REVIEW_SAY_MAX_CHARS, non_empty=True),
                             "thought": self.THOUGHT})
        if name == "review_pass":
            return ToolSpec(name, f"复盘 (post-game review): say nothing more -- 我复盘完了。The review ends as soon as every seat passes in the "
                            f"same round, and in any case after {self._review_cap()} rounds. If somebody says something afterwards you get another "
                            "turn and may speak then. ENDS YOUR TURN.",
                            {"thought": self.THOUGHT})
        if name == "pack_plan":
            wolves = self._living_wolves()
            staffable = tuple(p.id for p in plan_for(len(wolves)))
            def validate(args: dict) -> str | None:
                roles = args.get("roles") or {}
                unknown = sorted(set(roles.values()) - set(strategy_ids()))
                if unknown:
                    return f"unknown tactic(s): {unknown}"
                off = sorted(set(roles) - set(wolves))
                if off:
                    return f"player(s) {off} are not living wolves; the pack is {wolves}"
                missing = sorted(set(wolves) - set(roles))
                if missing:
                    return f"every living wolf needs a tactic; {missing} have none"
                return None
            return ToolSpec(name, "WEREWOLF ONLY: commit the pack. Name a plan id as the shape you are going for, and give EVERY living wolf a tactic -- any tactic id from list_strategies, not only the plan's own. ENDS YOUR TURN.",
                            {"plan": Param("str", "a plan id from list_plans", choices=staffable,
                                           labels=plan_labels()),
                             "roles": Param("dict[int,str]", "每一个活着的狼都要给一个战术 id，不能留空；可以照方案的组合，也可以自己配",
                                            choices=tuple(strategy_ids()), labels=strategy_labels()),
                             "vote_target": Param("int", "狼队明天白天约定投谁（一个活着的非狼座位）。约定不是命令，队友仍可自行判断；留空表示不约定",
                                                  required=False, choices=tuple(self._kill_targets())),
                             "thought": self.THOUGHT},
                            validate=validate)
        if name == "campaign_run":
            return ToolSpec(name, "Stand for sheriff and make your campaign speech. The sheriff speaks first each day and casts 1.5 votes. Candidates do not vote in the election. ENDS YOUR TURN.",
                            {"content": text("why the table should elect you"),
                             "thought": self.THOUGHT})
        if name == "campaign_pass":
            return ToolSpec(name, "Stand down: do not run for sheriff. You keep your vote in the election. ENDS YOUR TURN.", {"reason": reason, "thought": self.THOUGHT})
        if name == "campaign_vote":
            return ToolSpec(name, "Vote for one of the candidates for sheriff, or abstain by giving no target. ENDS YOUR TURN.",
                            {"target_id": Param("int", "a standing candidate", required=False, choices=tuple(self._candidates)),
                             "thought": self.THOUGHT})
        if name == "last_words":
            return ToolSpec(name, "You are dead. Say one last thing to the table; nobody can question you about it. ENDS YOUR TURN.",
                            {"content": text("your final speech"),
                             "thought": self.THOUGHT})
        if name == "hunter_shoot":
            return ToolSpec(name, "HUNTER ONLY, on your death: take one living player with you. ENDS YOUR TURN.",
                            {"target_id": Param("int", "a living player", choices=tuple(others)), "reason": reason,
                             "thought": self.THOUGHT})
        if name == "hunter_hold":
            return ToolSpec(name, "HUNTER ONLY, on your death: do not fire. ENDS YOUR TURN.", {"reason": reason, "thought": self.THOUGHT})
        if name == "badge_transfer":
            return ToolSpec(name, "SHERIFF ONLY, on your death: pass the badge to a living player. ENDS YOUR TURN.",
                            {"target_id": Param("int", "a living player", choices=tuple(others)), "reason": reason})
        if name == "badge_tear":
            return ToolSpec(name, "SHERIFF ONLY, on your death: destroy the badge so nobody inherits it. ENDS YOUR TURN.", {"reason": reason, "thought": self.THOUGHT})
        raise KeyError(name)

    def _fallback(self, actor: int, task: str, tools: tuple[ToolSpec, ...]) -> dict:
        """The conservative default when a turn cannot be salvaged: abstention where the rules allow it."""
        names = {t.name for t in tools}
        if task == "night":
            return self._fallback_night(actor)
        if task == "speak":
            return {"name": "speak", "args": {"content": "I have nothing to add this round."}}
        if task == "vote":
            # the rules say there is no abstention, so the default is the first legal seat
            t = next(t for t in tools if t.name == "vote")
            return {"name": "vote", "args": {"target_id": t.params["target_id"].choices[0]}}
        if task == "pack_chat":
            # silence is the quiet default, and it also moves the discussion towards its end
            return {"name": "pack_pass", "args": {}}
        if task == "review":
            # silence is the quiet default, and it also moves the review towards its end
            return {"name": "review_pass", "args": {}}
        if task == "pack_plan":
            wolves = self._living_wolves()
            plans = plan_for(len(wolves))
            plan = plans[0] if plans else None
            spread = list(plan.roles) if plan else []
            roles = {w: (spread[i] if i < len(spread) else "shenshui") for i, w in enumerate(wolves)}
            return {"name": "pack_plan", "args": {"plan": plan.id if plan else "solo", "roles": roles, "vote_target": None}}
        if task == "campaign":
            return {"name": "campaign_pass", "args": {"reason": "the turn could not be salvaged"}}
        if task == "campaign_vote":
            return {"name": "campaign_vote", "args": {"target_id": None}}
        if task == "last_words":
            return {"name": "last_words", "args": {"content": "I have nothing to add."}}
        if task == "hunter_shoot":
            return {"name": "hunter_hold", "args": {"reason": "the turn could not be salvaged"}}
        if task == "badge":
            return {"name": "badge_tear", "args": {"reason": "the turn could not be salvaged"}}
        raise KeyError(f"no default action for task {task!r}")

    def _fallback_night(self, actor: int) -> dict:
        role = self._roles[actor]
        allowed = self._night_allowed(actor)
        if role == "witch" or allowed == ("night_skip",) or not allowed:
            return {"name": "night_skip", "args": {}}
        name = {"werewolf": "night_kill", "guard": "night_protect", "seer": "night_check"}[role]
        legal = set(self._night_targets(actor, name))
        target = next((s for s in self._fallback_order if s in legal), None)
        if target is None:
            return {"name": "night_skip", "args": {}}
        return {"name": name, "args": {"target_id": target}}

    # ================================================================ step
    def step(self, actor: int, task: str, action: dict) -> Turn | Terminal:
        spec = self.action_space(actor, task)
        reason = spec.check(action)
        if reason:
            raise RuntimeError(f"illegal action reached step(): {reason}")
        tool = spec.tool(action["name"])
        args = tool.normalise(action.get("args") or {})
        name = tool.name

        if name.startswith("night_"):
            self._apply_night(actor, name, args)
        elif name == "pack_say":
            self._apply_pack_say(actor, args)
        elif name == "pack_pass":
            self._apply_pack_pass(actor)
        elif name == "pack_plan":
            self._apply_pack_plan(actor, args)
        elif name == "review_say":
            self._apply_review_say(actor, args)
        elif name == "review_pass":
            self._apply_review_pass(actor)
        elif name == "campaign_run":
            self._candidates.append(actor)
            self._speeches.append(Speech(self._round, actor, len(self._speeches), args["content"], "campaign"))
            self._log(actor, "campaign_run", said=args["content"][:200])
            self._advance_campaign()
        elif name == "campaign_pass":
            self._log(actor, "campaign_pass", reason=args["reason"][:120])
            self._advance_campaign()
        elif name == "campaign_vote":
            self._sheriff_votes[actor] = args.get("target_id")
            self._log(actor, "campaign_vote", target=args.get("target_id"), abstained=args.get("target_id") is None)
            if len(self._sheriff_votes) >= len(self._electorate()):
                self._resolve_election()
        elif name == "speak":
            self._speeches.append(Speech(self._round, actor, self._speech_cursor, args["content"], "speech"))
            self._log(actor, "speak", said=args["content"][:200])
            self._speech_cursor += 1
            if self._speech_cursor >= len(self._speech_order()):
                self._phase = "day_vote"
                self._day_voters = sorted(self._alive)
                self._vote_weights = {self._sheriff: SHERIFF_VOTE_WEIGHT} if self._sheriff in self._alive else {}
        elif name == "vote":
            self._votes.setdefault(self._round, {})[actor] = args["target_id"]
            self._log(actor, "vote", target=args["target_id"])
            if all(v in self._votes[self._round] for v in self._day_voters):
                self._resolve_vote()
        elif name == "self_detonate":
            self._apply_detonate(actor, args["reason"])
        elif name == "last_words":
            self._speeches.append(Speech(self._round, actor, len(self._speeches), args["content"], "last_words"))
            self._log(actor, "last_words", said=args["content"][:200])
            self._pending_last_words.pop(0)
            self._last_words_given.add(actor)
            self._settle_if_clear()
        elif name in ("hunter_shoot", "hunter_hold"):
            target = args.get("target_id") if name == "hunter_shoot" else None
            self._pending_hunter = None
            self._shots.append({"round": self._round, "by": actor, "target": target})
            self._log(actor, name, target=target, reason=args["reason"][:120])
            if target is not None:
                self._kill(target, "hunter")
            self._settle_if_clear()
        elif name in ("badge_transfer", "badge_tear"):
            target = args.get("target_id") if name == "badge_transfer" else None
            self._pending_badge = None
            self._sheriff = target
            self._badge_transfers.append({"round": self._round, "from": actor, "to": target})
            self._log(actor, name, target=target, reason=args["reason"][:120])
            self._settle_if_clear()
        else:
            raise KeyError(name)
        return self._next()

    # ================================================================ night
    def _begin_night(self) -> None:
        self._phase = "night"
        self._night_done = set()
        self._wolf_intents = {}
        self._wolf_order = []
        self._guarded = None
        self._saved = None
        self._poisoned = None
        self._night_victims[self._round] = []
        if self._chat_due():
            self._phase = "pack_chat"
            self._chat_cursor = 0
            self._chat_round = 0
            self._chat_quiet = True
            self._chat_done = False
        elif self._plan_due():
            self._phase = "pack_plan"
        self._night_queue = self._night_actors()

    def _night_actors(self) -> list[int]:
        actors = [s for s in self._order if s in self._alive and NIGHT_ACTIONS[self._roles[s]]]
        actors = [s for s in actors if self._roles[s] != "witch" or self._antidote or self._poison]
        return sorted(actors, key=lambda s: (NIGHT_PRIORITY[self._roles[s]], s))

    def _night_actor(self) -> int | None:
        if self._phase != "night":
            return None
        for s in self._night_queue:
            if s not in self._night_done and s in self._alive:
                return s
        return None

    def _night_allowed(self, actor: int) -> tuple[str, ...]:
        role = self._roles[actor]
        out = []
        for name in NIGHT_ACTIONS[role]:
            if name == "night_skip":
                out.append(name)
            elif self._night_targets(actor, name):
                out.append(name)
        return tuple(out)

    def _night_targets(self, actor: int, name: str) -> list[int]:
        return {"night_check": self._seer_targets, "night_kill": lambda a: self._kill_targets(),
                "night_save": self._save_targets, "night_poison": self._poison_targets,
                "night_protect": self._guard_targets}[name](actor)

    def _seer_targets(self, actor: int) -> list[int]:
        checked = {c["target"] for c in self._seer_checks}
        return sorted(self._alive - {actor} - checked)

    def _kill_targets(self) -> list[int]:
        return sorted(s for s in self._alive if self._roles[s] != "werewolf")

    def _victims_so_far(self) -> list[int]:
        return [self._pack_choice()] if self._wolf_intents else []

    def _save_targets(self, actor: int) -> list[int]:
        if not self._antidote:
            return []
        victims = [v for v in self._victims_so_far() if self.variant.witch_self_save or v != actor]
        return victims

    def _poison_targets(self, actor: int) -> list[int]:
        if not self._poison:
            return []
        return sorted(self._alive - {actor} - set(self._victims_so_far()))

    def _guard_targets(self, actor: int) -> list[int]:
        out = self._alive - {self._guard_last} if self._guard_last is not None else set(self._alive)
        if not self.variant.guard_self:
            out = out - {actor}
        return sorted(out)

    def _night_options(self, actor: int) -> dict:
        role = self._roles[actor]
        opts: dict[str, Any] = {"actions": list(self._night_allowed(actor))}
        if role == "witch":
            opts["victims_tonight"] = self._victims_so_far()
            opts["antidote_available"] = self._antidote
            opts["poison_available"] = self._poison
        if role == "werewolf":
            opts["pack_votes_so_far"] = dict(self._wolf_intents)
        return opts

    def _apply_night(self, actor: int, name: str, args: dict) -> None:
        target = args.get("target_id")
        outcome = "skipped"
        if name == "night_check":
            is_wolf = self._roles[target] == "werewolf"
            self._seer_checks.append({"round": self._round, "target": target, "is_wolf": is_wolf})
            outcome = "werewolf" if is_wolf else "not a werewolf"
        elif name == "night_kill":
            self._wolf_intents[actor] = target
            self._wolf_order.append(actor)
            outcome = "noted; the pack decides together"
        elif name == "night_save":
            self._saved = target
            self._antidote = False
            outcome = "antidote used"
        elif name == "night_poison":
            self._poisoned = target
            self._poison = False
            outcome = "poison used"
        elif name == "night_protect":
            self._guarded = target
            outcome = "protected"
        self._night_records.append({"round": self._round, "player_id": actor, "action": name, "target": target, "outcome": outcome})
        self._action_log[actor].append({"round": self._round, "phase": "night", "kind": name, "target": target, "outcome": outcome})
        self._night_done.add(actor)
        if self._night_actor() is None:
            self._end_night()

    def _pack_choice(self) -> int | None:
        if not self._wolf_intents:
            return None
        counts = Counter(self._wolf_intents.values())
        top = max(counts.values())
        leaders = {t for t, n in counts.items() if n == top}
        for w in self._wolf_order:                      # tie: whoever named first
            if self._wolf_intents[w] in leaders:
                return self._wolf_intents[w]
        return sorted(leaders)[0]

    def _end_night(self) -> None:
        before = set(self._alive)
        knife = self._pack_choice()
        kills: list[tuple[int, str]] = []
        if knife is not None and knife in self._alive:
            kills.append((knife, "werewolf"))
        if self._poisoned is not None and self._poisoned in self._alive:
            kills.append((self._poisoned, "witch"))
        for target, cause in kills:
            if cause == "werewolf":
                guarded = target == self._guarded
                saved = target == self._saved
                if guarded and saved and self.variant.guard_witch_overlap_kills:
                    self._kill(target, cause)
                elif guarded or saved:
                    continue
                else:
                    self._kill(target, cause)
            else:                                        # the guard's shield stops the knife, not the poison
                self._kill(target, cause)
        died = sorted(before - self._alive)
        self._night_victims[self._round] = died
        if knife is not None and knife not in died:
            self._wasted_knives.append({"round": self._round, "target": knife})
        self._guard_last = self._guarded
        self._queue_deaths(died, "night")
        if self._pending():
            self._settle_to = "day"
            self._phase = "interrupt"
        else:
            self._open_day()

    # ================================================================ pack
    def _living_wolves(self) -> list[int]:
        return [s for s in self._order if s in self._alive and self._roles[s] == "werewolf"]

    def _chat_due(self) -> bool:
        if self.playbook_mode == "none" or not self.variant.pack_planning or not self._living_wolves():
            return False
        return (not self._pack_planned) if self.playbook_mode == "static" else True

    def _plan_due(self) -> bool:
        if self.playbook_mode == "none" or not self.variant.pack_planning:
            return False
        if self.playbook_mode != "dynamic" and self._pack_planned:
            return False
        if self._planned_round == self._round:
            return False
        wolves = self._living_wolves()
        return bool(wolves) and bool(plan_for(len(wolves)))

    def _chat_round_cap(self) -> int:
        """How many rounds of pack discussion the variant allows at most."""
        return max(1, self.variant.chat_passes)

    def _chat_speaker(self) -> int | None:
        """Whose turn it is in the night channel. Read-only: only step() moves the cursor."""
        wolves = self._living_wolves()
        if not wolves or self._chat_done:
            return None
        return wolves[self._chat_cursor % len(wolves)]

    def _apply_pack_say(self, actor: int, args: dict) -> None:
        self._pack_chat.append({"round": self._round, "player_id": actor, "content": args["content"], "wants": args.get("tactic")})
        self._advance_chat(spoke=True)

    def _apply_pack_pass(self, actor: int) -> None:
        self._advance_chat(spoke=False)

    def _advance_chat(self, spoke: bool) -> None:
        """One wolf has finished his turn in the night channel; hand the floor on.

        The discussion ends when a whole round goes by with every living wolf passing, or
        when the variant's round cap is reached -- each round is one model call per wolf,
        so the cap is what keeps a chatty pack from burning the budget.
        """
        if spoke:
            self._chat_quiet = False
        self._chat_cursor += 1
        if self._chat_cursor >= len(self._living_wolves()):
            self._chat_cursor = 0
            self._chat_round += 1
            if self._chat_quiet or self._chat_round >= self._chat_round_cap():
                self._chat_done = True
            self._chat_quiet = True
        if self._chat_done:
            self._phase = "pack_plan" if self._plan_due() else "night"

    def _planner(self) -> int | None:
        wolves = self._living_wolves()
        return wolves[0] if wolves else None

    def _apply_pack_plan(self, actor: int, args: dict) -> None:
        roles = {int(k): v for k, v in args["roles"].items()}
        changed = {w: [self._pack_roles.get(w), r] for w, r in roles.items() if self._pack_roles.get(w) != r} if self._pack_roles else {}
        self._pack_plan = args["plan"]
        self._pack_roles = roles
        self._pack_vote_target = args.get("vote_target")
        self._pack_planned = True
        self._planned_round = self._round
        self._plan_history.append({"round": self._round, "by": actor, "plan": args["plan"], "roles": dict(roles), "changed": changed})
        self._log(actor, "pack_plan", plan=args["plan"], my_tactic=roles.get(actor))
        self._phase = "night"

    # ================================================================ review (复盘)
    def _review_cap(self) -> int:
        """How many rounds of review the variant allows. 0 means the review is off."""
        return max(0, self.variant.review_rounds)

    def _review_seats(self) -> list[int]:
        """Everyone, alive and dead: a player who was knifed on night one has as much to
        say about the game as the last survivor. The list cannot change during the review
        -- nothing kills anybody once the result is frozen -- so the cursor never drifts."""
        return list(self._order)

    def _review_speaker(self) -> int | None:
        """Whose turn it is in the review. Read-only: only step() moves the cursor."""
        seats = self._review_seats()
        if not seats or self._review_done or self._review_cap() == 0:
            return None
        return seats[self._review_cursor % len(seats)]

    def _apply_review_say(self, actor: int, args: dict) -> None:
        self._review_chat.append({"round": self._review_round + 1, "player_id": actor, "content": args["content"]})
        self._log(actor, "review_say", said=args["content"][:200])
        self._advance_review(spoke=True)

    def _apply_review_pass(self, actor: int) -> None:
        self._log(actor, "review_pass")
        self._advance_review(spoke=False)

    def _advance_review(self, spoke: bool) -> None:
        """One seat has finished its review turn; hand the floor on.

        The review ends when a whole round goes by with every seat passing, or when the
        variant's round cap is reached -- each round is one model call per SEAT (12p = 12
        calls a round), so the cap is a token budget as much as a rule.
        """
        if spoke:
            self._review_quiet = False
        self._review_cursor += 1
        if self._review_cursor >= len(self._review_seats()):
            self._review_cursor = 0
            self._review_round += 1
            if self._review_quiet or self._review_round >= self._review_cap():
                self._review_done = True
            self._review_quiet = True
        if self._review_done:
            self._phase = "over"

    # ================================================================ day
    def _open_day(self) -> None:
        if self._check_end():
            return
        if self.variant.sheriff and self._round == 1 and self._sheriff is None and not self._sheriff_settled:
            self._phase = "campaign"
            self._sheriff_cursor = 0
        else:
            self._phase = "day_speech"
        self._speech_cursor = 0

    def _close_day(self) -> None:
        if self._check_end():
            return
        self._round += 1
        if self._round > MAX_ROUNDS:
            self._finish("werewolf")
            return
        self._speech_cursor = 0
        self._begin_night()

    def _speech_order(self) -> list[int]:
        order = [s for s in self._order if s in self._alive]
        if self._sheriff in order:
            i = order.index(self._sheriff)
            order = order[i:] + order[:i]
        return order

    # -- election
    def _campaign_speaker(self) -> int | None:
        order = [s for s in self._order if s in self._alive]
        return order[self._sheriff_cursor] if self._sheriff_cursor < len(order) else None

    def _advance_campaign(self) -> None:
        self._sheriff_cursor += 1
        if self._campaign_speaker() is None:
            if len(self._candidates) == 1:
                self._sheriff = self._candidates[0]
            if len(self._candidates) <= 1:
                self._sheriff_settled = True
                self._phase = "day_speech"
                self._speech_cursor = 0
            else:
                self._phase = "campaign_vote"

    def _electorate(self) -> list[int]:
        return [s for s in sorted(self._alive) if s not in self._candidates]

    def _resolve_election(self) -> None:
        counts = Counter(t for t in self._sheriff_votes.values() if t is not None)
        if counts:
            top = max(counts.values())
            leaders = [t for t, n in counts.items() if n == top]
            if len(leaders) == 1:
                self._sheriff = leaders[0]
        self._sheriff_settled = True
        self._phase = "day_speech"
        self._speech_cursor = 0

    # -- vote
    def _resolve_vote(self) -> None:
        ballots = self._votes[self._round]
        weighted: dict[int, float] = {}
        for voter, target in ballots.items():
            if target is not None:
                weighted[target] = weighted.get(target, 0.0) + self._vote_weights.get(voter, 1.0)
        exiled = None
        if weighted:
            top = max(weighted.values())
            leaders = [t for t, n in weighted.items() if n == top]
            if len(leaders) == 1:
                exiled = leaders[0]
            elif self.rules.tie_revote and not self._revote and self._begin_revote(sorted(leaders)):
                return
        self._revote = []
        self._exiles[self._round] = exiled
        if exiled is not None:
            self._kill(exiled, "vote")
            self._queue_deaths([exiled], "vote")
        if self._pending():
            self._settle_to = "night"
            self._phase = "interrupt"
        else:
            self._close_day()

    def _begin_revote(self, tied: list[int]) -> bool:
        """平票 PK (rules.tie_revote, engine v2): the day's vote runs once more, between the
        tied players only, and they do not vote in it. The first ballot is kept as a public
        record. False when nobody is left to vote (every living player is tied): the
        tie then stands and exiles nobody, as before v2."""
        voters = [s for s in sorted(self._alive) if s not in tied]
        if not voters:
            return False
        self._tie_ballots[self._round] = dict(self._votes[self._round])
        self._votes[self._round] = {}
        self._revote = list(tied)
        self._day_voters = voters
        return True

    def _apply_detonate(self, actor: int, reason: str) -> None:
        self._revote = []
        self._kill(actor, "detonate")
        self._detonations.append({"round": self._round, "player_id": actor, "reason": reason})
        self._log(actor, "self_detonate", reason=reason[:120])
        self._exiles[self._round] = None
        if self._sheriff == actor:
            self._sheriff = None
        if self._phase in ("campaign", "campaign_vote"):
            self._sheriff_settled = True
        self._pending_hunter = None            # a detonator forfeits every consequence of his death
        self._pending_badge = None
        self._close_day()

    # ================================================================ deaths & interrupts
    def _kill(self, seat: int, cause: str) -> None:
        if seat not in self._alive:
            return
        self._alive = self._alive - {seat}
        self._deaths.append(Death(self._round, seat, cause))
        if cause == "hunter":                                    # a shot is public; its victim owes nothing, but the badge moves on
            self._queue_deaths([seat], "hunter")

    def _queue_deaths(self, died: list[int], cause: str) -> None:
        for pid in died:
            if self._roles[pid] == "hunter" and self._pending_hunter is None and self._death_cause(pid) in self.variant.hunter_shoots_on:
                self._pending_hunter = pid
            if pid == self._sheriff and self._pending_badge is None:
                self._pending_badge = pid
            if self.variant.last_words and pid not in self._last_words_given and (cause == "vote" or (cause == "night" and self._round == 1)):
                if pid not in self._pending_last_words:
                    self._pending_last_words.append(pid)

    def _death_cause(self, pid: int) -> str:
        return next(d.cause for d in reversed(self._deaths) if d.player_id == pid)

    def _pending(self) -> bool:
        return bool(self._pending_hunter or self._pending_badge or self._pending_last_words)

    def _settle_if_clear(self) -> None:
        if self._pending():
            return
        to, self._settle_to = self._settle_to, None
        if to == "day":
            self._open_day()
        elif to == "night":
            self._close_day()
        else:
            raise RuntimeError("interrupt settled with nowhere to go")

    # ================================================================ next
    def _finish(self, winner: str) -> None:
        """Fix the result, then hand the table to the review.

        The outcome dict is built HERE, at the moment the win condition fires, and is
        never rebuilt: `_next` returns a copy of it and nothing else ever writes it. That
        is what makes a review turn structurally unable to change who won -- the review
        runs after this line, and there is no code path from a review action back to it.
        """
        if self._outcome is not None:
            return
        self._winner = winner
        self._outcome = {"winner": winner, "round": self._round, "roles": dict(self._roles)}
        if self._review_cap() > 0:
            self._phase = "review"
            self._review_cursor = 0
            self._review_round = 0
            self._review_quiet = True
            self._review_done = False
        else:
            self._phase = "over"

    def _check_end(self) -> bool:
        if self._outcome is not None:                 # already decided; never re-decided
            return True
        wolves = [s for s in self._alive if self._roles[s] == "werewolf"]
        others = [s for s in self._alive if self._roles[s] != "werewolf"]
        if not wolves:
            self._finish("village")
        elif len(wolves) >= len(others):
            self._finish("werewolf")
        return self._outcome is not None

    def _next(self) -> Turn | Terminal:
        if self._phase == "over":
            return Terminal(dict(self._outcome))       # the dict `_finish` froze, never rebuilt
        if self._phase == "review":
            s = self._review_speaker()
            if s is not None:
                return Turn(s, "review")
            self._phase = "over"
            return self._next()
        if self._phase == "interrupt":
            if self._pending_hunter is not None:
                return Turn(self._pending_hunter, "hunter_shoot")
            if self._pending_badge is not None:
                return Turn(self._pending_badge, "badge")
            if self._pending_last_words:
                return Turn(self._pending_last_words[0], "last_words")
            raise RuntimeError("interrupt phase with nothing pending")
        if self._phase == "pack_chat":
            s = self._chat_speaker()
            if s is not None:
                return Turn(s, "pack_chat")
            self._phase = "pack_plan" if self._plan_due() else "night"
            return self._next()
        if self._phase == "pack_plan":
            p = self._planner()
            if p is not None and self._plan_due():
                return Turn(p, "pack_plan")
            self._phase = "night"
            return self._next()
        if self._phase == "night":
            a = self._night_actor()
            if a is not None:
                return Turn(a, "night")
            self._end_night()
            return self._next()
        if self._phase == "campaign":
            s = self._campaign_speaker()
            if s is not None:
                return Turn(s, "campaign")
            self._advance_campaign()
            return self._next()
        if self._phase == "campaign_vote":
            for s in self._electorate():
                if s not in self._sheriff_votes:
                    return Turn(s, "campaign_vote")
            self._resolve_election()
            return self._next()
        if self._phase == "day_speech":
            order = self._speech_order()
            if self._speech_cursor < len(order):
                return Turn(order[self._speech_cursor], "speak")
            self._phase = "day_vote"
            self._day_voters = sorted(self._alive)
            self._vote_weights = {self._sheriff: SHERIFF_VOTE_WEIGHT} if self._sheriff in self._alive else {}
            return self._next()
        if self._phase == "day_vote":
            for s in self._day_voters:
                if s in self._alive and s not in self._votes.get(self._round, {}):
                    return Turn(s, "vote")
            self._resolve_vote()
            return self._next()
        raise RuntimeError(f"unknown phase {self._phase}")

    # ================================================================ rules text
    def rules_text(self) -> str:
        v = self.variant
        counts = v.role_counts
        n = v.num_players
        setup = ", ".join(f"{c} {(_PLURAL.get(r, r + 's') if c != 1 else r)}" for r in ROLE_ORDER if (c := counts.get(r, 0)))
        lines = [
            f"You are playing a game of Werewolf ({n} players, ids 1-{n}).",
            "",
            f"Setup: {setup}.",
            "Each night the wolves kill someone, the seer checks one player's alignment, and the witch may save the victim or poison somebody.",
        ]
        if counts.get("guard"):
            lines.append("The guard protects one player each night and may not protect the same seat twice running.")
            lines.append(self.RULES["guard_self"] if v.guard_self else "The guard may not protect himself.")
            lines.append("The guard's protection stops the wolves' knife; it does not stop the witch's poison.")
            if counts.get("witch"):
                lines.append(self.RULES["guard_witch_overlap_kills"] if v.guard_witch_overlap_kills else self.RULES["guard_witch_overlap"])
        if counts.get("witch"):
            lines.append("The witch may use her antidote on herself." if v.witch_self_save else self.RULES["witch_self"])
        lines += [
            "Each day every living player speaks once, in a fixed order, and then everyone votes at the same time: nobody sees a ballot until all are cast. The player with the most votes is exiled; "
            + (self.RULES["tie_revote"] if self.rules.tie_revote else "a tie exiles nobody."),
            "",
            "Night actions are decided by the players who hold them: the seer chooses who to check, each wolf names a kill and the pack's majority stands, and the witch chooses whether to spend her antidote or her poison.",
        ]
        if v.sheriff:
            lines += ["", "On the first day there is an election for the badge. Standing costs you your vote in that election. The sheriff speaks first each day and casts 1.5 votes, and on death passes the badge on or destroys it."]
        if v.last_words:
            lines.append("A player who is exiled, or who dies on the first night, gets last words before leaving.")
        if counts.get("hunter"):
            lines.append(self.RULES["hunter_any_death"])
        if counts.get("werewolf") and v.pack_planning and self.playbook_mode != "none":
            lines.append(f"Before they act, the wolves talk among themselves in a private night channel: on his turn each living wolf either speaks or passes, "
                         f"and a wolf who has passed speaks again in a later round if a packmate says something worth answering. "
                         f"The discussion ends as soon as one whole round goes by with every living wolf passing, and in any case after {self._chat_round_cap()} rounds.")
        if counts.get("werewolf"):
            lines.append("A werewolf may reveal himself during the day (self-detonate): the day ends at once with no vote and no exile, he leaves the game without last words, and a badge he holds is destroyed.")
        if v.review_rounds > 0:
            lines.append(f"When the game ends the result is already final and nothing said afterwards can move it. Every player then -- everyone, "
                         f"alive and dead -- is shown the whole truth of the game: every role, every night action, every check, and what the wolves "
                         f"said to each other in private. The review goes in rounds; in each round every player at once either speaks or passes, and sees what the others said "
                         f"only when the round is over. This review (复盘) ends as soon as one "
                         f"whole round goes by with every player passing, and in any case after {v.review_rounds} rounds.")
        lines += [
            "",
            f"The village wins when all {counts.get('werewolf', 0)} wolves are dead. The wolves win when they equal or outnumber the players left on the other side, or when the game reaches round {MAX_ROUNDS} unresolved.",
            "",
            "You win or lose with your team. Wolves may lie freely; villagers must work out the truth from what people say and how they vote.",
        ]
        return "\n".join(lines)

    RULES: dict[str, str] = {
        "guard_self": "The guard may protect himself.",
        "guard_witch_overlap": "A player the guard protects and the witch saves on the same night survives; the two do not cancel each other out.",
        "guard_witch_overlap_kills": "A player the guard protects and the witch saves on the same night dies; the two cancel each other out.",
        "witch_self": "The witch may not use her antidote on herself.",
        "tie_revote": "on a tie the vote is held once more between the tied players only (they do not vote in it), and a second tie exiles nobody.",
        "hunter_any_death": "The hunter may shoot whenever he dies -- exiled, killed by the wolves, or poisoned by the witch. Nothing silences the shot.",
    }

    # ================================================================ helpers
    def _team(self, role: str) -> list[int]:
        return [s for s in self._order if self._roles[s] == role]

    def _log(self, actor: int, kind: str, **detail) -> None:
        self._action_log[actor].append({"round": self._round, "phase": "day", "kind": kind, **detail})

    @staticmethod
    def _public_cause(cause: str) -> str:
        return {"vote": "exiled", "hunter": "shot", "detonate": "detonated"}.get(cause, "night")

    # ---- introspection for tests and tooling (read-only, not part of the protocol)
    def ground_truth(self) -> dict:
        return {"roles": dict(self._roles), "winner": self._winner}

    def night_records(self) -> list[dict]:
        return list(self._night_records)

    def pack_chat_log(self) -> list[dict]:
        return list(self._pack_chat)

    def review_log(self) -> list[dict]:
        return [dict(e) for e in getattr(self, '_review_chat', [])]

    def public_state(self) -> dict:
        """The public record of the world -- what the whole table knows happened -- for
        comparing two replays step by step (scripts/compat_check.py).

        Deliberately not a view: ballots are listed as cast even while sealed (sealing
        is a view rule, the ballots are world facts), and the rules text is left out --
        both differ between engine versions without the world differing. Keys a newer
        rule adds appear only once that rule has acted, so two versions agree until a
        rule actually fires.
        """
        out: dict[str, Any] = {
            "round": self._round, "phase": self._phase, "alive": sorted(self._alive),
            "dead": [[d.round, d.player_id, self._public_cause(d.cause)] for d in self._deaths],
            "speeches": [[s.round, s.player_id, s.kind, s.content] for s in self._speeches],
            "votes": {r: dict(b) for r, b in self._votes.items()},
            "exiles": dict(self._exiles), "sheriff": self._sheriff, "candidates": list(self._candidates),
            "badge": list(self._badge_transfers), "shots": list(self._shots),
            "detonations": [[d["round"], d["player_id"]] for d in self._detonations],
            "review": [dict(e) for e in self._review_chat], "winner": self._winner,
        }
        if self._tie_ballots or self._revote:
            out["tie_votes"] = {r: dict(b) for r, b in self._tie_ballots.items()}
            out["revote"] = list(self._revote)
        return out

    def world_state(self) -> dict:
        """Every private field of the world, normalised to JSON-able values: what
        ``step`` reads and writes. Two replays that agree here are the same world."""
        def norm(x):
            if isinstance(x, (set, frozenset)):
                return sorted(x)
            if isinstance(x, dict):
                return {str(k): norm(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [norm(v) for v in x]
            if hasattr(x, "__dataclass_fields__"):
                return norm(dict(x.__dict__))
            return x
        return {k: norm(v) for k, v in sorted(vars(self).items()) if k.startswith("_") and k != "_playbook"}

    def outcome(self) -> dict | None:
        """The frozen result, or None while the game is still live."""
        return dict(self._outcome) if self._outcome else None
