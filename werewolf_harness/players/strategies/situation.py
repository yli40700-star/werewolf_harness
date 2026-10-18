"""Reading the board: what is true right now, stated as facts.

This module answers "what is the situation" and stops there. It does not
answer "which tactic is therefore allowed", and an earlier version that did
was a mistake worth recording: every tactic carried a set of machine
preconditions, and the tool filtered the ones that failed out of the agent's
view. Two things were wrong with it.

It moved the decision. A wolf that never sees 深水 has not declined it, and a
shortlist of one is an instruction wearing a shortlist's clothes. The
judgement about when a tactic fits is the interesting part of playing this
game, and handing the agent a pre-filtered menu is taking that part away and
then measuring what is left.

It also destroyed the measurement it was supposed to enable. "The agent chose
a tactic that did not fit the board" is the single most informative event
here, and filtering made it unobservable by construction: an unfitting tactic
could not be chosen because it could not be seen.

So the facts are computed and reported, the playbook keeps its own account of
when each tactic applies in the prose a human guide uses, and the agent puts
the two together. How well it does that is then something the logs can be
asked about afterwards, rather than something the tool has already decided.

No model call may appear in this file: the facts have to come from the engine,
or an agent's misreading of the board would be confirmed by the reading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .playbook import Strategy, load_playbook

# --- what a wolf can actually see about the board ------------------------
#
# Everything here is derivable from public transcript plus the pack's own
# private knowledge. Nothing reads a role the seat is not entitled to know:
# `Situation` is built for a specific wolf, and building it for a villager
# would produce a strictly smaller set of facts.


@dataclass(frozen=True)
class Situation:
    seat: int
    round: int
    phase: str                      # "night" | "campaign" | "speak" | "vote"
    alive: tuple[int, ...]
    pack: tuple[int, ...]           # living packmates, including `seat`
    pack_roles: dict[int, str]      # current assignment, may be empty
    sheriff: int | None
    speech_index: int               # 0-based position in today's speaking order
    speakers_today: int
    # Who has publicly claimed which god role, by seat. Keyword-derived, so it
    # is an approximation -- deliberately the same approximation the retrieval
    # index makes, rather than a second and better one that would flatter these
    # numbers relative to what the agent itself can do.
    claims: dict[int, tuple[str, ...]] = field(default_factory=dict)
    # Seats named as suspects in today's speeches, and by how many speakers.
    pressure: dict[int, int] = field(default_factory=dict)
    # And in yesterday's. The pack meets at night, before anyone has spoken
    # today, so `pressure` is empty at exactly the moment the assignment is
    # made -- which made 抗推 structurally unassignable: its precondition asked
    # about a push that had not happened yet. Yesterday's names are the best
    # evidence available at the time the decision is actually taken.
    pressure_last: dict[int, int] = field(default_factory=dict)
    wolves_lost: int = 0
    # Deaths, and how many of the dead had *claimed* a god role before dying.
    # Not how many gods actually died: roles are never revealed at this table,
    # so counting real ones would hand the pack a fact no seat can observe. An
    # earlier version did exactly that, and it slipped past `assert_no_leak`
    # because that guard inspects the view and this arrives through a tool.
    deaths_so_far: int = 0
    claimed_gods_dead: int = 0
    saves_seen: int = 0             # nights where the pack killed and nobody died
    # The table's opening composition, which is announced before the game and
    # is therefore something every seat can count from. 屠边 is the tactic that
    # consists of doing that arithmetic, and it was unplayable without this.
    setup_line: str = ""

    def packmates(self) -> tuple[int, ...]:
        return tuple(p for p in self.pack if p != self.seat)


# --- claim and pressure detection ----------------------------------------
#
# Keyword matching, not a classifier. A claim is a short, highly stereotyped
# utterance in this game ("我是预言家", "我跳女巫") and the false-positive it
# is prone to -- quoting someone else's claim -- is handled by requiring the
# first person marker rather than the role word alone.

# A claim is made by *doing the role*, not by naming it. Nobody at a real
# table says "我是预言家" and stops -- they report a check and give a badge
# flow, and that is the claim. Matching only the literal phrase missed 30% of
# them, which mattered twice over: adherence scored four wolves as having
# abandoned 悍跳 while they were in the middle of executing it, and
# `read_board` told the pack "还没有人跳身份" while a seer had been reporting
# results for two rounds. The tool was feeding the agents a false board.
#
# `_ME` is the filler allowed between 我 and the verb: time words and
# emphasis only. Without it, "我不相信他验了5号" reads as a claim.
# `_NEAR` is what may sit between 我 and the verb: anything short that is not
# a third party. The exclusion is the whole point — "我不相信他验了5号" and
# "7号说他验了5号" are not claims, and an unbounded gap swallows both.
_NEAR = r"(?:[^。！？；，,;他她它你您]{0,10})"

_CLAIM = {
    # A claim is made by *doing the role*. Four shapes, and the first three
    # were each missed by an earlier version of this pattern:
    #   我是3号预言家            -- a seat number between 我是 and the role
    #   昨晚我按警徽流验了10号   -- a modifier between 我 and the verb
    #   我昨天公布的警徽流是5-11 -- 警徽流 with anything in front of it
    #   我是预言家               -- the literal form, which is the rare one
    "seer": re.compile(
        r"我(?:是|跳|就是|本身是|作为)(?:\s*\d{1,2}\s*号)?(?:真)?预言家"
        r"|以(?:真)?预言家(?:的)?身份"
        # Announcing a badge flow, not being named in someone else's. "警徽流
        # 里有我10号" and "3号把我放进警徽流" are the opposite of a claim, and
        # both matched until this needed the flow to actually be stated.
        rf"|我{_NEAR}警徽流\s*(?:是|：|:)"
        r"|(?:公布|报出?|给出)(?:了|的)?警徽流"
        rf"|我{_NEAR}(?:验|查验|查)了?\s*\d{{1,2}}\s*号"
        rf"|我{_NEAR}验人(?:结果|信息)"
        # NOT "X号金水" on its own. Reporting someone else's result is the
        # single most common sentence a villager writes -- "7号验出5号金水",
        # "12号跳预言家报了5号查杀" -- and matching it turned every reader of
        # the board into a claimant.
    ),
    "witch": re.compile(
        r"我(?:是|跳|就是|本身是|作为)(?:\s*\d{1,2}\s*号)?(?:真)?女巫|以女巫(?:的)?身份"
        rf"|我{_NEAR}(?:用了|开了|留着)?(?:解药|毒药)"
        rf"|我{_NEAR}(?:救|毒)了?\s*\d{{1,2}}\s*号"
    ),
    "hunter": re.compile(
        r"我(?:是|跳|就是|本身是|作为)(?:\s*\d{1,2}\s*号)?(?:真)?猎人|以猎人(?:的)?身份"
        r"|我的枪口|票我.{0,6}挨枪|我(?:会|要)开枪(?:带走)?"
    ),
    "guard": re.compile(
        r"我(?:是|跳|就是|本身是|作为)(?:\s*\d{1,2}\s*号)?(?:真)?守卫|以守卫(?:的)?身份"
        rf"|我{_NEAR}守了?\s*\d{{1,2}}\s*号"
    ),
}

# Naming someone as a suspect. Three shapes, because players use all three and
# an index that only knows the first was silently blind to the other two:
#   "我怀疑 3 号" / "今天出 7 号"      -- verb before the seat
#   "7 号是狼" / "7 号很可疑"          -- verdict after the seat
#   "7 号昨天改票没解释"               -- a criticism a few characters later
# The third needs a window rather than adjacency, and the window is short on
# purpose: widen it and every seat mentioned in a long speech becomes accused.
_CRITICISM = ("是狼|有问题|很可疑|可疑|悍跳|心虚|划水|站不住|说不通|自相矛盾"
              "|没解释|解释不了|前后不一|漂|摇摆|不敢站边|发言空")
_SUSPECT = re.compile(
    r"(?:怀疑|悍跳|狼|出|票|投|推|归票到|带走)\s*(\d{1,2})\s*号"
    rf"|(\d{{1,2}})\s*号(?:[^。！？，,;；]{{0,12}})(?:{_CRITICISM})"
)


def _claims_from(speeches) -> dict[int, tuple[str, ...]]:
    out: dict[int, set[str]] = {}
    for s in speeches:
        for role, pat in _CLAIM.items():
            if pat.search(s.content):
                out.setdefault(s.player_id, set()).add(role)
    return {k: tuple(sorted(v)) for k, v in out.items()}


_RECOUNT = re.compile(r"上一?轮|上一?天|昨天|昨夜|之前|当时|第[一二三四五六]轮|回顾")


def _pressure_from(speeches, alive) -> dict[int, int]:
    """How many *distinct* speakers named each seat as a suspect today.

    Distinct speakers, not mentions: a wolf that repeats itself five times has
    not put anyone under pressure, and counting mentions would let a single
    talkative seat manufacture a `under_vote_pressure` reading for a target it
    is alone in suspecting.
    """
    by_seat: dict[int, set[int]] = {}
    for s in speeches:
        for m in _SUSPECT.finditer(s.content):
            target = int(m.group(1) or m.group(2))
            if target not in alive or target == s.player_id:
                continue
            # "第二轮我明确支持投 9 号" is a player recounting the vote record,
            # not naming a suspect today. Reading it as fresh pressure told the
            # pack that a seat was under fire when the table had moved on.
            lead = s.content[max(0, m.start() - 10):m.start()]
            if _RECOUNT.search(lead):
                continue
            by_seat.setdefault(target, set()).add(s.player_id)
    return {k: len(v) for k, v in by_seat.items()}


_ROLE_CN = {"werewolf": "狼", "seer": "预言家", "witch": "女巫",
            "hunter": "猎人", "guard": "守卫", "villager": "平民"}


def read_situation(state, seat: int, phase: str | None = None) -> Situation:
    """Build the board reading for one wolf. Pure: touches state, changes none."""
    alive = tuple(state.alive_sorted())
    pack = tuple(p for p in state.living_wolves() if p in alive)
    today = [s for s in state.speeches if s.round == state.round]
    order = [p for p in state.setup.speech_order if p in alive]

    wolves_lost = len(state.setup.wolves()) - len(pack)
    dead = [p for p in range(1, state.num_players + 1) if p not in alive]
    # Anyone who claimed a god role at any point, alive or not. Claims are
    # public; roles are not.
    all_claims = _claims_from(state.speeches)
    claimed_gods_dead = len([p for p in dead if all_claims.get(p)])

    saves = 0
    for r, victims in state.night_victims.items():
        if r < state.round and not victims:
            saves += 1

    return Situation(
        seat=seat,
        round=state.round,
        phase=phase or state.phase.value.replace("day_", "").replace("sheriff_", ""),
        alive=alive,
        pack=pack,
        pack_roles=dict(state.pack_roles),
        sheriff=state.sheriff,
        speech_index=len([s for s in today if s.kind == "speech"]),
        speakers_today=len(order),
        claims=_claims_from(today),
        pressure=_pressure_from(today, set(alive)),
        pressure_last=_pressure_from(
            [sp for sp in state.speeches if sp.round == state.round - 1], set(alive)),
        wolves_lost=wolves_lost,
        deaths_so_far=len(dead),
        claimed_gods_dead=claimed_gods_dead,
        saves_seen=saves,
        setup_line="，".join(
            f"{n} {_ROLE_CN.get(r, r)}"
            for r, n in sorted(state.variant.role_counts.items(), key=lambda x: -x[1])),
    )


# --- saying it out loud --------------------------------------------------


def describe(s: Situation) -> list[str]:
    """The board as a list of plain statements.

    Facts only, and deliberately the same facts a human player would have from
    watching the table. Nothing here recommends anything.
    """
    lines = [
        f"第 {s.round} 天，{'夜里' if s.phase == 'night' else s.phase}。"
        f"场上 {len(s.alive)} 人：{list(s.alive)}",
        f"你是 {s.seat} 号。狼队还剩 {len(s.pack)} 只：{list(s.pack)}",
    ]
    lines.append(f"警长：{s.sheriff if s.sheriff else '还没有产生'}")
    if s.pack_roles:
        lines.append("狼队当前分工：" + "，".join(
            f"{p} 号 {t}" for p, t in sorted(s.pack_roles.items())))
    else:
        lines.append("狼队还没有分工")

    if s.claims:
        lines.append("已经跳出来的身份：" + "，".join(
            f"{p} 号自称 {'/'.join(cs)}" for p, cs in sorted(s.claims.items())))
    else:
        lines.append("还没有人跳身份")

    if s.pressure:
        lines.append("今天被点名的（几个人点的）：" + "，".join(
            f"{p} 号 ×{n}" for p, n in sorted(s.pressure.items(), key=lambda x: -x[1])))
    else:
        lines.append("今天还没有人被点名")
    if s.pressure_last:
        lines.append("昨天被点名的：" + "，".join(
            f"{p} 号 ×{n}" for p, n in sorted(s.pressure_last.items(),
                                              key=lambda x: -x[1])))

    lines.append(
        f"狼队已折损 {s.wolves_lost} 只。场上共死 {s.deaths_so_far} 人，"
        f"其中生前跳过身份的 {s.claimed_gods_dead} 人"
        f"（这局死亡不翻牌，谁是神只能推）")
    lines.append(f"开局配置：{s.setup_line}")
    lines.append("解药：还没见过有人被救" if s.saves_seen == 0
                 else f"解药：已经见过 {s.saves_seen} 次夜里无人死亡")
    return lines
