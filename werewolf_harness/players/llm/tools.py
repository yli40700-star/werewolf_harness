"""Information tools for the LLM seat.

They read the seat's *view* (what the environment already decided it may know)
and the seat's own belief memory. They never touch the environment: a tool that
could would be a second information exit, and there is exactly one.

Terminal actions are not here -- they come from the environment as ToolSpecs
in the ActionSpec, computed per turn.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

from ...arena import Param, ToolSpec
from ..strategies.playbook import get_strategy, load_playbook, plan_for, search_strategies
from ..strategies.situation import Situation, describe, _claims_from, _pressure_from
from .belief import BeliefState

SPEECH_MAX_CHARS = 1200


def speech_id(round_no: int, pid: int) -> str:
    return f"r{round_no}p{pid}"


@dataclass
class ToolResult:
    observation: str
    data: Any = None
    untrusted: bool = False
    sources: list[str] = field(default_factory=list)


@dataclass
class ToolContext:
    view: dict
    belief: BeliefState
    task: str

    @property
    def me(self) -> int:
        return self.view["you"]

    @property
    def role(self) -> str:
        return self.view["role"]

    @property
    def pub(self) -> dict:
        return self.view["public"]


@dataclass
class InfoTool:
    spec: ToolSpec
    fn: Callable[[ToolContext, dict], ToolResult]
    roles: tuple[str, ...] = ()
    untrusted_output: bool = False

    @property
    def name(self) -> str:
        return self.spec.name


class SemanticError(Exception):
    """Gate 3: a well-formed call the tool refuses on its meaning."""


class InfoRegistry:
    def __init__(self, tools: list[InfoTool]):
        self._tools = {t.name: t for t in tools}

    def all(self) -> list[InfoTool]:
        return list(self._tools.values())

    def get(self, name: str) -> InfoTool | None:
        return self._tools.get(name)

    def visible(self, role: str) -> list[InfoTool]:
        return [t for t in self._tools.values() if not t.roles or role in t.roles]

    def execute(self, name: str, args: dict, ctx: ToolContext) -> ToolResult:
        tool = self._tools[name]
        return tool.fn(ctx, tool.spec.normalise(args))


# ------------------------------------------------------------------ tools

def _pid(n: int) -> Param:
    return Param("int", f"player 1-{n}")


def _query_history(ctx: ToolContext, a: dict) -> ToolResult:
    n = ctx.view["num_players"]
    pid, rnd = a["player_id"], a["round"]
    if not (1 <= pid <= n):
        raise SemanticError(f"query_history.player_id must be a player between 1 and {n}")
    if not (1 <= rnd <= ctx.view["round"]):
        raise SemanticError(f"query_history.round must be between 1 and {ctx.view['round']} (the current round)")
    found = [s for s in ctx.pub["speeches"] if s["player_id"] == pid and s["round"] == rnd]
    if not found:
        return ToolResult(f"player {pid} did not speak in round {rnd}", data=[])
    return ToolResult("\n".join(s["content"] for s in found), data=found, untrusted=True,
                      sources=[speech_id(rnd, pid)])


def _query_votes(ctx: ToolContext, a: dict) -> ToolResult:
    rnd = a["round"]
    if not (1 <= rnd <= ctx.view["round"]):
        raise SemanticError(f"query_votes.round must be between 1 and {ctx.view['round']}")
    ballots = ctx.pub["votes"].get(rnd) or ctx.pub["votes"].get(str(rnd))
    if not ballots:
        return ToolResult(f"no votes recorded in round {rnd}")
    body = ", ".join(f"{v}->{'abstain' if t is None else t}" for v, t in sorted(ballots.items(), key=lambda kv: int(kv[0])))
    ex = ctx.pub["exiles"].get(rnd, ctx.pub["exiles"].get(str(rnd)))
    tail = f"exiled: {ex}" if ex is not None else "nobody was exiled"
    return ToolResult(f"round {rnd} votes: {body}; {tail}", data=ballots)


def _query_deaths(ctx: ToolContext, a: dict) -> ToolResult:
    label = {"exiled": "exiled by vote", "night": "killed at night", "shot": "shot", "detonated": "self-detonated"}
    dead = ctx.pub["dead"]
    if not dead:
        return ToolResult("nobody has died yet", data=[])
    return ToolResult("; ".join(f"round {d['round']}: player {d['player_id']} died ({label.get(d['cause'], d['cause'])})" for d in dead), data=dead)


def _read_belief(ctx: ToolContext, a: dict) -> ToolResult:
    e = ctx.belief.get(a["player_id"])
    if e is None:
        return ToolResult(f"no belief recorded for player {a['player_id']}")
    return ToolResult(f"player {e.player_id}: suspicion={e.suspicion:.2f}, reason={e.reason!r}, evidence={e.evidence_refs}", data=e.to_dict())


def _update_belief(ctx: ToolContext, a: dict) -> ToolResult:
    n = ctx.view["num_players"]
    pid = a["player_id"]
    if pid == ctx.me:
        raise SemanticError("update_belief cannot target yourself")
    if not (1 <= pid <= n):
        raise SemanticError(f"update_belief.player_id must be a player between 1 and {n}")
    e = ctx.belief.update(pid, a["suspicion"], a["reason"], ctx.view["round"], a.get("evidence_refs") or [])
    return ToolResult(f"belief updated: player {pid} suspicion={e.suspicion:.2f}", data=e.to_dict())


def _check_ability(ctx: ToolContext, a: dict) -> ToolResult:
    priv, role = ctx.view["private"], ctx.role
    if role == "seer":
        checks = priv.get("checks", [])
        body = ", ".join(f"r{c['round']} player {c['target']} = {'WOLF' if c['is_wolf'] else 'not a wolf'}" for c in checks) or "none yet"
        text = f"you are the seer. checks so far: {body}"
    elif role == "witch":
        text = (f"you are the witch. antidote available: {priv.get('antidote_available')}, poison available: {priv.get('poison_available')}. "
                "You spend them yourself, on your night turn.")
    elif role == "werewolf":
        text = f"you are a werewolf. fellow wolves: {priv.get('fellow_wolves', [])}. Each of you names a kill at night; the majority decides."
    elif role == "guard":
        text = "you are the guard. you protect one player each night; never the same seat twice running."
    elif role == "hunter":
        text = "you are the hunter. when you die, however you die, you may shoot one living player."
    else:
        text = "you are a villager. you have no night ability."
    return ToolResult(text, data={"role": role, **priv})


# -- wolf playbook (pulled context; may hold a view, see PROMPTS.md)

def _list_plans(ctx: ToolContext, a: dict) -> ToolResult:
    pack = 1 + len(ctx.view["private"].get("fellow_wolves", []))
    plans = plan_for(pack)
    if not plans:
        return ToolResult("no plan fits a pack this small", data=[])
    lines = [f"- {p.id}：{p.name}（需要 {p.needs} 狼）分工 {list(p.roles)} — {p.when}" for p in plans]
    return ToolResult("\n".join(lines), data=[p.id for p in plans])


def _list_strategies(ctx: ToolContext, a: dict) -> ToolResult:
    strategies, _ = load_playbook()
    lines = []
    for s in strategies.values():
        lines.append(f"- {s.id}｜{s.name}（{s.role}）{s.summary}")
        if s.when:
            lines.append("    适用时机：" + "；".join(s.when[:2]))
        if s.needs_teammates:
            lines.append("    需要队友打：" + "、".join(s.needs_teammates))
        if s.abort_signal:
            lines.append("    放弃信号：" + s.abort_signal)
    return ToolResult("\n".join(lines), untrusted=True)


def _query_strategy(ctx: ToolContext, a: dict) -> ToolResult:
    hits = search_strategies(a["situation"], limit=2)
    if not hits:
        return ToolResult("没有匹配的战术", untrusted=True)
    return ToolResult("\n\n".join(s.full() for s, _ in hits), untrusted=True, sources=[s.id for s, _ in hits])


def _my_assignment(ctx: ToolContext, a: dict) -> ToolResult:
    priv = ctx.view["private"]
    tactic = priv.get("my_tactic")
    if not tactic:
        return ToolResult("你还没有被分配战术")
    s = get_strategy(tactic)
    return ToolResult(f"本局狼队打法：{priv.get('pack_plan')}\n你的分工：\n\n{s.full() if s else tactic}")


_ROLE_CN = {"werewolf": "狼", "seer": "预言家", "witch": "女巫", "hunter": "猎人", "guard": "守卫", "villager": "平民"}


def _speeches_ns(speeches: list[dict]) -> list[SimpleNamespace]:
    return [SimpleNamespace(**s) for s in speeches]


def situation_from_view(view: dict, phase: str) -> Situation:
    pub, priv = view["public"], view["private"]
    alive = tuple(pub["alive"])
    pack = tuple(sorted([view["you"]] + list(priv.get("fellow_wolves", []))))
    speeches = _speeches_ns(pub["speeches"])
    rnd = view["round"]
    today = [s for s in speeches if s.round == rnd]
    n = view["num_players"]
    dead = [p for p in range(1, n + 1) if p not in alive]
    all_claims = _claims_from(speeches)
    counts = view.get("role_counts") or _role_counts_from_rules(view["rules"])
    wolves_total = counts.get("werewolf", len(pack))
    saves = sum(1 for d in range(1, rnd) if not any(x["round"] == d and x["cause"] == "night" for x in pub["dead"]))
    return Situation(
        seat=view["you"], round=rnd, phase=phase, alive=alive, pack=pack,
        pack_roles={int(k): v for k, v in (priv.get("pack_roles") or {}).items()},
        sheriff=pub.get("sheriff"),
        speech_index=len([s for s in today if s.kind == "speech"]),
        speakers_today=len(pub["speech_order"]),
        claims=_claims_from(today),
        pressure=_pressure_from(today, set(alive)),
        pressure_last=_pressure_from([s for s in speeches if s.round == rnd - 1], set(alive)),
        wolves_lost=wolves_total - len(pack),
        deaths_so_far=len(dead),
        claimed_gods_dead=len([p for p in dead if all_claims.get(p)]),
        saves_seen=saves,
        setup_line="，".join(f"{c} {_ROLE_CN.get(r, r)}" for r, c in sorted(counts.items(), key=lambda x: -x[1])),
    )


_SETUP = re.compile(r"Setup: (.+?)\.")
_COUNT = re.compile(r"(\d+) (werewol(?:f|ves)|seers?|witch(?:es)?|hunters?|guards?|villagers?)")
_SINGULAR = {"werewolves": "werewolf", "witches": "witch"}


def _role_counts_from_rules(rules: str) -> dict[str, int]:
    m = _SETUP.search(rules)
    out: dict[str, int] = {}
    if not m:
        return out
    for n, word in _COUNT.findall(m.group(1)):
        role = _SINGULAR.get(word, word.rstrip("s"))
        out[role] = int(n)
    return out


def _read_board(ctx: ToolContext, a: dict) -> ToolResult:
    phase = "night" if ctx.task in ("pack_chat", "pack_plan") else {"night": "night", "campaign": "campaign", "speak": "speak", "vote": "vote"}.get(ctx.task, "speak")
    sit = situation_from_view(ctx.view, phase)
    lines = describe(sit)
    t = ctx.view["private"].get("pack_vote_target")
    if ctx.task not in ("pack_chat", "pack_plan") and t:
        lines.append(f"狼队昨晚约定今天投 {t} 号——目标还活着。这是约定不是命令，但四狼同票是全场最大的票块。" if t in sit.alive
                     else f"狼队昨晚约定投 {t} 号，但他已经出局了，约定作废。")
    if ctx.task in ("pack_chat", "pack_plan"):
        lines += ["", "提醒：今晚这一刀是现在的决定，分工管的是明天白天。"]
    return ToolResult("\n".join(lines), data={"round": sit.round, "alive": list(sit.alive), "pack": list(sit.pack)})


_GOLD = re.compile(r"(\d{1,2})\s*号(?:是)?金水")
_CHECKED_WOLF = re.compile(r"(\d{1,2})\s*号(?:是)?查杀")
_CLAIM_SEER = re.compile(r"我是预言家|我预言家|预言家是我")


def _kill_options(ctx: ToolContext, a: dict) -> ToolResult:
    pub, priv = ctx.pub, ctx.view["private"]
    pack = {ctx.me, *priv.get("fellow_wolves", [])}
    targets = [p for p in pub["alive"] if p not in pack]
    if not targets:
        return ToolResult("场上没有可刀的目标", data=[])
    speeches = pub["speeches"]
    claimed_seers = {s["player_id"] for s in speeches if _CLAIM_SEER.search(s["content"])}
    claims = _claims_from(_speeches_ns(speeches))
    gold: dict[int, set[int]] = {}
    kill: dict[int, set[int]] = {}
    for s in speeches:
        if s["player_id"] not in claimed_seers:
            continue
        for m in _GOLD.finditer(s["content"]):
            gold.setdefault(int(m.group(1)), set()).add(s["player_id"])
        for m in _CHECKED_WOLF.finditer(s["content"]):
            kill.setdefault(int(m.group(1)), set()).add(s["player_id"])
    pressure = _pressure_from(_speeches_ns([s for s in speeches if s["round"] == ctx.view["round"]]), set(pub["alive"]))
    rows = []
    for p in targets:
        marks = []
        if claims.get(p):
            marks.append("自称 " + "/".join(claims[p]))
        if gold.get(p):
            marks.append(f"被 {sorted(gold[p])} 发过金水")
        if kill.get(p):
            marks.append(f"被 {sorted(kill[p])} 报过查杀")
        if pressure.get(p):
            marks.append(f"今天被 {pressure[p]} 人点名")
        if pub.get("sheriff") == p:
            marks.append("警长")
        rows.append(f"  {p} 号：" + ("；".join(marks) if marks else "到目前为止没有任何公开标记"))
    wasted = priv.get("wasted_knives") or []
    if wasted:
        rows.append("落空过的刀：" + "、".join(f"第{k['round']}夜的 {k['target']} 号" for k in wasted) + "（守卫守了那里，或者女巫把解药花在了他身上）")
    return ToolResult("\n".join(rows), data={"targets": targets})


def build_registry(playbook: bool = True, num_players: int = 12) -> InfoRegistry:
    n = num_players
    tools = [
        InfoTool(ToolSpec("query_history", "Read what a player actually said in a given round. Rounds before the current one are NOT in your context: this is the only way to recall them.",
                          {"player_id": _pid(n), "round": Param("int", "round number, 1-based")}), _query_history, untrusted_output=True),
        InfoTool(ToolSpec("query_votes", "Read the full ballot of a past round (who voted for whom).",
                          {"round": Param("int", "round number")}), _query_votes),
        InfoTool(ToolSpec("query_deaths", "List every death so far with its round and cause."), _query_deaths),
        InfoTool(ToolSpec("read_belief", "Read your own recorded judgement about a player.", {"player_id": _pid(n)}), _read_belief),
        InfoTool(ToolSpec("update_belief", 'Record or revise your judgement about a player. Cite the speech ids you relied on, e.g. "r2p5".',
                          {"player_id": _pid(n),
                           "suspicion": Param("float", "0 = certainly village, 1 = certainly wolf", range=(0.0, 1.0)),
                           "reason": Param("str", "why, in one sentence", max_len=400),
                           "evidence_refs": Param("list[str]", "speech ids such as r2p5 that support this", required=False, default=[])}), _update_belief),
        InfoTool(ToolSpec("check_ability", "Check your own role, night ability status and private results."), _check_ability),
    ]
    if playbook:
        tools += [
            InfoTool(ToolSpec("list_plans", "WEREWOLF ONLY: the division-of-labour plans this pack is large enough to staff."), _list_plans, roles=("werewolf",)),
            InfoTool(ToolSpec("query_strategy", "WEREWOLF ONLY: look up wolf tactics for a situation you describe. Returns playbook entries -- text you did not write.",
                              {"situation": Param("str", "the spot you are in, in your own words", max_len=300, non_empty=True)}), _query_strategy, roles=("werewolf",), untrusted_output=True),
            InfoTool(ToolSpec("read_board", "WEREWOLF ONLY: the state of the board as plain facts -- who has claimed what, who is taking fire, how many wolves are left. It recommends nothing; the reading is yours to make."), _read_board, roles=("werewolf",)),
            InfoTool(ToolSpec("list_strategies", "WEREWOLF ONLY: every tactic in the playbook, one line each, with what it is for and the signal to abandon it. Use query_strategy to read one in full."), _list_strategies, roles=("werewolf",), untrusted_output=True),
            InfoTool(ToolSpec("kill_options", "WEREWOLF ONLY: every living non-wolf and what the table has publicly shown about it -- claims, who vouched for or accused whom, who leads votes, and which of your knives landed on nothing. Public signal only; it states, it does not advise."), _kill_options, roles=("werewolf",)),
            InfoTool(ToolSpec("my_assignment", "WEREWOLF ONLY: the tactic the pack assigned you this game, in full."), _my_assignment, roles=("werewolf",)),
        ]
    return InfoRegistry(tools)
