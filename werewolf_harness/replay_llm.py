"""Re-run a recorded game through the real LLMPlayer stack, with the model replaced by its recording.

    play --record      the cassette is written next to the game log (providers/recorded.py)
    replay_game(...)   same env, same seed, same LLMPlayer settings, same parallelism; every
                       seat's client is a ReplayClient over the cassette. Nothing calls a model.
    compare(...)       the replayed journal, outcome and per-turn records against the log.

`cli replay` checks that the *engine* is a function of (seed, journal). This checks the
same of the *agent stack*: given the same model replies, the loop, the guard, the
context builder and the parallel driver make the same decisions -- so a bug seen in
a real game reproduces on a laptop, and a harness change can be tested against
games that already cost money.
"""
from __future__ import annotations

import gzip
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from .arena import Terminal, Turn, run
from .envs import VARIANTS, Werewolf, engine_version_of
from .players import GuardStack, LLMPlayer, TokenBudget, TransportBudget
from .players.llm.cascade import CascadePolicy
from .players.providers.recorded import Cassette, ReplayClient, ReplayDivergence, cassette_path_for
from .players.recovery import RecoveryPolicy

DEFAULTS = {"guard": "L1,L2,L3", "evidence": False, "playbook": "static", "temperature": 0.7, "lang": "zh",
            "max_tokens": 6_000_000, "max_usd": None, "parallel": 1}


def settings_of(log: dict) -> dict:
    """Everything that shapes a seat's prompts. Logs from before `settings` existed carry
    the guard only as a label ("L1+L2+L3+E"); the rest were the defaults of the day."""
    s = dict(DEFAULTS)
    if "settings" in log:
        s.update(log["settings"])
        return s
    parts = [p for p in str(log.get("guard", "")).split("+") if p and p != "none"]
    s["evidence"] = "E" in parts
    s["guard"] = ",".join(p for p in parts if p != "E")
    s["playbook"] = log.get("playbook", "static")
    s["parallel"] = log.get("parallel", 1)
    return s


WOLF_HINTS = {
    # docs/experiments: in 8p the pack has no planning phase and the wolves never claimed seer
    "hantiao": ("本局狼队的打法是悍跳：狼队里号码最小的那只狼，白天第一次发言时声称自己是预言家，并报出一个查验结果"
                "（给好人发查杀，或者给队友发金水），与真预言家对跳、争夺好人的信任；另一只狼不要跳，帮他拉票。"),
}


def build_players(s: dict, n: int, client_for, *, policy_for=None, strong_for=None) -> tuple[dict, TokenBudget, GuardStack]:
    """The seats of `cli play`, built from its settings. One builder for play and replay, so
    a replay cannot drift from the run it reproduces by a default someone changed in one place.

    With ``s["cascade"]`` (a CascadePolicy dict, players/llm/cascade.py) every seat gets a
    second client from ``strong_for(seat)``: ``client_for`` is then the cheap tier."""
    guard = GuardStack(tuple(x for x in s["guard"].split(",") if x), evidence_forced=s["evidence"])
    budget = TransportBudget()
    tokens = TokenBudget(max_tokens=s["max_tokens"], max_usd=s["max_usd"])
    cascade = CascadePolicy.from_dict(s.get("cascade"))
    if cascade is not None and strong_for is None:
        raise ValueError("these settings play a cascade: build_players needs strong_for")

    def tiers(seat: int) -> dict:
        return {"strong_client": strong_for(seat), "cascade": cascade} if cascade is not None else {}
    players = {seat: LLMPlayer(client_for(seat), guard=guard, transport_budget=budget, token_budget=tokens,
                               playbook=s["playbook"] != "none", temperature=s["temperature"], lang=s["lang"],
                               async_note=s["parallel"] > 1, memory=s.get("memory", True), invoke_calls=s.get("invoke_calls", False),
                               **({"policy": policy_for(seat)} if policy_for else {}), **tiers(seat))
               for seat in range(1, n + 1)}
    if s.get("wolf_hint"):
        for p in players.values():
            p.context.wolf_hint = WOLF_HINTS[s["wolf_hint"]]
    return players, tokens, guard


def no_wait_policy() -> RecoveryPolicy:
    """The recorded game's retry policy without its sleeps: a replayed 503 is retried exactly
    as the original was, but the backoff it waited out is not waited again."""
    p = RecoveryPolicy()
    p.sleep = lambda _s: None
    return p


class Truncated:
    """An environment that stops after `upto` committed steps. Lets a regression case keep
    only the part of a long game that matters (and only the calls that part made)."""

    def __init__(self, env, upto: int):
        self._env, self._upto, self._n = env, upto, 0

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, seed):
        self._n = 0
        return self._cut(self._env.reset(seed))

    def step(self, actor, task, action):
        self._n += 1
        return self._cut(self._env.step(actor, task, action))

    def open_turns(self):
        return self._env.open_turns() if self._n < self._upto else []

    def _cut(self, t):
        if isinstance(t, Turn) and self._n >= self._upto:
            return Terminal({"truncated": self._upto})
        return t


@dataclass
class ReplayReport:
    journal: list[dict] = field(default_factory=list)
    outcome: Any = None
    turns: list[dict] = field(default_factory=list)
    calls: int = 0                        # requests the replay answered from the cassette
    recorded: int = 0                     # calls in the cassette
    used: set = field(default_factory=set)
    drifts: list = field(default_factory=list)
    error: str | None = None              # ReplayDivergence (or a crash) that stopped the game
    ms: float = 0.0
    misses: int = 0


def load(path: str, cassette: str | None = None) -> tuple[dict, Cassette]:
    with (gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")) as f:
        log = json.load(f)
    cpath = cassette or log.get("cassette")
    if cpath and not os.path.isabs(cpath):
        cpath = os.path.join(os.path.dirname(os.path.abspath(path)), cpath)
    if not cpath or not os.path.exists(cpath):
        guess = cassette_path_for(path)
        if not os.path.exists(guess):
            raise FileNotFoundError(f"{path} 没有录像（用 play --record 录一局；找过 {guess}）")
        cpath = guess
    return log, Cassette.load(cpath)


def replay_game(log: dict, cas: Cassette, *, allow_drift: bool = False, upto: int | None = None,
                on_player=None) -> ReplayReport:
    """Re-run the recorded game on the recorded replies. `on_player(player)` may adjust each
    seat before play (tests stream the replay through drafts with it)."""
    s = settings_of(log)
    n = VARIANTS[log["variant"]].num_players
    rep = ReplayReport(recorded=len(cas.calls))
    clients: dict = {}
    counters: dict = {}            # keyed (seat, turn): a seat's two cascade tiers share one numbering

    def client_for(seat: int, tier: str | None = None) -> ReplayClient:
        clients[(seat, tier)] = ReplayClient(cas, seat, allow_drift=allow_drift, used=rep.used, drifts=rep.drifts,
                                             counters=counters, tier=tier)
        return clients[(seat, tier)]

    players, _tokens, _guard = build_players(s, n, client_for, policy_for=lambda _seat: no_wait_policy(),
                                             strong_for=lambda seat: client_for(seat, "strong"))
    for p in players.values():
        if on_player is not None:
            on_player(p)
    # the rules the game was recorded under (a log from before versioning = baseline): the
    # rules text is part of every prompt, so a different version would drift every call
    env = Werewolf(log["variant"], playbook_mode=s["playbook"], engine_version=engine_version_of(log))
    env_run = Truncated(env, upto) if upto is not None else env
    t0 = time.perf_counter()
    try:
        ep = run(env_run, players, log["seed"], parallel=s["parallel"])
        rep.journal = _plain([r.__dict__ for r in ep.journal])
        rep.outcome = _plain(ep.outcome)
        rep.misses = ep.misses
    except ReplayDivergence as exc:
        rep.error = str(exc)
    except Exception as exc:                                   # noqa: BLE001 -- reported, not hidden
        rep.error = f"{type(exc).__name__}: {exc}"
    rep.ms = (time.perf_counter() - t0) * 1000
    rep.turns = _plain([t.to_dict() for p in players.values() for t in p.history])
    rep.calls = sum(c.calls for c in clients.values())
    return rep


def compare(log: dict, rep: ReplayReport, *, upto: int | None = None) -> list[str]:
    """Where the replay differs from the log, in reading order; [] means identical."""
    out: list[str] = []
    if rep.error:
        out.append(rep.error)
        return out
    want = _plain(log["journal"])[: upto] if upto is not None else _plain(log["journal"])
    for i, (a, b) in enumerate(zip(want, rep.journal)):
        if a != b:
            out.append(f"journal 第 {i} 步不同：录制 {a} / 回放 {b}")
            break
    if len(want) != len(rep.journal):
        out.append(f"journal 长度不同：录制 {len(want)} / 回放 {len(rep.journal)}")
    if upto is None and _plain(log.get("outcome")) != rep.outcome:
        out.append(f"结局不同：录制 {log.get('outcome')} / 回放 {rep.outcome}")
    if upto is None and "turns" in log:
        got, exp = rep.turns, _plain(log["turns"])
        if len(got) != len(exp):
            out.append(f"回合记录条数不同：录制 {len(exp)} / 回放 {len(got)}")
        for a, b in zip(exp, got):
            a, b = _decided(a), _decided(b)
            if a != b:
                keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
                out.append(f"座位 {a.get('player_id')} 第 {a.get('round')} 轮 {a.get('task')} 的回合记录不同：{keys}")
                break
    return out


# What a turn record says about who was watching rather than what the seat decided:
# draft_attempt exists only when the speech was streamed to someone (docs/adr/0009), so
# a game recorded while streaming and replayed without it (or the reverse) is the same game.
_WATCHER_FIELDS = ("draft_attempt",)


def _decided(t: dict) -> dict:
    return {k: v for k, v in t.items() if k not in _WATCHER_FIELDS} if any(k in t for k in _WATCHER_FIELDS) else t


def check_expectations(rep: ReplayReport, expects: list[str]) -> list[str]:
    """Statements a regression case makes about its replay, beyond "same journal".

        not_forced:SEQ        journal step SEQ (0-based, as in `timings`) was the model's own choice
        forced:SEQ            ... was filled in by the harness
        action:SEQ:NAME       journal step SEQ is a NAME action (speak, vote, night_kill, ...)
        winner:SIDE           the game ended with SIDE winning
        max_forced:K          at most K forced steps in the whole journal
        no_fallback           no turn needed a fallback (transport failure or default action)
    """
    out = []
    for e in expects:
        kind, _, rest = e.partition(":")
        try:
            if kind in ("not_forced", "forced", "action"):
                seq, _, name = rest.partition(":")
                rec = rep.journal[int(seq)]
                if kind == "not_forced" and rec["forced"]:
                    out.append(f"{e}：第 {seq} 步（{rec['actor']} 号 {rec['task']}）被兜底了")
                elif kind == "forced" and not rec["forced"]:
                    out.append(f"{e}：第 {seq} 步没有被兜底")
                elif kind == "action" and (rec["action"] or {}).get("name") != name:
                    out.append(f"{e}：第 {seq} 步是 {(rec['action'] or {}).get('name')}")
            elif kind == "winner":
                if (rep.outcome or {}).get("winner") != rest:
                    out.append(f"{e}：结局是 {rep.outcome}")
            elif kind == "max_forced":
                n = sum(bool(r["forced"]) for r in rep.journal)
                if n > int(rest):
                    out.append(f"{e}：{n} 步被兜底")
            elif kind == "no_fallback":
                bad = [(t["player_id"], t["round"], t["task"]) for t in rep.turns if (t.get("recovery") or {}).get("fallback_used")]
                if bad:
                    out.append(f"{e}：这些回合用了兜底 {bad[:5]}")
            else:
                out.append(f"{e}：不认识的期望（见 replay_llm.check_expectations）")
        except (IndexError, ValueError) as exc:
            out.append(f"{e}：{exc}")
    return out


def _plain(v):
    """Through JSON and back, as the log was: int dict keys become strings, tuples lists."""
    return json.loads(json.dumps(v, ensure_ascii=False, default=str))
