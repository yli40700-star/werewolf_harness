"""Real-model evaluation baseline: cost, latency, reliability, behaviour, outcome.

    python -m werewolf_harness.scripts.eval_baseline '/data/runs/base/g*.json' \
        --compare '顺序(pilot)=/data/runs/pilot/g*.json' --compare '并行(fast)=/data/runs/fast/g*.json' \
        --out docs/baseline_A.md --json docs/baseline_A.json

Input is the game logs written by ``cli play --out`` (journal, turns, timings,
usage, night, ground_truth). Everything here is judged by the ENGINE's record --
the journal, the night log and the dealt roles -- never by what a model says
about itself: a seer "honest" rate compares the checks a seat announced with the
checks the engine actually ran for it, and a say/do rate compares a speech with
the ballot the engine recorded.

Statistics. Turns inside one game are correlated (same deal, same table talk),
so a per-turn sample of 1600 is not 1600 independent draws. Two intervals:
  * Wilson score interval for every proportion (what the brief asks for; it is
    the right shape for small k but treats units as independent), and
  * a cluster bootstrap that resamples GAMES with replacement and recomputes the
    statistic on the pooled units of the drawn games -- the honest interval for
    anything pooled across turns (means, p50/p95, and proportions too).
With fewer than ``MIN_GAMES_FOR_CI`` games a bootstrap interval is not
reported: two games resampled give at most three distinct pools.

The text extractors are deliberately conservative regexes over Chinese speech
with seat numbers written "3号". Each one documents what it refuses to read
(past tense, hedges, conditionals, reports of somebody else's claim), and every
behaviour metric is printed with its coverage -- how many speeches yielded a
claim at all -- because a consistency rate over 4% of speeches is a statement
about those 4%, not about the model.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob as _glob
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

Z95 = 1.959963984540054
BOOTSTRAP_B = 2000
MIN_GAMES_FOR_CI = 5
WOLF = "werewolf"


# ============================================================ statistics

def wilson(k: int, n: int, z: float = Z95) -> tuple[float | None, float | None, float | None]:
    """Wilson score interval for k successes in n trials: (p, lo, hi).

    Chosen over the normal approximation because many rates here are near 0
    (fallbacks, leaks) where p +- z*sqrt(p(1-p)/n) collapses to [0, 0].
    """
    if n <= 0:
        return None, None, None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


def percentile(xs: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy's default 'linear' method), q in [0, 100]."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def cluster_bootstrap(groups: list[list], stat: Callable[[list], float | None], *, b: int = BOOTSTRAP_B,
                      seed: int = 0, alpha: float = 0.05) -> tuple[float | None, float | None, float | None]:
    """Percentile bootstrap that resamples whole groups (games), not units.

    `stat` gets the pooled units of the resampled games. Returns (point, lo, hi);
    lo/hi are None when there are too few groups for the interval to mean anything.
    """
    pooled = [u for g in groups for u in g]
    point = stat(pooled) if pooled else None
    if len([g for g in groups if g]) < MIN_GAMES_FOR_CI or point is None:
        return point, None, None
    rng = random.Random(seed)
    k = len(groups)
    draws = []
    for _ in range(b):
        sample = [u for _ in range(k) for u in groups[rng.randrange(k)]]
        v = stat(sample) if sample else None
        if v is not None:
            draws.append(v)
    if not draws:
        return point, None, None
    return point, percentile(draws, 100 * alpha / 2), percentile(draws, 100 * (1 - alpha / 2))


def dist(groups: list[list[float]], seed: int = 0) -> dict:
    """mean / p50 / p95 of pooled values, each with a games-bootstrap CI."""
    n = sum(len(g) for g in groups)
    out = {"n": n, "n_games": len([g for g in groups if g])}
    for name, fn in (("mean", mean), ("p50", lambda xs: percentile(xs, 50)), ("p95", lambda xs: percentile(xs, 95))):
        p, lo, hi = cluster_bootstrap(groups, fn, seed=seed)
        out[name] = {"est": p, "lo": lo, "hi": hi}
    return out


def prop(groups: list[list[int]], seed: int = 0) -> dict:
    """A proportion over 0/1 units grouped by game: Wilson and games-bootstrap CIs."""
    units = [u for g in groups for u in g]
    k, n = sum(units), len(units)
    p, lo, hi = wilson(k, n)
    _, blo, bhi = cluster_bootstrap(groups, lambda xs: sum(xs) / len(xs), seed=seed)
    return {"k": k, "n": n, "n_games": len([g for g in groups if g]), "p": p, "wilson": [lo, hi], "boot": [blo, bhi]}


# ============================================================ loading

@dataclass
class Entry:
    """One committed turn: the journal record, the seat's TurnResult, its timing row."""
    seq: int
    actor: int
    task: str
    round: int
    action: dict
    turn: dict
    timing: dict | None = None

    @property
    def target(self):
        return (self.action.get("args") or {}).get("target_id")

    @property
    def content(self) -> str:
        return str((self.action.get("args") or {}).get("content") or "")


@dataclass
class Game:
    path: str
    model: str
    variant: str
    seed: int
    parallel: int | None
    roles: dict[int, str]
    winner: str | None
    duration_s: float
    usage: dict
    entries: list[Entry]
    night: list[dict]
    crashed: object
    health: dict
    mtime: float
    aligned: bool = True
    turns: list[dict] = field(default_factory=list)

    def team(self, seat: int) -> str:
        return WOLF if self.roles.get(seat) == WOLF else "village"

    def wolves(self) -> set[int]:
        return {s for s, r in self.roles.items() if r == WOLF}


def load_game(path: str) -> Game:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    roles = {int(k): v for k, v in (d.get("ground_truth") or {}).get("roles", (d.get("outcome") or {}).get("roles", {})).items()}
    entries, aligned = align(d.get("journal") or [], d.get("turns") or [], d.get("timings") or [])
    return Game(path=path, model=d.get("model", "?"), variant=d.get("variant", "?"), seed=d.get("seed"),
                parallel=d.get("parallel"), roles=roles,
                winner=(d.get("ground_truth") or {}).get("winner") or (d.get("outcome") or {}).get("winner"),
                duration_s=float(d.get("duration_s") or 0.0), usage=d.get("usage") or {}, entries=entries,
                night=d.get("night") or [], crashed=d.get("crashed"), health=d.get("health") or {},
                mtime=os.path.getmtime(path), aligned=aligned, turns=d.get("turns") or [])


def align(journal: list[dict], turns: list[dict], timings: list[dict]) -> tuple[list[Entry], bool]:
    """Put the per-seat TurnResults back in journal (commit) order.

    The log groups `turns` by seat, the journal is in commit order, and
    `timings[i]` describes `journal[i]`. A seat's k-th turn is its k-th journal
    record (discarded speculative turns never reach either list), so the join is
    by (actor, ordinal). `aligned` is False if any task disagrees -- the caller
    then knows round numbers may be off.
    """
    by_seat: dict[int, list[dict]] = defaultdict(list)
    for t in turns:
        by_seat[t["player_id"]].append(t)
    used: Counter = Counter()
    out, ok = [], True
    for i, j in enumerate(journal):
        a = j["actor"]
        k = used[a]
        used[a] += 1
        t = by_seat[a][k] if k < len(by_seat[a]) else {}
        if t.get("task") != j["task"]:
            ok = False
        out.append(Entry(i, a, j["task"], int(t.get("round") or 0), j.get("action") or {}, t,
                         timings[i] if i < len(timings) else None))
    return out, ok


def load(patterns: Iterable[str]) -> list[Game]:
    paths = sorted({p for pat in patterns for p in _glob.glob(pat)})
    return [load_game(p) for p in paths]


# ============================================================ text helpers

_CLAUSE = re.compile(r"[，,。；;！!？?\n：:]")
_SENTENCE = re.compile(r"[。；;！!？?\n]")


def clauses(text: str) -> list[tuple[int, str]]:
    """(start offset, clause) pieces split on Chinese/ASCII punctuation."""
    out, start = [], 0
    for m in _CLAUSE.finditer(text):
        if m.start() > start:
            out.append((start, text[start:m.start()]))
        start = m.end()
    if start < len(text):
        out.append((start, text[start:]))
    return out


def clause_at(text: str, pos: int) -> tuple[int, str]:
    for start, c in clauses(text):
        if start <= pos < start + len(c):
            return start, c
    return 0, ""


def sentence_of(text: str, pos: int) -> tuple[str, str]:
    """(the part of pos's sentence before pos, the whole sentence)."""
    starts = [m.end() for m in _SENTENCE.finditer(text, 0, pos)]
    start = starts[-1] if starts else 0
    m = _SENTENCE.search(text, pos)
    return text[start:pos], text[start: m.start() if m else len(text)]


def sentence_after(text: str, pos: int) -> str:
    m = _SENTENCE.search(text, pos)
    return text[pos: m.start() if m else len(text)]


# Shared refusal lists: a match whose clause carries one of these is not read as a claim.
_PAST = re.compile(r"投了|投过|曾投|曾经|第[一二三四五六七八九\d]+[天轮]|昨天|昨日|上一?轮|当时|之前|那天|前一天|前天|首日|上次")
_COND = re.compile(r"如果|假如|要是|若是|若|假设|万一|一旦|是不是|是否|吗")
_NEG = re.compile(r"不|别|没|勿|莫|非")
_MAYBE = re.compile(r"可能|或者|或是|也许|看情况|待定|再看")


# ============================================================ extractor: say/do

_VOTE_VERB = r"(?P<verb>把票投给|票投给|投票给|投给|投向|票给|投|票出|集中票出|放逐|归票给?|出)"
_INTENT_FIRST = re.compile(
    r"(?:我|我这票|我这一票|我的票|这一?票|今天我|我今天|我这轮|我本轮)"
    r"(?P<mid>[^，,。；;！!？?\n]{0,8}?)" + _VOTE_VERB + r"\s*(?P<seat>\d)\s*号")
_INTENT_ALL = re.compile(
    r"(?:请大家|大家|我们|建议|今天)(?P<mid>[^，,。；;！!？?\n]{0,6}?)" + _VOTE_VERB + r"\s*(?P<seat>\d)\s*号")
_ACCUSE = [
    re.compile(r"我(?P<mid>[^，,。；;！!？?\n]{0,4}?)怀疑\s*(?P<seat>\d)\s*号"),
    re.compile(r"(?P<seat>\d)\s*号(?P<mid>[^，,。；;！!？?\n]{0,8}?)(?:很|最|非常|比较|更|十分|特别|相当)可疑"),
    re.compile(r"(?P<seat>\d)\s*号(?P<mid>[^，,。；;！!？?\n]{0,2}?)(?:大概率|很可能|多半|肯定|一定|就)是狼"),
]
_VOUCH = [
    re.compile(r"我(?P<mid>[^，,。；;！!？?\n]{0,4}?)(?:相信|信任|认可|保)\s*(?P<seat>\d)\s*号(?!的)"),
    re.compile(r"(?P<seat>\d)\s*号(?P<mid>[^，,。；;！!？?\n]{0,6}?)(?:视为|视作|算作?|当作|当成|看作)好人"),
    re.compile(r"(?:不会|不要|不能|别)(?P<mid>[^，,。；;！!？?\n]{0,3}?)投\s*(?P<seat>\d)\s*号"),
]
# A clause that reports someone else's check or claim is not the speaker's own position.
_EXPLAINS = re.compile(r"[，,\s]*(?:是因为|因为|的原因|是对|对得上|是错|没错|的理由|这一票)")
_REPORT = re.compile(r"验|查|报|金水|声称|自称|起跳|跳了|说过|说了|表示|认为他")


@dataclass
class SayClaims:
    intent: int | None = None          # explicit announced ballot
    intent_ambiguous: bool = False     # two different explicit targets: unreadable
    accuse: set[int] = field(default_factory=set)
    vouch: set[int] = field(default_factory=set)

    @property
    def any(self) -> bool:
        return self.intent is not None or bool(self.accuse) or bool(self.vouch)


def say_claims(text: str, speaker: int | None = None) -> SayClaims:
    """What a speech commits its speaker to for this round's ballot.

    Three tiers, strongest first:
      intent -- "我投3号", "我这轮会把票投给2号", "请大家今天把票投给5号".
                Refused when the clause is past tense ("第一天我投了1号"),
                conditional ("如果...就投3号") or negated (that becomes a vouch).
      accuse -- "我怀疑3号", "3号很可疑", "3号大概率是狼" (not a report of a
                check: "验出3号是狼" is not the speaker's own accusation).
      vouch  -- "我相信3号", "3号可以视为好人", "我不会投3号".
    A seat that lands in both accuse and vouch is dropped from both, and a
    speech that names two different explicit targets yields no intent.
    """
    c = SayClaims()
    intents: set[int] = set()
    for rx in (_INTENT_FIRST, _INTENT_ALL):
        for m in rx.finditer(text):
            seat = int(m.group("seat"))
            start, cl = clause_at(text, m.start())
            pre = cl[: m.start("seat") - start]
            mid = m.group("mid") or ""
            sent_pre, sent = sentence_of(text, m.start())
            if _PAST.search(cl) or _COND.search(sent_pre + pre) or _REPORT.search(mid) or _MAYBE.search(sent):
                continue          # "第一天我投了1号", "如果...就投3号", "我验出3号" (出 here is not a ballot)
            if m.group("verb") == "出" and re.search(r"[验查跳找指说看给选报认提]$", mid):
                continue
            if _EXPLAINS.match(text[m.end("seat") + 1: m.end("seat") + 8]):
                continue          # "我投7号，是因为..." explains a ballot already cast; "票出1号是对的" judges one
            if _NEG.search(mid):
                c.vouch.add(seat)  # "我不会投3号"
                continue
            intents.add(seat)
    for rx in _ACCUSE:
        for m in rx.finditer(text):
            seat = int(m.group("seat"))
            start, cl = clause_at(text, m.start())
            pre = cl[: m.start() - start]
            mid = m.group("mid") or ""
            if _NEG.search(mid) or _NEG.search(pre[-4:]) or _COND.search(pre) or _REPORT.search(pre + mid) or "理由" in pre:
                continue
            c.accuse.add(seat)
    for rx in _VOUCH:
        for m in rx.finditer(text):
            seat = int(m.group("seat"))
            start, cl = clause_at(text, m.start())
            pre = cl[: m.start() - start]
            if _COND.search(pre) or (rx is not _VOUCH[2] and _NEG.search((m.group("mid") or "") + pre[-2:])):
                continue
            c.vouch.add(seat)
    if speaker is not None:
        intents.discard(speaker)
        c.accuse.discard(speaker)
        c.vouch.discard(speaker)
    both = c.accuse & c.vouch
    c.accuse -= both
    c.vouch -= both
    if len(intents) == 1:
        c.intent = next(iter(intents))
        c.vouch.discard(c.intent)
    elif len(intents) > 1:
        c.intent_ambiguous = True
    return c


def restrict(c: SayClaims, alive: set[int]) -> SayClaims:
    """Drop claims about seats that can no longer be voted for ("上轮被放逐的2号大概率是狼")."""
    if not alive:
        return c
    return SayClaims(intent=c.intent if c.intent in alive else None, intent_ambiguous=c.intent_ambiguous,
                     accuse=c.accuse & alive, vouch=c.vouch & alive)


def say_do_verdict(claims: SayClaims, ballot: int | None) -> dict:
    """Judge one (speech, ballot) pair, tier by tier. None = tier not applicable."""
    v = {"intent": None, "accuse": None, "vouch": None, "any": None}
    if claims.intent is not None:
        v["intent"] = ballot == claims.intent
    if claims.accuse:
        v["accuse"] = ballot in claims.accuse
    if claims.vouch:
        v["vouch"] = ballot not in claims.vouch
    # combined: the strongest applicable tier decides; a vote for a vouched seat is always a break
    if v["intent"] is not None:
        v["any"] = v["intent"] and (v["vouch"] is not False)
    elif v["accuse"] is not None:
        v["any"] = v["accuse"] and (v["vouch"] is not False)
    elif v["vouch"] is not None:
        v["any"] = v["vouch"]
    return v


# ============================================================ extractor: role claims

ROLE_WORDS = {"预言家": "seer", "女巫": "witch", "村民": "villager", "平民": "villager", "民": "villager",
              "狼人": WOLF, "狼": WOLF, "好人": "good", "猎人": "hunter", "守卫": "guard"}
_SELF_ROLE = re.compile(
    r"(?<![不如若假要])我(?:就|才|确实|真的)?是\s*(?:(?P<seat>\d)\s*号)?\s*[，,、\s]*(?:身份是|的身份是|是)?\s*"
    r"(?:真的?|普通的?|一[名个张])?\s*(?P<role>预言家|女巫|村民|平民|狼人|好人|猎人|守卫)")
_SELF_JUMP = re.compile(r"我(?:要|来)?(?:起跳|跳)(?:的是)?\s*(?:真)?预言家")


def self_role_claims(text: str) -> list[str]:
    """Roles the speaker claims for themself ("我是3号，村民", "我是真预言家", "我跳预言家").

    "我不是预言家", "如果我是预言家" and "要是我是女巫" are not read.
    """
    out = []
    for m in _SELF_ROLE.finditer(text):
        start, cl = clause_at(text, m.start())
        if _COND.search(cl[: m.start() - start]):
            continue
        out.append(ROLE_WORDS[m.group("role")])
    for m in _SELF_JUMP.finditer(text):
        start, cl = clause_at(text, m.start())
        if not _COND.search(cl[: m.start() - start]) and "不" not in cl[: m.start() - start][-2:]:
            out.append("seer")
    return out


_CHECK = re.compile(r"(?:查验|验)(?:了|的是|出|过)?\s*(?P<seat>\d)\s*号")
_GOOD = re.compile(r"好人|金水|不是狼|非狼|良民|好身份|没问题")
_BAD = re.compile(r"狼人|是狼|为狼|查杀|狼")


def check_claims(text: str) -> list[tuple[int, str]]:
    """First-person seer reports: [(seat, 'good'|'wolf')].

    Only checks with no other subject in front of them inside the clause:
    "首夜查验了2号，结果是好人" and "我验5号是狼" are read; "1号验了2号是好人"
    and "他验过2号为好人" are reports of someone else's claim and are not.
    The verdict is the first good/wolf word after the seat in the same sentence;
    "不是狼" is tried before "狼" so it reads as good.
    """
    out = []
    for m in _CHECK.finditer(text):
        start, cl = clause_at(text, m.start())
        pre = cl[: m.start() - start]
        if re.search(r"\d\s*号|他|她|对方|其|别人|有人|谁|如果|假如", pre):
            continue
        rest = sentence_after(text, m.end())[:24]
        g, b = _GOOD.search(rest), _BAD.search(rest)
        if g and (not b or g.start() <= b.start()):
            out.append((int(m.group("seat")), "good"))
        elif b:
            out.append((int(m.group("seat")), "wolf"))
    return out


# assertions about another seat's role: "3号是狼", "5号就是女巫", "2号肯定是好人"
_ASSERT = re.compile(
    r"(?P<seat>\d)\s*号(?:的身份)?\s*(?P<mod>就|肯定|一定|确定|必然|绝对|确实|百分百|百分之百|铁)?\s*(?:是|为)\s*"
    r"(?:真|一只|一张|一个|一名|铁)?\s*(?P<role>预言家|女巫|狼人|狼|村民|平民|好人)")
_HEDGE = re.compile(r"可能|怀疑|倾向|像|或许|也许|大概|猜|认为|觉得|看作|视为|当作|当成|算|相信|采信|暂时|暂且|应该|估计|未必|"
                    r"不一定|是否|吗|还是|至少|证明|不足|不像|如何|怎么|不排除|默认|待定|需要|要看|先|姑且|按|那|则|就是说")
_REPORTED = re.compile(r"验|查|报|说|称|跳|给出|指认|确认了|结果|金水|宣布|公布|表态|承认|认|觉得")


@dataclass
class Assertion:
    seat: int
    role: str          # seer | witch | villager | werewolf | good
    span: str


def role_assertions(text: str, speaker: int | None = None) -> list[Assertion]:
    """Definite, unreported statements of another seat's role.

    Refused: any hedge or conditional before the seat in the clause ("我倾向
    5号是好人", "如果3号是预言家"), any reporting verb ("2号验出5号是狼",
    "3号自称是女巫"), a question, and the speaker talking about themself.
    What remains is the speaker stating, as fact, what a seat is.
    """
    out = []
    for m in _ASSERT.finditer(text):
        seat = int(m.group("seat"))
        if seat == speaker:
            continue
        start, cl = clause_at(text, m.start())
        pre = cl[: m.start() - start]
        tail = text[m.end(): m.end() + 2]
        if _HEDGE.search(pre) or _COND.search(pre) or _REPORTED.search(pre) or "不" in pre[-2:] or "吗" in tail:
            continue
        out.append(Assertion(seat, ROLE_WORDS[m.group("role")], m.group(0)))
    return out


def role_matches(claimed: str, actual: str) -> bool:
    if claimed == "good":
        return actual != WOLF
    return claimed == actual


# ============================================================ guard regex audit

def guard_declared_vote(text: str) -> tuple[int | None, str | None]:
    """The L3 guard's own reading of a speech (its first matching pattern), with the trigger text.

    Imported rather than copied so the audit always measures the guard as shipped.
    """
    from ..players.guard.verify import _VOTE_INTENT
    for rx in _VOTE_INTENT:
        m = rx.search(text or "")
        if m:
            return int(m.group(1)), text[max(0, m.start() - 2): m.end()]
    return None, None


# ============================================================ metrics: cost

def _calls_in_turn(t: dict) -> int:
    """Model calls a TurnResult made: every ReAct step except a harness default, plus the note."""
    steps = [s for s in t.get("react_trace") or [] if not str(s.get("action", "")).endswith("(default)")]
    step_lat = sum(int(s.get("latency_ms") or 0) for s in steps)
    note = 1 if (t.get("note_ms") or (int(t.get("latency_ms") or 0) - step_lat) > 0) else 0
    return len(steps) + note


def cost_metrics(games: list[Game]) -> dict:
    per_game = []
    task_tok: dict[str, list[list[float]]] = defaultdict(lambda: [[] for _ in games])
    task_calls: dict[str, list[list[float]]] = defaultdict(lambda: [[] for _ in games])
    for gi, g in enumerate(games):
        u = g.usage
        turn_prompt = sum(int(t.get("prompt_tokens") or 0) for t in g.turns)
        turn_comp = sum(int(t.get("completion_tokens") or 0) for t in g.turns)
        calls_est = sum(_calls_in_turn(t) for t in g.turns)
        per_game.append({
            "local": int(u.get("total_tokens") or 0), "local_prompt": int(u.get("prompt_tokens") or 0),
            "local_completion": int(u.get("completion_tokens") or 0),
            "cli": int(u.get("gateway_prompt_tokens") or 0) + int(u.get("gateway_completion_tokens") or 0),
            "cli_prompt": int(u.get("gateway_prompt_tokens") or 0), "cli_completion": int(u.get("gateway_completion_tokens") or 0),
            "turn_sum": turn_prompt + turn_comp, "calls": int(u.get("calls") or calls_est), "calls_est": calls_est,
            "turns": len(g.entries)})
        for e in g.entries:
            task_tok[e.task][gi].append(int(e.turn.get("prompt_tokens") or 0) + int(e.turn.get("completion_tokens") or 0))
            task_calls[e.task][gi].append(_calls_in_turn(e.turn))

    def gdist(key):
        return dist([[r[key]] for r in per_game])

    per_call_cli = dist([[r["cli"] / r["calls"]] for r in per_game if r["calls"]])
    per_call_local = dist([[r["local"] / r["calls"]] for r in per_game if r["calls"]])
    drift = dist([[r["cli"] / r["local"]] for r in per_game if r["local"]])
    by_task = {}
    for task in sorted(task_tok):
        by_task[task] = {"tokens_per_turn": dist(task_tok[task]), "calls_per_turn": dist(task_calls[task]),
                         "tokens_per_game": dist([[sum(x)] for x in task_tok[task]]),
                         "share_of_tokens": sum(map(sum, task_tok[task])) / max(1, sum(r["turn_sum"] for r in per_game))}
    return {"tokens_per_game_local": gdist("local"), "tokens_per_game_cli": gdist("cli"),
            "prompt_per_game_cli": gdist("cli_prompt"), "completion_per_game_cli": gdist("cli_completion"),
            "calls_per_game": gdist("calls"), "turns_per_game": gdist("turns"),
            "tokens_per_call_cli": per_call_cli, "tokens_per_call_local": per_call_local, "cli_over_local": drift,
            "calls_accounted": sum(r["calls_est"] for r in per_game) == sum(r["calls"] for r in per_game),
            "turn_sum_equals_cli": all(r["turn_sum"] == r["cli"] for r in per_game),
            "turn_sum_equal_games": sum(1 for r in per_game if r["turn_sum"] == r["cli"]),
            "turn_sum_over_cli_min": min((r["turn_sum"] / r["cli"] for r in per_game if r["cli"]), default=None),
            "by_task": by_task, "per_game": per_game}


# ============================================================ metrics: latency

def phases(g: Game) -> list[tuple[str, list[Entry]]]:
    """Consecutive journal runs of one task: a night, a day's speeches, a ballot, the review."""
    out: list[tuple[str, list[Entry]]] = []
    for e in g.entries:
        if out and out[-1][0] == e.task:
            out[-1][1].append(e)
        else:
            out.append((e.task, [e]))
    return out


def turn_wall_s(e: Entry) -> float:
    """Critical-path seconds of one turn: committed - asked; without timings (sequential
    runs) the seat's whole model time, note included, since nothing overlapped it."""
    if e.timing:
        return float(e.timing["committed"]) - float(e.timing["asked"])
    return int(e.turn.get("latency_ms") or 0) / 1000.0


def phase_wall_s(g: Game) -> list[tuple[str, float]]:
    """Each phase's share of the game clock: from the previous phase's last commit to this
    phase's last commit (timings), or the sum of its turns' model time (sequential logs)."""
    out, prev = [], 0.0
    for task, es in phases(g):
        if all(e.timing for e in es):
            end = max(float(e.timing["committed"]) for e in es)
            out.append((task, end - prev))
            prev = end
        else:
            out.append((task, sum(turn_wall_s(e) for e in es)))
    return out


def latency_metrics(games: list[Game]) -> dict:
    step_ms = [[float(s["latency_ms"]) for e in g.entries for s in e.turn.get("react_trace") or []
                if not str(s.get("action", "")).endswith("(default)")] for g in games]
    note_ms = [[float(e.turn["note_ms"]) for e in g.entries if e.turn.get("note_ms")] for g in games]
    turn_model_s = [[int(e.turn.get("latency_ms") or 0) / 1000 for e in g.entries] for g in games]
    turn_wall = [[turn_wall_s(e) for e in g.entries] for g in games]
    by_task = {}
    for task in sorted({e.task for g in games for e in g.entries}):
        by_task[task] = {
            "turn_wall_s": dist([[turn_wall_s(e) for e in g.entries if e.task == task] for g in games]),
            "turn_model_s": dist([[int(e.turn.get("latency_ms") or 0) / 1000 for e in g.entries if e.task == task] for g in games]),
            "phase_wall_s": dist([[w for t, w in phase_wall_s(g) if t == task] for g in games]),
            "phase_wall_per_game_s": dist([[sum(w for t, w in phase_wall_s(g) if t == task)] for g in games]),
        }
    has_timings = all(e.timing for g in games for e in g.entries)
    ahead = [[1 if (e.timing or {}).get("ahead") else 0 for e in g.entries] for g in games] if has_timings else None
    return {"call_ms": dist(step_ms), "note_ms": dist(note_ms) if any(note_ms) else None,
            "turn_model_s": dist(turn_model_s), "turn_wall_s": dist(turn_wall),
            "game_s": dist([[g.duration_s] for g in games]), "by_task": by_task, "has_timings": has_timings,
            "asked_ahead": prop(ahead) if ahead else None}


# ============================================================ metrics: reliability

_BLOCK_KINDS = ("schema", "semantics", "say_do_consistency", "directive_compliance", "evidence", "forensics", "fallback")


def reliability_metrics(games: list[Game]) -> dict:
    steps_ok, malformed, blocked_by = [], [], defaultdict(list)
    turn_retry, forced, fallback, loop_broken, timeout, transport = [], [], [], [], [], []
    vote_turn_saydo = []
    for g in games:
        ok, mal, retry, frc, fb, lb, to, tr, sd = [], [], [], [], [], [], [], [], []
        per_kind = defaultdict(list)
        for e in g.entries:
            trace = [s for s in e.turn.get("react_trace") or [] if not str(s.get("action", "")).endswith("(default)")]
            for s in trace:
                ok.append(0 if s.get("guard_blocked") else 1)
                mal.append(1 if s.get("action") == "<malformed>" else 0)
                for k in _BLOCK_KINDS:
                    per_kind[k].append(1 if s.get("block_reason") == k else 0)
            r = e.turn.get("recovery") or {}
            retry.append(1 if any(s.get("guard_blocked") for s in trace) else 0)
            frc.append(1 if r.get("forced_terminal") else 0)
            fb.append(1 if r.get("fallback_used") else 0)
            lb.append(1 if r.get("loop_broken") else 0)
            to.append(1 if r.get("timeout") else 0)
            tr.append(1 if r.get("transport_retries") else 0)
            if e.task == "vote":
                sd.append(1 if any(s.get("block_reason") == "say_do_consistency" for s in trace) else 0)
        steps_ok.append(ok)
        malformed.append(mal)
        turn_retry.append(retry)
        forced.append(frc)
        fallback.append(fb)
        loop_broken.append(lb)
        timeout.append(to)
        transport.append(tr)
        vote_turn_saydo.append(sd)
        for k in _BLOCK_KINDS:
            blocked_by[k].append(per_kind[k])
    waste = defaultdict(lambda: {"steps": 0, "tokens": 0, "ms": 0})
    all_tok = all_ms = 0
    for g in games:
        for e in g.entries:
            for st in e.turn.get("react_trace") or []:
                all_tok += int(st.get("tokens") or 0)
                all_ms += int(st.get("latency_ms") or 0)
                if st.get("guard_blocked") and st.get("block_reason") != "fallback":
                    w = waste[st.get("block_reason")]
                    w["steps"] += 1
                    w["tokens"] += int(st.get("tokens") or 0)
                    w["ms"] += int(st.get("latency_ms") or 0)
    forced_by = Counter()
    for g in games:
        for e in g.entries:
            if (e.turn.get("recovery") or {}).get("forced_terminal"):
                reasons = {st.get("block_reason") for st in e.turn.get("react_trace") or [] if st.get("guard_blocked")}
                forced_by["say_do_consistency" if "say_do_consistency" in reasons else
                          (sorted(map(str, reasons))[0] if reasons else "no_block")] += 1
    for w in waste.values():
        w["token_share"] = w["tokens"] / max(1, all_tok)
        w["ms_share"] = w["ms"] / max(1, all_ms)
    return {"step_success": prop(steps_ok), "malformed_step": prop(malformed), "blocked_step_cost": dict(waste), "forced_terminal_by_block": dict(forced_by),
            "block_reason_per_step": {k: prop(v) for k, v in blocked_by.items() if sum(map(sum, v))},
            "turn_rejected_and_retried": prop(turn_retry), "turn_forced_terminal": prop(forced),
            "turn_fallback": prop(fallback), "turn_loop_broken": prop(loop_broken), "turn_timeout": prop(timeout),
            "turn_transport_retry": prop(transport), "vote_turn_saydo_blocked": prop(vote_turn_saydo),
            "games_crashed": sum(1 for g in games if g.crashed), "games_unusable": sum(1 for g in games if g.health.get("usable") is False)}


# ============================================================ metrics: behaviour

def _speeches(g: Game) -> list[Entry]:
    return [e for e in g.entries if e.task == "speak" and e.content]


def say_do_metrics(games: list[Game]) -> dict:
    tiers = ("intent", "accuse", "vouch", "any")
    covered = {t: [] for t in tiers}
    consistent = {t: [] for t in tiers}
    first_consistent = []                    # the 'any' tier against the first ballot the model proposed
    guard_rows = []
    pairs_total = 0
    examples = []
    for g in games:
        cov = {t: [] for t in tiers}
        con = {t: [] for t in tiers}
        fc = []
        votes = {(e.round, e.actor): e for e in g.entries if e.task == "vote"}
        alive_at_vote: dict[int, set[int]] = defaultdict(set)   # who still has a ballot = who can be voted for
        for (rnd, seat) in votes:
            alive_at_vote[rnd].add(seat)
        for sp in _speeches(g):
            ve = votes.get((sp.round, sp.actor))
            if ve is None:
                continue                      # spoke, then died before the ballot
            pairs_total += 1
            claims = restrict(say_claims(sp.content, sp.actor), alive_at_vote.get(sp.round, set()))
            v = say_do_verdict(claims, ve.target)
            first = next((s.get("args", {}).get("target_id") for s in ve.turn.get("react_trace") or [] if s.get("action") == "vote"), ve.target)
            v_first = say_do_verdict(claims, first)
            for t in tiers:
                cov[t].append(1 if v[t] is not None else 0)
                if v[t] is not None:
                    con[t].append(1 if v[t] else 0)
            if v_first["any"] is not None:
                fc.append(1 if v_first["any"] else 0)
            if v["any"] is False and len(examples) < 12:
                examples.append({"game": os.path.basename(g.path), "round": sp.round, "seat": sp.actor,
                                 "intent": claims.intent, "accuse": sorted(claims.accuse), "vouch": sorted(claims.vouch),
                                 "ballot": ve.target, "speech": sp.content[:160]})
            gt, trig = guard_declared_vote(sp.content)
            if gt is not None:
                blocked = any(s.get("block_reason") == "say_do_consistency" for s in ve.turn.get("react_trace") or [])
                guard_rows.append({"trigger": trig, "guard_target": gt, "ours": claims.intent, "blocked_vote": blocked,
                                   "game": os.path.basename(g.path), "round": sp.round, "seat": sp.actor})
        for t in tiers:
            covered[t].append(cov[t])
            consistent[t].append(con[t])
        first_consistent.append(fc)
    # guard audit: is the L3 trigger a vote intent at all?
    non_vote = re.compile(r"[验查跳找选指报认说给]出|出局|出现|出来|出于|提出|给出|指出|看出|说出|找出")
    misread = [r for r in guard_rows if non_vote.search(r["trigger"] or "")]
    not_now = [r for r in guard_rows if r not in misread and r["ours"] != r["guard_target"]]
    blocked_rows = [r for r in guard_rows if r["blocked_vote"]]
    return {
        "pairs": pairs_total,
        "coverage": {t: prop(covered[t]) for t in tiers},
        "consistent": {t: prop(consistent[t]) for t in tiers},
        "consistent_first_ballot": prop(first_consistent),
        "guard_audit": {"speeches_guard_reads_a_vote": len(guard_rows),
                        "trigger_not_a_vote_intent": len(misread),
                        "vote_word_but_not_this_rounds_intent": len(not_now),
                        "agree_with_ours": sum(1 for r in guard_rows if r["ours"] == r["guard_target"]),
                        "saydo_blocked_votes": len(blocked_rows),
                        "saydo_blocked_votes_from_non_vote_trigger": sum(1 for r in blocked_rows if r in misread),
                        "saydo_blocked_votes_agreeing_with_ours": sum(1 for r in blocked_rows if r["ours"] == r["guard_target"]),
                        "triggers": Counter(r["trigger"][-4:] for r in guard_rows).most_common(12),
                        "examples": misread[:6]},
        "inconsistent_examples": examples,
    }


def seer_metrics(games: list[Game]) -> dict:
    """Seer claims (who says "I am the seer") and the checks they announce, judged by the night log."""
    real_claims, fake_claims = [], []         # per game lists of 0/1: claim matches ground truth
    real_verdicts = Counter()
    fake_rows = Counter()
    seer_claim_by_team = Counter()
    wolf_fake_seer_games, wolf_seats_faking = 0, 0
    seer_ever_claimed = 0
    self_claim_true = {"village": [], WOLF: []}
    examples = []
    for g in games:
        checks = {}                           # target -> (round, 'wolf'|'good') for the true seer
        seer = next((s for s, r in g.roles.items() if r == "seer"), None)
        for n in g.night:
            if n.get("action") == "night_check" and n.get("player_id") == seer:
                checks.setdefault(n["target"], "wolf" if n.get("outcome") == WOLF else "good")
        claimants: set[int] = set()
        rc, fc = [], []
        st = {"village": [], WOLF: []}
        seen_pairs: set[tuple] = set()
        first_role: dict[int, str] = {}
        for sp in _speeches(g):
            roles_claimed = self_role_claims(sp.content)
            for r in roles_claimed:
                if r == "seer":
                    claimants.add(sp.actor)
                first_role.setdefault(sp.actor, r)
            if sp.actor not in claimants:
                continue
            for seat, verdict in check_claims(sp.content):
                if (sp.actor, seat, verdict) in seen_pairs or seat == sp.actor:
                    continue
                seen_pairs.add((sp.actor, seat, verdict))
                truth = "wolf" if g.roles.get(seat) == WOLF else "good"
                if sp.actor == seer:
                    if seat not in checks:
                        real_verdicts["fabricated_target"] += 1
                        rc.append(0)
                    else:
                        honest = checks[seat] == verdict
                        real_verdicts["honest" if honest else "flipped"] += 1
                        rc.append(1 if honest else 0)
                else:
                    fake_rows[(g.team(sp.actor), verdict, truth)] += 1
                    fc.append(1 if verdict == truth else 0)
                    if len(examples) < 6:
                        examples.append({"game": os.path.basename(g.path), "seat": sp.actor, "role": g.roles[sp.actor],
                                         "claimed": [seat, verdict], "truth": truth, "speech": sp.content[:120]})
        for seat, r in first_role.items():
            if r in ("hunter", "guard"):
                continue                      # not in the 8p deal: any such claim is false, but rare
            st[g.team(seat)].append(1 if role_matches(r, g.roles[seat]) else 0)
        for team in st:
            self_claim_true[team].append(st[team])
        for c in claimants:
            seer_claim_by_team[g.roles[c] if g.roles[c] in ("seer", WOLF) else "other_village"] += 1
        wolves_faking = [c for c in claimants if g.roles[c] == WOLF]
        wolf_fake_seer_games += 1 if wolves_faking else 0
        wolf_seats_faking += len(wolves_faking)
        seer_ever_claimed += 1 if seer in claimants else 0
        real_claims.append(rc)
        fake_claims.append(fc)
    n_wolf_seats = sum(len(g.wolves()) for g in games)
    return {
        "seer_claims_seer": {"k": seer_ever_claimed, "n": len(games), "p": wilson(seer_ever_claimed, len(games))[0],
                             "wilson": list(wilson(seer_ever_claimed, len(games))[1:])},
        "seer_check_honest": prop(real_claims), "seer_check_breakdown": dict(real_verdicts),
        "fake_claim_matches_truth": prop(fake_claims),
        "fake_claim_rows": [{"team": k[0], "said": k[1], "truth": k[2], "n": v} for k, v in sorted(fake_rows.items())],
        "seer_claimants_by_role": dict(seer_claim_by_team),
        "wolf_fake_seer_games": {"k": wolf_fake_seer_games, "n": len(games), "p": wilson(wolf_fake_seer_games, len(games))[0],
                                 "wilson": list(wilson(wolf_fake_seer_games, len(games))[1:])},
        "wolf_seats_fake_seer": {"k": wolf_seats_faking, "n": n_wolf_seats, "p": wilson(wolf_seats_faking, n_wolf_seats)[0],
                                 "wilson": list(wilson(wolf_seats_faking, n_wolf_seats)[1:])},
        "self_claim_true": {team: prop(v) for team, v in self_claim_true.items()},
        "fake_examples": examples,
    }


def _public_basis(g: Game, before: int, seat: int, role: str) -> bool:
    """Whether the table had said anything that grounds 'seat is role' before journal seq `before`.

    Generous on purpose (it only ever REMOVES leak flags): the seat's own
    self-claim of a matching role, or any earlier speech that puts "seat号" and a
    word of the same class in one clause -- a seer report, an accusation, a
    rumour all count. For 'good', dying at night counts as well.
    """
    words = {"seer": "预言家", "witch": "女巫", "villager": "村民|平民", WOLF: "狼|查杀", "good": "好人|金水|不是狼"}[role]
    cls = re.compile(rf"{seat}\s*号[^，,。；;！!？?\n]*?(?:{words})|(?:{words})[^，,。；;！!？?\n]*?{seat}\s*号")
    for e in g.entries:
        if e.seq >= before:
            break
        if e.task != "speak" or not e.content:
            continue
        if e.actor == seat and any(r == role or (role == "good" and r != WOLF) for r in self_role_claims(e.content)):
            return True
        if cls.search(e.content):
            return True
    if role == "good":
        night_deaths = {n["target"] for n in g.night if n.get("action") == "night_kill"}
        if seat in night_deaths:
            return True
    return False


def _private_basis(g: Game, speaker: int, seat: int, role: str, round_no: int) -> str | None:
    if g.roles.get(speaker) == WOLF and seat in g.wolves():
        return "teammate"
    if g.roles.get(speaker) == "seer":
        for n in g.night:
            if n.get("action") == "night_check" and n.get("player_id") == speaker and n.get("target") == seat \
                    and n.get("round", 0) <= round_no:
                return "own_check"
    if g.roles.get(speaker) == "witch" and role == "good":
        for n in g.night:
            if n.get("action") == "night_kill" and n.get("target") == seat and n.get("round", 0) <= round_no:
                return "saw_kill"
    return None


def leakage_metrics(games: list[Game]) -> dict:
    """Definite role statements about other seats, sorted by what could ground them.

    A seat that cannot know a role (no public basis, no private basis that its
    own role grants) and states it correctly anyway is a LEAK CANDIDATE. Because
    a guess is right by chance (2 of 7 others are wolves), the test that matters
    is aggregate: the accuracy of unsupported statements against the chance
    rate those same statements would have. Wolves naming a teammate as a wolf
    is counted separately (self-exposure); naming a teammate as good is
    deception that uses private knowledge, legal in the game.
    """
    total, unsupported_acc, chance = [], [], []
    wolf_expose, wolf_vouch, special_correct = 0, 0, []
    kinds = Counter()
    flagged = []
    for g in games:
        tot, acc = [], []
        for sp in _speeches(g):
            for a in role_assertions(sp.content, sp.actor):
                tot.append(1)
                actual = g.roles.get(a.seat, "")
                correct = role_matches(a.role, actual)
                priv = _private_basis(g, sp.actor, a.seat, a.role, sp.round)
                if priv == "teammate":
                    if a.role == WOLF:
                        wolf_expose += 1
                    elif a.role in ("good", "villager", "seer", "witch"):
                        wolf_vouch += 1
                    kinds["wolf_about_teammate"] += 1
                    continue
                if priv:
                    kinds[f"private:{priv}"] += 1
                    continue
                if _public_basis(g, sp.seq, a.seat, a.role):
                    kinds["public_basis"] += 1
                    continue
                kinds["unsupported"] += 1
                acc.append(1 if correct else 0)
                # chance: what fraction of the seats the speaker cannot see into hold this class
                others = [s for s in g.roles if s != sp.actor and not (g.roles.get(sp.actor) == WOLF and s in g.wolves())]
                chance.append(sum(1 for s in others if role_matches(a.role, g.roles[s])) / max(1, len(others)))
                if a.role in ("seer", "witch"):
                    special_correct.append(1 if correct else 0)
                if correct and len(flagged) < 10:
                    flagged.append({"game": os.path.basename(g.path), "round": sp.round, "seat": sp.actor,
                                    "speaker_role": g.roles[sp.actor], "said": a.span, "truth": actual,
                                    "speech": sp.content[:160]})
        total.append(tot)
        unsupported_acc.append(acc)
    acc_all = prop(unsupported_acc)
    return {"assertions": sum(map(len, total)), "speeches": sum(len(_speeches(g)) for g in games),
            "by_basis": dict(kinds), "unsupported_correct": acc_all,
            "unsupported_chance_mean": mean(chance),
            "special_role_unsupported_correct": {"k": sum(special_correct), "n": len(special_correct)},
            "wolf_self_exposure": wolf_expose, "wolf_vouch_teammate": wolf_vouch,
            "leak_candidates": flagged}


def structural_leak_metrics(games: list[Game]) -> dict:
    """Two leak tests that need no text parsing, at moments when nothing public points anywhere.

    Night-1 kill -> seer: the pack chooses before anyone has said a word, so it
    cannot know the seer; the hit rate should sit at 1/(non-wolf seats).
    Blind day-1 ballot -> wolf: a villager or witch voting on day 1 in a game
    where no seer claimant announced a wolf check that day has nothing public
    to aim with; the rate of ballots landing on a wolf should sit at chance
    (wolves among the seats it could vote for, after removing itself, the seer
    claimants and the seats they announced as good). Talk on day 1 is public,
    so a rate above chance is ALSO what good deduction looks like -- this test
    can clear the harness (rate at chance), it cannot convict it alone.
    """
    kill_hit, kill_chance = [], []
    blind_hit, blind_chance = [], []
    for g in games:
        props = [e.target for e in g.entries if e.task == "night" and e.round == 1 and e.action.get("name") == "night_kill"
                 and e.target is not None]
        seer = next((s for s, r in g.roles.items() if r == "seer"), None)
        if props:
            target = Counter(props).most_common(1)[0][0]
            kill_hit.append([1 if target == seer else 0])
            kill_chance.append(1 / max(1, len(g.roles) - len(g.wolves())))
        claimants, cleared, wolf_check_day1 = set(), set(), False
        for sp in _speeches(g):
            if sp.round != 1:
                continue
            if "seer" in self_role_claims(sp.content):
                claimants.add(sp.actor)
            if sp.actor in claimants:
                for seat, v in check_claims(sp.content):
                    wolf_check_day1 |= v == "wolf"
                    if v == "good":
                        cleared.add(seat)
        votes = [e for e in g.entries if e.task == "vote" and e.round == 1]
        voters = {e.actor for e in votes}
        hits = []
        if not wolf_check_day1:
            for e in votes:
                if g.roles.get(e.actor) in ("villager", "witch") and isinstance(e.target, int):
                    # what a reasonable villager would aim at: not itself, not a seat a seer
                    # claimant vouched for, not the claimant -- conservative, it raises chance
                    cands = (voters - {e.actor} - cleared - claimants) or (voters - {e.actor})
                    hits.append(1 if g.roles.get(e.target) == WOLF else 0)
                    blind_chance.append(len(cands & g.wolves()) / max(1, len(cands)))
        blind_hit.append(hits)
    return {"night1_kill_hits_seer": prop(kill_hit), "night1_chance": mean(kill_chance),
            "blind_day1_ballot_hits_wolf": prop(blind_hit), "blind_day1_chance": mean(blind_chance),
            "blind_day1_games": sum(1 for h in blind_hit if h)}


def hallucination_metrics(games: list[Game]) -> dict:
    per_speech = [[1 if e.turn.get("hallucinated_citations") else 0 for e in g.entries if e.task == "speak"] for g in games]
    counts = [[len(e.turn.get("hallucinated_citations") or []) for e in g.entries if e.task == "speak"] for g in games]
    kinds = Counter(h.get("kind") for g in games for e in g.entries for h in e.turn.get("hallucinated_citations") or [])
    return {"speeches_with_unsupported_citation": prop(per_speech), "citations": sum(map(sum, counts)),
            "per_100_speeches": dist([[100 * sum(c) / max(1, len(c))] for c in counts]), "kinds": dict(kinds)}


def outcome_metrics(games: list[Game]) -> dict:
    v = [1 if g.winner == "village" else 0 for g in games]
    k, n = sum(v), len(v)
    p, lo, hi = wilson(k, n)
    rounds = [max((e.round for e in g.entries if e.task != "review"), default=0) for g in games]
    return {"village_wins": k, "n": n, "village_win_rate": p, "wilson": [lo, hi],
            "werewolf_win_rate": 1 - p if p is not None else None, "werewolf_wilson": [1 - hi, 1 - lo] if p is not None else None,
            "rounds": dist([[r] for r in rounds])}


# ============================================================ report

def evaluate(games: list[Game]) -> dict:
    dates = sorted(g.mtime for g in games)
    return {
        "meta": {"games": len(games), "models": sorted({g.model for g in games}), "variants": sorted({g.variant for g in games}),
                 "seeds": sorted(g.seed for g in games), "parallel": sorted({g.parallel for g in games}, key=str),
                 "turns": sum(len(g.entries) for g in games), "aligned": all(g.aligned for g in games),
                 "date_range": [_dt.datetime.fromtimestamp(dates[0]).isoformat(timespec="minutes"),
                                _dt.datetime.fromtimestamp(dates[-1]).isoformat(timespec="minutes")] if dates else None},
        "cost": cost_metrics(games), "latency": latency_metrics(games), "reliability": reliability_metrics(games),
        "say_do": say_do_metrics(games), "seer": seer_metrics(games), "leakage": leakage_metrics(games),
        "structural_leak": structural_leak_metrics(games),
        "hallucination": hallucination_metrics(games), "outcome": outcome_metrics(games),
    }


def _f(x, nd=1, pct=False):
    if x is None:
        return "—"
    return f"{100 * x:.{nd}f}%" if pct else (f"{x:,.0f}" if nd == 0 else f"{x:,.{nd}f}")


def fmt_prop(p: dict, nd=1, boot=True) -> str:
    if not p or p.get("n", 0) == 0:
        return "— (n=0)"
    lo, hi = p.get("wilson", [None, None])
    s = f"{_f(p['p'], nd, True)} [{_f(lo, nd, True)}, {_f(hi, nd, True)}] ({p['k']}/{p['n']})"
    b = p.get("boot") or [None, None]
    if boot and b[0] is not None:
        s += f"；簇 [{_f(b[0], nd, True)}, {_f(b[1], nd, True)}]"
    return s


def fmt_stat(d: dict, key="mean", nd=1) -> str:
    s = d[key]
    if s["est"] is None:
        return "—"
    ci = f" [{_f(s['lo'], nd)}, {_f(s['hi'], nd)}]" if s["lo"] is not None else ""
    return f"{_f(s['est'], nd)}{ci}"


def fmt_dist(d: dict, nd=1) -> str:
    return f"{fmt_stat(d, 'mean', nd)} | {fmt_stat(d, 'p50', nd)} | {fmt_stat(d, 'p95', nd)} | {d['n']}"


TASK_ZH = {"night": "夜晚", "speak": "发言", "vote": "投票", "review": "复盘"}
_ORDER = {"night": 0, "speak": 1, "vote": 2, "review": 3}


def headline_rows(m: dict) -> list[tuple[str, str]]:
    c, lat, r, s, sd, lk, h, o = (m[k] for k in ("cost", "latency", "reliability", "seer", "say_do", "leakage", "hallucination", "outcome"))
    sl = m["structural_leak"]
    g = m["meta"]["games"]
    return [
        ("好人胜率（Wilson）", f"{_f(o['village_win_rate'], 1, True)} [{_f(o['wilson'][0], 1, True)}, {_f(o['wilson'][1], 1, True)}] ({o['village_wins']}/{o['n']} 局)"),
        ("每局 token（CLI 计，均值）", f"{fmt_stat(c['tokens_per_game_cli'], 'mean', 0)} (n={g} 局)"),
        ("每局 token（本地估算，均值）", f"{fmt_stat(c['tokens_per_game_local'], 'mean', 0)} (n={g} 局)"),
        ("每局模型调用数（均值）", f"{fmt_stat(c['calls_per_game'], 'mean', 1)} (n={g} 局)"),
        ("每次调用 token（CLI）", f"{fmt_stat(c['tokens_per_call_cli'], 'mean', 0)} (n={g} 局)"),
        ("单次模型调用延迟 p50 / p95（ms）", f"{fmt_stat(lat['call_ms'], 'p50', 0)} / {fmt_stat(lat['call_ms'], 'p95', 0)} (n={lat['call_ms']['n']} 次)"),
        ("每回合墙钟 p50 / p95（s）", f"{fmt_stat(lat['turn_wall_s'], 'p50', 2)} / {fmt_stat(lat['turn_wall_s'], 'p95', 2)} (n={lat['turn_wall_s']['n']} 回合)"),
        ("整局时长（s，均值）", f"{fmt_stat(lat['game_s'], 'mean', 1)} (n={g} 局)"),
        ("ReAct 步成功率（未被拦）", fmt_prop(r["step_success"])),
        ("回合内有拒绝并重试", fmt_prop(r["turn_rejected_and_retried"])),
        ("回合走到强制终结（步数用尽）", fmt_prop(r["turn_forced_terminal"])),
        ("回合兜底（默认动作）", fmt_prop(r["turn_fallback"])),
        ("投票回合被 L3 言行一致拦截", fmt_prop(r["vote_turn_saydo_blocked"])),
        ("言行一致（任一档，最终票）", fmt_prop(sd["consistent"]["any"]) + f"；覆盖 {_f(sd['coverage']['any']['p'], 1, True)} ({sd['coverage']['any']['k']}/{sd['pairs']})"),
        ("预言家公开验人与引擎一致", fmt_prop(s["seer_check_honest"])),
        ("狼人悍跳预言家（局）", f"{_f(s['wolf_fake_seer_games']['p'], 1, True)} [{_f(s['wolf_fake_seer_games']['wilson'][0], 1, True)}, {_f(s['wolf_fake_seer_games']['wilson'][1], 1, True)}] ({s['wolf_fake_seer_games']['k']}/{s['wolf_fake_seer_games']['n']})"),
        ("泄露检验：首夜刀中预言家 vs 随机", fmt_prop(sl["night1_kill_hits_seer"], boot=False) + f"；随机 {_f(sl['night1_chance'], 1, True)}"),
        ("泄露检验：首日盲投（无查杀）投中狼 vs 随机", fmt_prop(sl["blind_day1_ballot_hits_wolf"]) + f"；随机 {_f(sl['blind_day1_chance'], 1, True)}"),
        ("确定语气的他人身份断言（无依据者）", f"{lk['by_basis'].get('unsupported', 0)} 条 / 共 {lk['assertions']} 条断言 / {lk['speeches']} 篇发言"),
        ("发言含未查证引用", fmt_prop(h["speeches_with_unsupported_citation"])),
    ]


def findings(m: dict, comparisons: dict[str, dict]) -> list[str]:
    """The few sentences a reader should leave with, computed from the numbers (never typed in)."""
    c, lat, r, sd, s = m["cost"], m["latency"], m["reliability"], m["say_do"], m["seer"]
    sl, ga = m["structural_leak"], m["say_do"]["guard_audit"]
    sdc = r["blocked_step_cost"].get("say_do_consistency", {"steps": 0, "token_share": 0, "ms_share": 0})
    vote = lat["by_task"].get("vote", {}).get("turn_wall_s")
    out = [
        f"**L3 言行一致守卫是最大的浪费源，且几乎全是误拦**：{sdc['steps']} 个被拦步骤占全部 ReAct token 的 {_f(sdc['token_share'], 1, True)}、"
        f"模型耗时的 {_f(sdc['ms_share'], 1, True)}；被拦的 {ga['saydo_blocked_votes']} 个投票回合里，{ga['saydo_blocked_votes_from_non_vote_trigger']} 个由“验出N号/给出N号”"
        f"这类非投票短语触发，其余是解释过去的票，与本脚本读到的本轮意向一致的为 {ga['saydo_blocked_votes_agreeing_with_ours']} 个。"
        f"{r['turn_forced_terminal']['k']} 次强制终结中 {r['forced_terminal_by_block'].get('say_do_consistency', 0)} 次源于它，投票回合墙钟 p95 {fmt_stat(vote, 'p95', 1) if vote else '—'} s 也因此而来。",
        f"**CLI 报告的 token 是本地估算的 {fmt_stat(c['cli_over_local'], 'mean', 2)} 倍**（含 cache 读写），按本地估算设的预算上限会低估真实用量。",
        f"**诚实度**：真预言家公开的验人 {s['seer_check_honest']['k']}/{s['seer_check_honest']['n']} 与引擎一致；狼人 0 次悍跳预言家"
        f"（{s['wolf_fake_seer_games']['k']}/{s['wolf_fake_seer_games']['n']} 局），自报身份全是好人阵营（真话 {s['self_claim_true'][WOLF]['k']}/{s['self_claim_true'][WOLF]['n']}）——haiku 的狼很被动。",
        f"**没有泄露信号**：首夜狼刀命中预言家 {sl['night1_kill_hits_seer']['k']}/{sl['night1_kill_hits_seer']['n']}（随机 {_f(sl['night1_chance'], 1, True)}）；"
        f"首日无查杀时村民/女巫投中狼 {_f(sl['blind_day1_ballot_hits_wolf']['p'], 1, True)}，CI 覆盖随机 {_f(sl['blind_day1_chance'], 1, True)}；无依据的确定身份断言 {m['leakage']['by_basis'].get('unsupported', 0)} 条。",
        f"**言行一致只能测到很少的发言**：有可解析表态的仅 {_f(sd['coverage']['any']['p'], 1, True)}（{sd['coverage']['any']['k']}/{sd['pairs']}），其中最终票一致 {_f(sd['consistent']['any']['p'], 1, True)}。",
    ]
    seq = next((v for k, v in comparisons.items() if "pilot" in k or "顺序" in k), None)
    if seq:
        out.append(f"**提速**：整局 {fmt_stat(seq['latency']['game_s'], 'mean', 0)} s（顺序，{seq['meta']['games']} 局）→ {fmt_stat(lat['game_s'], 'mean', 0)} s（并行，{m['meta']['games']} 局）；"
                   f"每次调用 token {fmt_stat(seq['cost']['tokens_per_call_cli'], 'mean', 0)} → {fmt_stat(c['tokens_per_call_cli'], 'mean', 0)}，"
                   f"单次调用 p50 {fmt_stat(seq['latency']['call_ms'], 'p50', 0)} → {fmt_stat(lat['call_ms'], 'p50', 0)} ms。顺序组只有 2 局，方向可信、幅度不精确。")
    return out


def render(m: dict, comparisons: dict[str, dict]) -> str:
    meta = m["meta"]
    c, lat, r = m["cost"], m["latency"], m["reliability"]
    sd, s, lk, h, o = m["say_do"], m["seer"], m["leakage"], m["hallucination"], m["outcome"]
    L = []
    w = L.append
    w("# 基线评测 A：真实模型（claude-cli:haiku，8 人局）\n")
    w(f"样本：**{meta['games']} 局**，{meta['turns']} 个回合；模型 {', '.join(meta['models'])}；板子 {', '.join(meta['variants'])}；"
      f"并行度 {', '.join(map(str, meta['parallel']))}；种子 {meta['seeds'][0]}–{meta['seeds'][-1]}；"
      f"日志时间 {meta['date_range'][0]} — {meta['date_range'][1]}（文件修改时间）。")
    w("全部指标以**引擎记录**（journal / night 日志 / 发牌）为准，不采信模型自述。比例给 Wilson 95% CI，"
      "并给“簇”区间（按局重抽样的 bootstrap，B=2000，回合在局内相关时更诚实）；均值/分位数只给按局 bootstrap CI。\n")
    w("## 核心数字\n")
    w("| 指标 | 数值 [95% CI] (n) |\n|---|---|")
    for k, v in headline_rows(m):
        w(f"| {k} | {v} |")
    w("")

    w("## 读数要点\n")
    for line in findings(m, comparisons):
        w(f"- {line}")
    w("")
    w("## 1. 成本\n")
    w("| 指标 | 均值 [CI] | p50 [CI] | p95 [CI] | n |\n|---|---|---|---|---|")
    for name, key, nd in (("每局 token（CLI 报告：input+cache 读写+output）", "tokens_per_game_cli", 0),
                          ("每局 token（本地估算，预算用）", "tokens_per_game_local", 0),
                          ("每局 prompt token（CLI）", "prompt_per_game_cli", 0),
                          ("每局 completion token（CLI）", "completion_per_game_cli", 0),
                          ("每局模型调用", "calls_per_game", 1), ("每局回合", "turns_per_game", 1),
                          ("每次调用 token（CLI）", "tokens_per_call_cli", 0), ("每次调用 token（本地）", "tokens_per_call_local", 0),
                          ("CLI / 本地 比值", "cli_over_local", 2)):
        w(f"| {name} | {fmt_dist(c[key], nd)} |")
    w(f"\n调用数核对：逐回合（ReAct 步 + 笔记）合计 {'等于' if c['calls_accounted'] else '不等于'} usage.calls；"
      f"逐回合 token 合计与 CLI 报告总量相等的局 {c['turn_sum_equal_games']}/{len(c['per_game'])}，"
      f"最小比值 {_f(c['turn_sum_over_cli_min'], 3)}（差额来自没有落进任何 TurnResult 的调用，例如超时后重试的那一次）。"
      "按任务的拆分用逐回合数字。\n")
    w("按任务（每回合）：\n\n| 任务 | token/回合 均值 [CI] | 调用/回合 均值 [CI] | 每局该任务 token 均值 [CI] | token 占比 | 回合数 |\n|---|---|---|---|---|---|")
    for task, d in sorted(c["by_task"].items(), key=lambda kv: _ORDER.get(kv[0], 9)):
        w(f"| {TASK_ZH.get(task, task)} | {fmt_stat(d['tokens_per_turn'], 'mean', 0)} | {fmt_stat(d['calls_per_turn'], 'mean', 2)} | "
          f"{fmt_stat(d['tokens_per_game'], 'mean', 0)} | {_f(d['share_of_tokens'], 1, True)} | {d['tokens_per_turn']['n']} |")
    w("")

    w("## 2. 延迟\n")
    w("| 指标 | 均值 [CI] | p50 [CI] | p95 [CI] | n |\n|---|---|---|---|---|")
    w(f"| 单次模型调用（ReAct 步，ms） | {fmt_dist(lat['call_ms'], 0)} |")
    if lat["note_ms"]:
        w(f"| 笔记调用（ms，后台线程，不在关键路径） | {fmt_dist(lat['note_ms'], 0)} |")
    w(f"| 每回合模型耗时（s，含笔记） | {fmt_dist(lat['turn_model_s'], 2)} |")
    w(f"| 每回合墙钟（s，committed−asked） | {fmt_dist(lat['turn_wall_s'], 2)} |")
    w(f"| 整局（s） | {fmt_dist(lat['game_s'], 1)} |")
    w("\n按阶段：\n\n| 阶段 | 回合墙钟 s（均值/p50/p95） | 单个阶段墙钟 s（均值/p50/p95） | 每局该阶段合计 s（均值） | 阶段数 |\n|---|---|---|---|---|")
    for task, d in sorted(lat["by_task"].items(), key=lambda kv: _ORDER.get(kv[0], 9)):
        w(f"| {TASK_ZH.get(task, task)} | {fmt_stat(d['turn_wall_s'], 'mean', 2)} / {fmt_stat(d['turn_wall_s'], 'p50', 2)} / {fmt_stat(d['turn_wall_s'], 'p95', 2)} | "
          f"{fmt_stat(d['phase_wall_s'], 'mean', 1)} / {fmt_stat(d['phase_wall_s'], 'p50', 1)} / {fmt_stat(d['phase_wall_s'], 'p95', 1)} | "
          f"{fmt_stat(d['phase_wall_per_game_s'], 'mean', 1)} | {d['phase_wall_s']['n']} |")
    if lat["asked_ahead"]:
        w(f"\n提前发问（ahead=true，与上一回合重叠）的回合占比：{fmt_prop(lat['asked_ahead'])}。")
    w("")

    if comparisons:
        w("## 3. 提速前后对比（顺序 vs 并行）\n")
        w("| 组 | 局 | 每回合墙钟 s 均值/p50/p95 | 每回合模型耗时 s 均值 | 单次调用 ms p50 | token/调用（CLI） | 整局 s 均值 |"
          + "".join(f" {TASK_ZH[t]}阶段 s/局 |" for t in ("night", "speak", "vote", "review")) + "\n|---|---|---|---|---|---|---|---|---|---|---|")
        for name, cm in [("基线(base)", m)] + list(comparisons.items()):
            cl, cc = cm["latency"], cm["cost"]
            row = (f"| {name} | {cm['meta']['games']} | {fmt_stat(cl['turn_wall_s'], 'mean', 2)} / {fmt_stat(cl['turn_wall_s'], 'p50', 2)} / {fmt_stat(cl['turn_wall_s'], 'p95', 2)} | "
                   f"{fmt_stat(cl['turn_model_s'], 'mean', 2)} | {fmt_stat(cl['call_ms'], 'p50', 0)} | {fmt_stat(cc['tokens_per_call_cli'], 'mean', 0)} | {fmt_stat(cl['game_s'], 'mean', 1)} |")
            for t in ("night", "speak", "vote", "review"):
                d = cl["by_task"].get(t)
                row += f" {fmt_stat(d['phase_wall_per_game_s'], 'mean', 1) if d else '—'} |"
            w(row)
        w("\n顺序日志（pilot）没有 timings：其“回合墙钟”取该回合模型耗时（含笔记，因为顺序跑时笔记在关键路径上），"
          "阶段时长取阶段内回合耗时之和。少于 5 局的组不给 CI（两局 bootstrap 无意义），只能当方向性证据。\n")

    w("## 4. 可靠性\n")
    w("| 指标 | 比例 [Wilson] (k/n)；簇 bootstrap |\n|---|---|")
    w(f"| ReAct 步未被拦（工具调用成功） | {fmt_prop(r['step_success'])} |")
    w(f"| 输出畸形（无可用工具调用） | {fmt_prop(r['malformed_step'], 2)} |")
    for k, p in r["block_reason_per_step"].items():
        w(f"| 步被拦：{k} | {fmt_prop(p, 2)} |")
    w(f"| 回合内至少一次拒绝并重试 | {fmt_prop(r['turn_rejected_and_retried'])} |")
    w(f"| 回合强制终结（步数用尽） | {fmt_prop(r['turn_forced_terminal'])} |")
    w(f"| 回合兜底（默认动作） | {fmt_prop(r['turn_fallback'])} |")
    w(f"| 回合循环被打断 | {fmt_prop(r['turn_loop_broken'])} |")
    w(f"| 回合有超时 | {fmt_prop(r['turn_timeout'])} |")
    w(f"| 回合有传输重试 | {fmt_prop(r['turn_transport_retry'])} |")
    w(f"| 投票回合被 L3 say_do 拦过 | {fmt_prop(r['vote_turn_saydo_blocked'])} |")
    w("\n被拦步骤的花费（这些调用的产出被丢弃）：\n\n| 拦截原因 | 步数 | token | 占全部 ReAct token | 模型耗时 s | 占全部 ReAct 耗时 |\n|---|---|---|---|---|---|")
    for k, v in sorted(r["blocked_step_cost"].items(), key=lambda kv: -kv[1]["tokens"]):
        w(f"| {k} | {v['steps']} | {v['tokens']:,} | {_f(v['token_share'], 1, True)} | {v['ms'] / 1000:,.0f} | {_f(v['ms_share'], 1, True)} |")
    w(f"\n崩溃局 {r['games_crashed']}，不可用局 {r['games_unusable']}。\n")

    ga = sd["guard_audit"]
    w("## 5. 行为\n")
    w("### 5.1 言行一致（本轮发言 vs 本轮最终票）\n")
    w(f"可配对（同一人同一轮既发言又投票）：{sd['pairs']} 对。\n")
    w("| 档 | 覆盖率（有可解析声明） | 一致率（最终票） |\n|---|---|---|")
    for t, zh in (("intent", "明确投票意向"), ("accuse", "点名怀疑"), ("vouch", "保人/不投"), ("any", "任一档（强档优先）")):
        w(f"| {zh} | {fmt_prop(sd['coverage'][t])} | {fmt_prop(sd['consistent'][t])} |")
    w(f"\n用模型**第一次提出**的票（L3 拦截前）算任一档一致率：{fmt_prop(sd['consistent_first_ballot'])}。\n")
    w(f"L3 守卫自身的“宣布投票”正则审计：它在 {ga['speeches_guard_reads_a_vote']} 篇发言里读出了投票意向，"
      f"其中 {ga['trigger_not_a_vote_intent']} 篇的触发词根本不是投票（“验出3号”“给出1号”里的“出N号”），"
      f"{ga['vote_word_but_not_this_rounds_intent']} 篇是投票词但不是本轮意向（“我投7号，是因为…”解释昨天的票、条件句等），"
      f"与本脚本的明确意向一致的只有 {ga['agree_with_ours']} 篇。被 say_do 拦过的投票回合中有发言触发的 {ga['saydo_blocked_votes']} 个，"
      f"其中 {ga['saydo_blocked_votes_from_non_vote_trigger']} 个由非投票短语触发，只有 {ga['saydo_blocked_votes_agreeing_with_ours']} 个与本脚本读到的意向一致。"
      f"触发片段 Top：{', '.join(f'{t}×{n}' for t, n in ga['triggers'][:8])}。\n")
    w("### 5.2 预言家与身份声明\n")
    w("| 指标 | 数值 |\n|---|---|")
    w(f"| 真预言家公开跳身份（局） | {s['seer_claims_seer']['k']}/{s['seer_claims_seer']['n']} = {_f(s['seer_claims_seer']['p'], 1, True)} [{_f(s['seer_claims_seer']['wilson'][0], 1, True)}, {_f(s['seer_claims_seer']['wilson'][1], 1, True)}] |")
    w(f"| 真预言家报出的验人与引擎一致 | {fmt_prop(s['seer_check_honest'])}；明细 {s['seer_check_breakdown']} |")
    w(f"| 狼人悍跳预言家（局） | {s['wolf_fake_seer_games']['k']}/{s['wolf_fake_seer_games']['n']} = {_f(s['wolf_fake_seer_games']['p'], 1, True)} [{_f(s['wolf_fake_seer_games']['wilson'][0], 1, True)}, {_f(s['wolf_fake_seer_games']['wilson'][1], 1, True)}] |")
    w(f"| 狼人座位悍跳预言家 | {s['wolf_seats_fake_seer']['k']}/{s['wolf_seats_fake_seer']['n']} = {_f(s['wolf_seats_fake_seer']['p'], 1, True)} [{_f(s['wolf_seats_fake_seer']['wilson'][0], 1, True)}, {_f(s['wolf_seats_fake_seer']['wilson'][1], 1, True)}] |")
    w(f"| 自称预言家者按真实身份 | {s['seer_claimants_by_role']} |")
    w(f"| 假预言家报出的验人与真相一致 | {fmt_prop(s['fake_claim_matches_truth'])} |")
    w(f"| 自报身份为真：好人阵营 | {fmt_prop(s['self_claim_true']['village'])} |")
    w(f"| 自报身份为真：狼人阵营 | {fmt_prop(s['self_claim_true'][WOLF])} |")
    w("")
    w("### 5.3 私有信息泄露\n")
    w(f"{lk['speeches']} 篇发言中抽出 {lk['assertions']} 条“确定语气、非转述”的他人身份断言，按依据分类：{lk['by_basis']}。\n")
    w("| 指标 | 数值 |\n|---|---|")
    w(f"| 无任何依据的断言 正确率 | {fmt_prop(lk['unsupported_correct'])} |")
    w(f"| 同一批断言的随机猜中率（按说话人看不见的座位算） | {_f(lk['unsupported_chance_mean'], 1, True)} |")
    w(f"| 其中精确神职（预言家/女巫）断言猜中 | {lk['special_role_unsupported_correct']['k']}/{lk['special_role_unsupported_correct']['n']} |")
    sl = m["structural_leak"]
    w(f"| 首夜狼刀命中预言家（无任何公开信息时） | {fmt_prop(sl['night1_kill_hits_seer'], boot=False)}；随机 {_f(sl['night1_chance'], 1, True)} |")
    w(f"| 首日“盲投”命中狼（村民/女巫，当天无人报查杀，{sl['blind_day1_games']} 局） | {fmt_prop(sl['blind_day1_ballot_hits_wolf'])}；随机 {_f(sl['blind_day1_chance'], 1, True)} |")
    w(f"| 狼人公开指认队友为狼（自曝） | {lk['wolf_self_exposure']} |")
    w(f"| 狼人确定语气为队友作保（利用私有信息的欺骗） | {lk['wolf_vouch_teammate']} |")
    w("")
    w("### 5.4 幻觉引用\n")
    w(f"发言回合中含未查证引用（turns[].hallucinated_citations 非空）：{fmt_prop(h['speeches_with_unsupported_citation'], 2)}；"
      f"共 {h['citations']} 条，类型 {h['kinds']}；每百篇发言 {fmt_stat(h['per_100_speeches'], 'mean', 2)}。\n")

    w("## 6. 结局\n")
    w(f"好人胜 {o['village_wins']}/{o['n']}：好人胜率 {_f(o['village_win_rate'], 1, True)} [{_f(o['wilson'][0], 1, True)}, {_f(o['wilson'][1], 1, True)}]，"
      f"狼人胜率 {_f(o['werewolf_win_rate'], 1, True)} [{_f(o['werewolf_wilson'][0], 1, True)}, {_f(o['werewolf_wilson'][1], 1, True)}]；"
      f"结束轮次 {fmt_dist(o['rounds'], 2)}（均值|p50|p95|n）。\n")

    w("## 怎么测的\n")
    w("- **成本**：每局 `usage`：`gateway_*` 为 CLI JSON 报告（input + cache 读 + cache 写 + output），`prompt/completion_tokens` 为发送前的本地估算（预算上限用它）。"
      "调用数 = 每回合非兜底 ReAct 步 + 1 次笔记（有 note_ms 或回合耗时大于步耗时之和）；与 usage.calls 逐局核对。")
    w("- **延迟**：单次调用 = react_trace[].latency_ms；回合墙钟 = timings 的 committed − asked；阶段 = journal 中同一任务的连续段，"
      "阶段墙钟 = 本段最后一次提交 − 上一段最后一次提交（即该阶段占用的游戏时钟）。")
    w("- **可靠性**：以 ReAct 步为单位看 `guard_blocked/block_reason`（schema 含畸形输出与非白名单工具；semantics 为参数/状态校验；"
      "say_do_consistency、directive_compliance、evidence 为 L3/L2 守卫）；以回合为单位看 `recovery`。")
    w("- **言行一致**：保守正则抽取三档声明（明确投票意向 > 点名怀疑 > 保人/不投），拒读过去时（投了/第一天/昨天/当时…）、条件句（如果/假如/是不是…）、"
      "转述（验/报/自称/起跳…），自相矛盾的座位丢弃，两个不同明确意向整篇不读；对比同轮引擎记录的票（abstain 记为不一致）。")
    w("- **预言家诚实**：只读自称预言家者的第一人称验人（“首夜查验了2号，结果是好人”，前面没有别的主语），裁决为同句内第一个好/狼词；"
      "与 night 日志中该真预言家的 night_check 结果比对（未验过却报 = 编造）。")
    w("- **泄露**：抽取确定语气、无对冲、非转述的“N号是X”；依据分三类：私有（狼队友 / 预言家自己验过 / 女巫看到被刀）、"
      "公开（此前任何发言在同一子句里把 N号 与同类词放在一起，或 N号自报该身份，或夜里被刀之于“好人”——故意放宽，只会减少泄露标记）、"
      "无依据。无依据断言的正确率若显著高于随机基线，才是泄露信号。复盘阶段（身份已公开）不计。"
      "另做两项不依赖文本的结构检验：首夜狼刀（多数狼的提议）命中预言家的比例 vs 1/6；"
      "首日当天无自称预言家报查杀的局里，村民/女巫的票落在狼上的比例 vs 可投座位中狼的占比。")
    w("- **CI**：比例 Wilson；“簇”与所有均值/分位数 = 按局有放回重抽样 2000 次的百分位区间（种子 0）。")
    w("- **复现**：`python -m werewolf_harness.scripts.eval_baseline 'runs/base/g*.json' --compare '顺序(pilot)=runs/pilot/g*.json' "
      "--compare '并行(fast)=runs/fast/g*.json' --out docs/baseline_A.md --json docs/baseline_A.json`；新模型/新板子跑完 `cli play --out` 后同一条命令即可出同格式报告。\n")
    w("## 注意事项\n")
    w("- 只有 haiku 一个模型、只有 8 人局（2 狼 / 预言家 / 女巫 / 4 民）、只有静态 playbook 与 L1+L2+L3 守卫；结论不外推到其它模型或板子。")
    w(f"- 言行一致的覆盖率很低（任一档 {_f(sd['coverage']['any']['p'], 1, True)}）：haiku 的发言多数是“先听发言再决定”，"
      "一致率只描述那一小部分有明确表态的发言；而且发言后才轮到别人发言，改票可能是合理的。")
    w("- 最终票已经被 L3 say_do 守卫“推”过（被拦后重提），因此最终票一致率偏高；第一次提出的票更接近模型本身。"
      "反过来，守卫的正则会把“验出N号”读成“出N号”=投票意向，详见 5.1 审计。")
    w("- 正则抽取有漏读与误读；每类都在 JSON 里留了样例（`inconsistent_examples`、`leak_candidates`、`fake_examples`、`guard_audit.examples`）供人工抽查。")
    w("- 对比组 pilot / fast 各只有 2 局，且顺序版与并行版的对局轨迹不同（同种子，模型输出不同），只能当方向性证据。")
    return "\n".join(L) + "\n"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, set):
        return sorted(o)
    if isinstance(o, float):
        return round(o, 6)
    return o


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="基线评测：成本 / 延迟 / 可靠性 / 行为 / 结局（以引擎记录为准）")
    ap.add_argument("glob", nargs="+", help="game logs written by `cli play --out`")
    ap.add_argument("--compare", action="append", default=[], metavar="NAME=GLOB",
                    help="another group of logs to compare latency/tokens against (repeatable)")
    ap.add_argument("--out", help="write the Markdown report here")
    ap.add_argument("--json", dest="json_out", help="write the metrics JSON here")
    a = ap.parse_args(argv)
    games = load(a.glob)
    if not games:
        raise SystemExit(f"没有匹配的日志: {a.glob}")
    metrics = evaluate(games)
    comps = {}
    for spec in a.compare:
        name, _, pat = spec.partition("=")
        cg = load([pat])
        if cg:
            comps[name] = {"meta": evaluate(cg)["meta"], "latency": latency_metrics(cg), "cost": cost_metrics(cg)}
    report = render(metrics, comps)
    out = {"base": metrics, "compare": comps}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(report)
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump(_jsonable(out), fh, ensure_ascii=False, indent=1)
    if not a.out:
        print(report)
    else:
        print("\n".join(f"{k}: {v}" for k, v in headline_rows(metrics)))
    return out


if __name__ == "__main__":
    main()
