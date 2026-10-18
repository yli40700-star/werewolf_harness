"""Paired experiment: strong model vs cheap model vs cascade, on the same deals.

    python -m werewolf_harness.scripts.paired_cascade --seeds 2001-2010 \
        --cheap claude-cli:haiku --strong claude-cli:sonnet --arms strong,cheap,cascade \
        --cascade-mode both --variant 8p --parallel 8 --record --out runs/cascade/

Design. The seed fixes the deal (roles, seats, the engine's own randomness), so
every arm plays the SAME tables; the only thing an arm changes is which model
answers. That makes the seed the unit of pairing: for each metric the runner
computes, per seed, (arm - strong) and bootstraps those differences over seeds.
Pairing removes the deal-to-deal variance that dominates werewolf outcomes --
a village that draws a seer check on night 1 wins whatever model plays it.

Layout:  <out>/<arm>/g<seed>.json  (+ .cassette.jsonl with --record), written by
`cli play` itself, so every log is an ordinary game log (`cli replay`,
`cli replay-llm`, eval_baseline all read it). A log that exists and parses is
not played again: kill the runner at any point and run the same command to
resume. Games run seed by seed, every arm of a seed before the next seed, so a
run stopped half way still holds complete pairs.

Metrics come from scripts/eval_baseline.py's functions, computed per game (so
they can be paired) and pooled per arm (with eval_baseline's game-clustered
bootstrap). Cost is per model: tokens as the CLI reports them (input + cache
reads/writes + output), priced with a per-model (input, output) list price --
see PRICES and --price. "Cost" in the report is USD at those prices AND
"cheap-equivalent tokens" (USD / the cheap model's blended price), so a strong
token counts as several cheap ones.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict

from . import eval_baseline as eb

# List prices, USD per million tokens (input, output). Source: anthropic.com/pricing
# (Claude Haiku 4.5 $1/$5, Sonnet 4.5 $3/$15, Opus 4.5 $5/$25), as known on 2026-10-14 --
# CHECK BEFORE QUOTING, prices change. `claude -p` with an alias runs the current
# model of that family. Cache reads (0.1x) and writes (1.25x) are priced here as plain
# input: the absolute USD is an upper bound, but the sonnet:haiku ratio is 3:1 for every
# token class, so the RELATIVE cost of two arms -- what this experiment is about -- is
# exact whatever the cache mix. The mock rows keep the same 3:1 ratio so a mock run
# exercises the arithmetic.
PRICES: dict[str, tuple[float, float]] = {
    "claude-cli:haiku": (1.0, 5.0), "claude-cli:sonnet": (3.0, 15.0), "claude-cli:opus": (5.0, 25.0),
    "mock-strong": (3.0, 15.0), "mock": (3.0, 15.0),
}
FLAKY_PRICE = (1.0, 5.0)              # any mock-flaky[:rate]

# What a game costs in wall-clock minutes and CLI tokens, for the estimate printed before
# a run. haiku: baseline A (30 games, 8p, parallel 8): 3.4 min, ~560k tokens. sonnet: not
# measured; assumed 2x slower per call and the same tokens (same prompts, similar replies).
ESTIMATES: dict[str, tuple[float, int]] = {
    "claude-cli:haiku": (3.4, 560_000), "claude-cli:sonnet": (6.8, 560_000), "claude-cli:opus": (10.0, 560_000),
}
DEFAULT_DELTA = 0.02                  # non-inferiority margin on the legal-action rate (2 points)
MODES = ("on-fail", "decisive", "both")


# ============================================================ planning

def parse_seeds(spec: str) -> list[int]:
    """"2001-2010", "1,2,5", "2001-2005,2010" -> sorted unique seeds."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def arm_spec(arm: str, cheap: str, strong: str, default_mode: str) -> dict:
    """strong | cheap | cascade | cascade-<mode> -> what `cli play` is told."""
    if arm == "strong":
        return {"arm": arm, "model": strong}
    if arm == "cheap":
        return {"arm": arm, "model": cheap}
    if arm == "cascade" or arm.startswith("cascade-"):
        mode = arm.split("-", 1)[1] if "-" in arm else default_mode
        if mode not in MODES:
            raise SystemExit(f"!! 不认识的级联方式 {arm}（可用：cascade、{'、'.join('cascade-' + m for m in MODES)}）")
        return {"arm": arm, "model": cheap, "strong": strong, "cascade": mode}
    raise SystemExit(f"!! 不认识的实验臂 {arm!r}（strong / cheap / cascade[-on-fail|-decisive|-both]）")


def log_path(out: str, arm: str, seed: int) -> str:
    return os.path.join(out, arm, f"g{seed}.json")


def done(path: str) -> bool:
    """A finished game: the log exists and parses (cli writes it at the very end; a run
    killed while writing leaves a file that does not parse, and is played again)."""
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return bool(d.get("journal")) and d.get("outcome") is not None
    except (OSError, ValueError):
        return False


def play_argv(spec: dict, seed: int, a: argparse.Namespace, path: str) -> list[str]:
    argv = ["play", "--variant", a.variant, "--seed", str(seed), "--model", spec["model"],
            "--parallel", str(a.parallel), "--out", path, "--guard", a.guard]
    if spec.get("cascade"):
        argv += ["--strong", spec["strong"], "--cascade", spec["cascade"]]
        if a.decisive:
            argv += ["--decisive", a.decisive]
        if a.low_confidence:
            argv.append("--low-confidence")
    if a.record:
        argv.append("--record")
    return argv


def estimate(specs: list[dict], seeds: list[int], todo: dict[str, int], est: dict) -> list[str]:
    """Lines saying how many games and roughly how long / how many tokens, before anything runs."""
    lines = []
    total_min = total_tok = 0.0
    for sp in specs:
        n = todo[sp["arm"]]
        if sp.get("cascade"):
            c, s = est.get(sp["model"]), est.get(sp["strong"])
            # unmeasured: between the two -- escalations and decisive turns move it toward strong
            per = ((c[0] + s[0]) / 2, c[1]) if c and s else None
            label = f"{sp['model']}→{sp['strong']}（{sp['cascade']}，粗估取两者之间）"
        else:
            per = est.get(sp["model"])
            label = sp["model"]
        if per:
            total_min += n * per[0]
            total_tok += n * per[1]
            lines.append(f"  {sp['arm']:<16} {label}: 要跑 {n}/{len(seeds)} 局，每局约 {per[0]:.1f} 分钟、{per[1] / 1000:.0f}k token "
                         f"→ {n * per[0]:.0f} 分钟、{n * per[1] / 1e6:.2f}M token")
        else:
            lines.append(f"  {sp['arm']:<16} {label}: 要跑 {n}/{len(seeds)} 局（没有这个模型的耗时估计）")
    lines.append(f"  合计约 {total_min / 60:.1f} 小时（一次一局；--jobs N 同时跑 N 局可缩短，但 CLI 额度消耗不变）、"
                 f"{total_tok / 1e6:.1f}M CLI token")
    return lines


def run_game(argv: list[str], subprocess_mode: bool) -> int:
    if subprocess_mode:
        return subprocess.run([sys.executable, "-m", "werewolf_harness.cli", *argv],
                              stdout=subprocess.DEVNULL).returncode
    from .. import cli
    return cli.main(argv)


def run_all(a: argparse.Namespace, specs: list[dict], seeds: list[int]) -> dict:
    """Play every missing (seed, arm), seed by seed. Returns {arm: [exit codes]}."""
    jobs = [(sp, seed, log_path(a.out, sp["arm"], seed)) for seed in seeds for sp in specs]
    jobs = [j for j in jobs if not done(j[2])]
    codes: dict[str, list[int]] = defaultdict(list)
    t0 = time.time()

    def one(j):
        sp, seed, path = j
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return sp["arm"], seed, run_game(play_argv(sp, seed, a, path), a.jobs > 1)

    if a.jobs > 1:
        with cf.ThreadPoolExecutor(a.jobs) as pool:
            for i, (arm, seed, rc) in enumerate(pool.map(one, jobs), 1):
                codes[arm].append(rc)
                print(f"   [{i}/{len(jobs)}] {arm} seed {seed} -> {'ok' if rc == 0 else f'退出码 {rc}'} "
                      f"（已用 {time.time() - t0:.0f}s）", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            arm, seed, rc = one(j)
            codes[arm].append(rc)
            print(f"   [{i}/{len(jobs)}] {arm} seed {seed} -> {'ok' if rc == 0 else f'退出码 {rc}'} "
                  f"（已用 {time.time() - t0:.0f}s）", flush=True)
    return codes


# ============================================================ per-game metrics

def price_of(model: str, prices: dict) -> tuple[float, float]:
    if model in prices:
        return prices[model]
    if model.startswith("mock-flaky"):
        return FLAKY_PRICE
    raise KeyError(model)


def tokens_by_model(log: dict) -> dict[str, list[int]]:
    """{model: [prompt, completion]} in CLI-reported tokens.

    A plain game is one model: its usage totals. A cascade game splits by the per-call
    record each turn keeps (TurnResult.cascade.calls); what usage holds beyond those calls
    (a call no TurnResult kept, e.g. one retried after a timeout) is put on the cheap model."""
    u = log.get("usage") or {}
    total = [int(u.get("gateway_prompt_tokens") or 0), int(u.get("gateway_completion_tokens") or 0)]
    if "cascade" not in log:
        return {log["model"]: total}
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for t in log.get("turns") or []:
        for c in (t.get("cascade") or {}).get("calls") or []:
            out[c["model"]][0] += int(c.get("prompt") or 0)
            out[c["model"]][1] += int(c.get("completion") or 0)
    seen = [sum(v[0] for v in out.values()), sum(v[1] for v in out.values())]
    rest = [max(0, total[0] - seen[0]), max(0, total[1] - seen[1])]
    if any(rest):
        out[log["model"]][0] += rest[0]
        out[log["model"]][1] += rest[1]
    return dict(out)


def usd(tok: dict[str, list[int]], prices: dict) -> float:
    return sum(p * price_of(m, prices)[0] / 1e6 + c * price_of(m, prices)[1] / 1e6 for m, (p, c) in tok.items())


_GUARD_BLOCKS = {"say_do_consistency", "directive_compliance", "evidence", "forensics", "unsupported_citation"}


def turn_flags(t: dict) -> dict:
    """0/1 facts about one turn, as recorded in its TurnResult."""
    r = t.get("recovery") or {}
    trace = [s for s in t.get("react_trace") or [] if not str(s.get("action", "")).endswith("(default)")]
    c = t.get("cascade") or {}
    return {
        # the model's own legal terminal, inside the ReAct loop: no forced terminal, no fallback
        "legal": 0 if (r.get("fallback_used") or r.get("forced_terminal")) else 1,
        "fallback": 1 if r.get("fallback_used") else 0,
        "clean": 0 if any(s.get("guard_blocked") for s in trace) else 1,
        "guard_block": 1 if any(s.get("block_reason") in _GUARD_BLOCKS for s in trace) else 0,
        "escalated": 1 if c.get("escalations") else 0,
        "strong": 1 if c.get("tier") == "strong" else 0,
        "reasons": [e["reason"] for e in c.get("escalations") or []],
    }


def game_row(path: str, prices: dict, cheap: str) -> dict:
    """Everything one game contributes, keyed for pairing by seed."""
    with open(path, encoding="utf-8") as fh:
        log = json.load(fh)
    g = eb.load_game(path)
    flags = [turn_flags(e.turn) for e in g.entries]
    steps = [s for e in g.entries for s in e.turn.get("react_trace") or [] if not str(s.get("action", "")).endswith("(default)")]
    tok = tokens_by_model(log)
    cost = usd(tok, prices)
    cheap_blend = blended(cheap, prices, tok)
    sd = eb.say_do_metrics([g])["consistent"]["any"]
    hal = eb.hallucination_metrics([g])["speeches_with_unsupported_citation"]
    lat = eb.latency_metrics([g])
    calls = [c for e in g.entries for c in (e.turn.get("cascade") or {}).get("calls") or []]
    return {
        "seed": g.seed, "path": path, "model": g.model, "winner": g.winner, "turns": len(g.entries),
        "tokens_cli": sum(p + c for p, c in tok.values()), "tokens_by_model": tok, "usd": cost,
        "cheap_eq_tokens": cost / cheap_blend * 1e6 if cheap_blend else None,
        "legal": mean01(f["legal"] for f in flags), "fallback": mean01(f["fallback"] for f in flags),
        "clean": mean01(f["clean"] for f in flags), "guard_block": mean01(f["guard_block"] for f in flags),
        "step_blocked": mean01(1 if s.get("guard_blocked") else 0 for s in steps),
        "escalated": mean01(f["escalated"] for f in flags), "strong_turns": mean01(f["strong"] for f in flags),
        "strong_calls": (sum(1 for c in calls if c.get("tier") == "strong") / len(calls)) if calls else None,
        "reasons": dict(Counter(r for f in flags for r in f["reasons"])),
        "say_do": sd["p"], "say_do_n": sd["n"],
        "hallucinated": hal["p"], "village_win": 1 if g.winner == "village" else 0,
        "duration_s": g.duration_s, "call_ms_p50": lat["call_ms"]["p50"]["est"],
        "turn_wall_p95": lat["turn_wall_s"]["p95"]["est"],
        "_flags": flags, "_game": g,
    }


def mean01(xs) -> float | None:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def blended(cheap: str, prices: dict, tok: dict) -> float:
    """The cheap model's USD per million tokens at this game's input/output mix."""
    p, c = sum(v[0] for v in tok.values()), sum(v[1] for v in tok.values())
    if p + c == 0:
        return 0.0
    pin, pout = price_of(cheap, prices)
    return (p * pin + c * pout) / (p + c)


# ============================================================ paired statistics

def paired_diff(a: dict[int, float | None], b: dict[int, float | None], *, bsamples: int = eb.BOOTSTRAP_B,
                seed: int = 0, alpha: float = 0.05) -> dict:
    """Mean of per-seed differences a - b over the seeds both have, with a percentile bootstrap
    over seeds (the pair is the unit: a seed is drawn with both its games).

    Returns {"n", "diff", "lo", "hi"}; no interval below eval_baseline.MIN_GAMES_FOR_CI pairs."""
    ds = [a[s] - b[s] for s in sorted(set(a) & set(b)) if a[s] is not None and b[s] is not None]
    out = {"n": len(ds), "diff": (sum(ds) / len(ds)) if ds else None, "lo": None, "hi": None}
    if len(ds) < eb.MIN_GAMES_FOR_CI:
        return out
    rng = random.Random(seed)
    k = len(ds)
    means = sorted(sum(ds[rng.randrange(k)] for _ in range(k)) / k for _ in range(bsamples))
    out["lo"], out["hi"] = eb.percentile(means, 100 * alpha / 2), eb.percentile(means, 100 * (1 - alpha / 2))
    return out


def paired_ratio(a: dict[int, float], b: dict[int, float], *, bsamples: int = eb.BOOTSTRAP_B, seed: int = 0,
                 alpha: float = 0.05) -> dict:
    """sum(a)/sum(b) over common seeds (a ratio of means, e.g. cascade cost / strong cost),
    with the same paired bootstrap."""
    seeds = [s for s in sorted(set(a) & set(b)) if a[s] is not None and b[s]]
    out = {"n": len(seeds), "ratio": None, "lo": None, "hi": None}
    if not seeds:
        return out
    out["ratio"] = sum(a[s] for s in seeds) / sum(b[s] for s in seeds)
    if len(seeds) < eb.MIN_GAMES_FOR_CI:
        return out
    rng = random.Random(seed)
    k = len(seeds)
    rs = []
    for _ in range(bsamples):
        pick = [seeds[rng.randrange(k)] for _ in range(k)]
        rs.append(sum(a[s] for s in pick) / sum(b[s] for s in pick))
    rs.sort()
    out["lo"], out["hi"] = eb.percentile(rs, 100 * alpha / 2), eb.percentile(rs, 100 * (1 - alpha / 2))
    return out


def non_inferiority(d: dict, delta: float, *, higher_is_better: bool = True) -> dict:
    """"arm is not worse than the reference by more than delta": the CI of (arm - ref) must lie
    entirely above -delta (for a rate where higher is better; below +delta otherwise)."""
    if d.get("lo") is None:
        return {"verdict": "insufficient", "delta": delta}
    ok = d["lo"] > -delta if higher_is_better else d["hi"] < delta
    return {"verdict": "non_inferior" if ok else "not_shown", "delta": delta}


# ============================================================ the analysis

PAIRED_METRICS = (   # key, label, higher is better (None = neither), percent?
    ("usd", "成本（USD，按标价）", False, False),
    ("cheap_eq_tokens", "成本（折合便宜模型 token）", False, False),
    ("tokens_cli", "CLI token / 局", False, False),
    ("legal", "合法动作率（步数内自己给出）", True, True),
    ("clean", "一次过率（回合内没被拦过）", True, True),
    ("fallback", "兜底率", False, True),
    ("guard_block", "守卫拦截率（回合）", False, True),
    ("say_do", "言行一致（任一档）", True, True),
    ("hallucinated", "发言含未查证引用", False, True),
    ("village_win", "好人胜率", None, True),
    ("duration_s", "整局时长（s）", False, False),
    ("call_ms_p50", "单次调用 p50（ms）", False, False),
)


def analyse(rows: dict[str, list[dict]], *, ref: str = "strong", delta: float = DEFAULT_DELTA) -> dict:
    """Per-arm pooled metrics, escalation rates by reason, and paired differences vs `ref`."""
    arms = {}
    for arm, rs in rows.items():
        games = [r["_game"] for r in rs]
        flags = [r["_flags"] for r in rs]
        reasons = sorted({x for r in rs for x in r["reasons"]})
        rel = eb.reliability_metrics(games) if games else {}
        arms[arm] = {
            "games": len(rs), "models": sorted({r["model"] for r in rs}),
            "usd": eb.dist([[r["usd"]] for r in rs]), "cheap_eq_tokens": eb.dist([[r["cheap_eq_tokens"] or 0] for r in rs]),
            "tokens_cli": eb.dist([[r["tokens_cli"]] for r in rs]),
            "tokens_by_model": dict(sum((Counter({m: p + c for m, (p, c) in r["tokens_by_model"].items()}) for r in rs), Counter())),
            "legal": eb.prop([[f["legal"] for f in fl] for fl in flags]),
            "clean": eb.prop([[f["clean"] for f in fl] for fl in flags]),
            "fallback": eb.prop([[f["fallback"] for f in fl] for fl in flags]),
            "guard_block": eb.prop([[f["guard_block"] for f in fl] for fl in flags]),
            "step_success": rel.get("step_success"),
            "escalated": eb.prop([[f["escalated"] for f in fl] for fl in flags]),
            "strong_turns": eb.prop([[f["strong"] for f in fl] for fl in flags]),
            "escalation_by_reason": {x: eb.prop([[1 if x in f["reasons"] else 0 for f in fl] for fl in flags])
                                     for x in reasons},
            "say_do": eb.say_do_metrics(games)["consistent"]["any"] if games else None,
            "hallucinated": eb.hallucination_metrics(games)["speeches_with_unsupported_citation"] if games else None,
            "outcome": eb.outcome_metrics(games) if games else None,
            "latency": {"game_s": eb.dist([[r["duration_s"]] for r in rs]),
                        "call_ms": eb.latency_metrics(games)["call_ms"] if games else None},
        }
    paired = {}
    if ref in rows:
        by = {arm: {r["seed"]: r for r in rs} for arm, rs in rows.items()}
        for arm in rows:
            if arm == ref:
                continue
            m = {}
            for key, _label, _hib, _pct in PAIRED_METRICS:
                m[key] = paired_diff({s: r[key] for s, r in by[arm].items()}, {s: r[key] for s, r in by[ref].items()})
            m["cost_ratio_usd"] = paired_ratio({s: r["usd"] for s, r in by[arm].items()}, {s: r["usd"] for s, r in by[ref].items()})
            m["non_inferiority_legal"] = non_inferiority(m["legal"], delta)
            paired[arm] = m
    return {"arms": arms, "paired": paired, "ref": ref, "delta": delta}


# ============================================================ report

def _pct(x, nd=1):
    return "—" if x is None else f"{100 * x:.{nd}f}%"


def _num(x, nd=0):
    return "—" if x is None else f"{x:,.{nd}f}"


def _ci(d: dict, pct: bool, nd: int = 1) -> str:
    """A paired difference: "+1.2 点 [-0.3, +2.8] (n=10)" for a rate, plain numbers otherwise."""
    if d.get("diff") is None:
        return "—"
    sign = "+" if d["diff"] >= 0 else ""
    if pct:
        body = f"{sign}{100 * d['diff']:.{nd}f} 点"
        ci = f" [{100 * d['lo']:+.{nd}f}, {100 * d['hi']:+.{nd}f}]" if d.get("lo") is not None else "（对数不够，不给区间）"
    else:
        body = f"{sign}{_num(d['diff'], nd)}"
        ci = f" [{d['lo']:+,.{nd}f}, {d['hi']:+,.{nd}f}]" if d.get("lo") is not None else "（对数不够，不给区间）"
    return body + ci + f" (n={d['n']})"


def _prop(p: dict | None) -> str:
    if not p or p.get("p") is None:
        return "—"
    b = p.get("boot") or [None, None]
    ci = f" [{_pct(b[0])}, {_pct(b[1])}]" if b[0] is not None else ""
    return f"{_pct(p['p'])}{ci} ({p['k']}/{p['n']})"


def _dist(d: dict | None, nd=0) -> str:
    if not d:
        return "—"
    m = d["mean"]
    ci = f" [{_num(m['lo'], nd)}, {_num(m['hi'], nd)}]" if m.get("lo") is not None else ""
    return f"{_num(m['est'], nd)}{ci}"


def render(res: dict, meta: dict) -> str:
    arms, paired, ref, delta = res["arms"], res["paired"], res["ref"], res["delta"]
    order = [a for a in meta["arms"] if a in arms]
    L = [f"# 级联配对实验：{meta['cheap']}（便宜） / {meta['strong']}（强）", ""]
    L.append(f"种子 {meta['seeds_label']}（每个种子 = 同一副牌，每个实验臂都打一遍），板子 {meta['variant']}，并行 {meta['parallel']}，"
             f"级联方式 {meta['cascade_mode']}{'（含低置信度升级）' if meta['low_confidence'] else ''}。"
             f"比例给按局 bootstrap 区间；配对差 = 实验臂 − {ref}，按种子成对重抽样（B={eb.BOOTSTRAP_B}）。")
    L.append("价格（USD / 百万 token，输入/输出）：" +"；".join(f"{m} {p[0]:g}/{p[1]:g}" for m, p in meta["prices"].items())
             + "。来源见 scripts/paired_cascade.py 的 PRICES；cache 读写按普通输入计价，绝对值偏高，两臂之比不受影响。")
    if meta.get("mock"):
        L += ["", "> **这是 mock 跑出来的数**：便宜档是 FlakyMockClient（按种子确定地犯小模型的错），强档是普通 MockClient。"
              "它们证明流水线和统计是通的，不代表任何真实模型的质量。"]
    L += ["", "## 各臂汇总", "", "| 指标 | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    rows = [
        ("局数", lambda a: str(a["games"])),
        ("成本 USD/局", lambda a: _dist(a["usd"], 3)),
        ("折合便宜模型 token/局", lambda a: _dist(a["cheap_eq_tokens"])),
        ("CLI token/局", lambda a: _dist(a["tokens_cli"])),
        ("合法动作率（回合）", lambda a: _prop(a["legal"])),
        ("一次过率（回合）", lambda a: _prop(a["clean"])),
        ("兜底率（回合）", lambda a: _prop(a["fallback"])),
        ("守卫拦截率（回合）", lambda a: _prop(a["guard_block"])),
        ("ReAct 步成功率", lambda a: _prop(a["step_success"])),
        ("升级的回合", lambda a: _prop(a["escalated"])),
        ("由强模型作答的回合", lambda a: _prop(a["strong_turns"])),
        ("言行一致（任一档）", lambda a: _prop(a["say_do"])),
        ("发言含未查证引用", lambda a: _prop(a["hallucinated"])),
        ("好人胜率", lambda a: f"{_pct((a['outcome'] or {}).get('village_win_rate'))} ({(a['outcome'] or {}).get('village_wins')}/{(a['outcome'] or {}).get('n')})"),
        ("整局时长 s", lambda a: _dist(a["latency"]["game_s"], 1)),
        ("单次调用 p50 ms", lambda a: _num(((a["latency"]["call_ms"] or {}).get("p50") or {}).get("est"))),
    ]
    for label, fn in rows:
        L.append(f"| {label} | " + " | ".join(fn(arms[a]) for a in order) + " |")
    esc = [a for a in order if arms[a]["escalation_by_reason"]]
    if esc:
        L += ["", "## 升级原因（占该臂回合数）", ""]
        for a in esc:
            L.append(f"- **{a}**：" + "；".join(f"{k} {_prop(v)}" for k, v in sorted(arms[a]["escalation_by_reason"].items(),
                                                                                    key=lambda kv: -kv[1]["k"])))
    for a in order:
        if len(arms[a]["tokens_by_model"]) > 1:
            tot = sum(arms[a]["tokens_by_model"].values())
            L.append(f"- **{a}** 的 token 按模型：" + "；".join(f"{m} {v:,}（{_pct(v / tot)}）" for m, v in arms[a]["tokens_by_model"].items()))
    if paired:
        L += ["", f"## 配对差（实验臂 − {ref}，同种子成对）", "", "| 指标 | " + " | ".join(paired) + " |", "|---|" + "---|" * len(paired)]
        for key, label, _hib, pct in PAIRED_METRICS:
            L.append(f"| {label} | " + " | ".join(_ci(paired[a][key], pct, 1 if pct else (3 if key == 'usd' else 0))
                                                  for a in paired) + " |")
        L.append("| 成本之比（USD，实验臂/强） | " + " | ".join(
            (f"{p['cost_ratio_usd']['ratio']:.2f}" + (f" [{p['cost_ratio_usd']['lo']:.2f}, {p['cost_ratio_usd']['hi']:.2f}]"
                                                      if p['cost_ratio_usd']['lo'] is not None else ""))
            if p["cost_ratio_usd"]["ratio"] is not None else "—" for p in paired.values()) + " |")
        L += ["", "## 非劣效结论（合法动作率）", ""]
        for a, p in paired.items():
            d, ni = p["legal"], p["non_inferiority_legal"]
            if ni["verdict"] == "insufficient":
                L.append(f"- **{a}**：配对数 {d['n']} 太少，不下结论。")
            elif ni["verdict"] == "non_inferior":
                L.append(f"- **{a} 的合法动作率不比 {ref} 差超过 {100 * delta:g} 个百分点**：差 {_ci(d, True)}，"
                         f"区间下界 {100 * d['lo']:+.1f} 点 > −{100 * delta:g} 点。")
            else:
                L.append(f"- **{a}：不能说不劣于 {ref}**（容差 {100 * delta:g} 点）：差 {_ci(d, True)}，"
                         f"区间下界 {100 * d['lo']:+.1f} 点 ≤ −{100 * delta:g} 点。")
    return "\n".join(L) + "\n"


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def collect(a: argparse.Namespace, specs: list[dict], seeds: list[int], prices: dict) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for sp in specs:
        rs = [game_row(log_path(a.out, sp["arm"], s), prices, a.cheap) for s in seeds if done(log_path(a.out, sp["arm"], s))]
        if rs:
            rows[sp["arm"]] = rs
    return rows


def parse_prices(items: list[str]) -> dict:
    prices = dict(PRICES)
    for it in items:
        name, _, val = it.partition("=")
        pin, _, pout = val.partition(",")
        prices[name] = (float(pin), float(pout or pin))
    return prices


def parse_estimates(items: list[str]) -> dict:
    est = dict(ESTIMATES)
    for it in items:
        name, _, val = it.partition("=")
        mins, _, tok = val.partition(":")
        est[name] = (float(mins), int(float(tok or 560_000)))
    return est


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="级联配对实验：同一批种子，强 / 便宜 / 级联 各打一遍，按种子配对比较")
    ap.add_argument("--seeds", required=True, help="2001-2010，或 1,2,5，或混写")
    ap.add_argument("--cheap", required=True)
    ap.add_argument("--strong", required=True)
    ap.add_argument("--arms", default="strong,cheap,cascade", help="strong,cheap,cascade[-on-fail|-decisive|-both]")
    ap.add_argument("--cascade-mode", default="both", choices=MODES, dest="cascade_mode", help="arm 'cascade' 用哪种级联")
    ap.add_argument("--decisive", default=None, help="关键回合（传给 play --decisive）")
    ap.add_argument("--low-confidence", action="store_true", dest="low_confidence")
    ap.add_argument("--variant", default="8p")
    ap.add_argument("--parallel", type=int, default=8, help="每局内同时想的座位数（play --parallel）")
    ap.add_argument("--guard", default="L1,L2,L3")
    ap.add_argument("--jobs", type=int, default=1, help="同时跑几局（子进程）；真实模型建议 1")
    ap.add_argument("--record", action="store_true", help="每局都录像（.cassette.jsonl），之后可 replay-llm")
    ap.add_argument("--out", default="runs/cascade/")
    ap.add_argument("--price", action="append", default=[], metavar="MODEL=IN,OUT", help="USD/百万 token，覆盖默认价格")
    ap.add_argument("--est", action="append", default=[], metavar="MODEL=MIN[:TOKENS]", help="每局耗时/token 估计")
    ap.add_argument("--delta", type=float, default=DEFAULT_DELTA, help="非劣效容差（合法动作率，比例）")
    ap.add_argument("--analyse-only", action="store_true", dest="analyse_only", help="不跑新局，只分析已有日志")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run", help="只打印计划和估计")
    a = ap.parse_args(argv)
    seeds = parse_seeds(a.seeds)
    arm_names = [x.strip() for x in a.arms.split(",") if x.strip()]
    specs = [arm_spec(x, a.cheap, a.strong, a.cascade_mode) for x in arm_names]
    prices = parse_prices(a.price)
    for sp in specs:
        for m in filter(None, (sp.get("model"), sp.get("strong"))):
            try:
                price_of(m, prices)
            except KeyError:
                raise SystemExit(f"!! 没有 {m} 的价格：用 --price {m}=输入,输出（USD/百万 token）") from None
    todo = {sp["arm"]: sum(1 for s in seeds if not done(log_path(a.out, sp["arm"], s))) for sp in specs}
    print(f"== 配对实验：{len(seeds)} 个种子 × {len(specs)} 臂 = {len(seeds) * len(specs)} 局"
          f"（已完成 {len(seeds) * len(specs) - sum(todo.values())} 局，续跑会跳过）→ {a.out}")
    for line in estimate(specs, seeds, todo, parse_estimates(a.est)):
        print(line)
    if a.dry_run:
        return {}
    if not a.analyse_only and sum(todo.values()):
        codes = run_all(a, specs, seeds)
        bad = {arm: sum(1 for c in cs if c) for arm, cs in codes.items() if any(cs)}
        if bad:
            print(f"!! 有局没跑完（退出码非 0）：{bad}——再跑同一条命令会重试这些局")
    rows = collect(a, specs, seeds, prices)
    if not rows:
        raise SystemExit("没有可分析的日志")
    res = analyse(rows, ref="strong", delta=a.delta)
    meta = {"cheap": a.cheap, "strong": a.strong, "arms": arm_names, "variant": a.variant, "parallel": a.parallel,
            "cascade_mode": a.cascade_mode, "low_confidence": a.low_confidence, "seeds": seeds,
            "seeds_label": f"{seeds[0]}–{seeds[-1]}（{len(seeds)} 个）",
            "prices": {m: prices[m] if m in prices else FLAKY_PRICE for m in {a.cheap, a.strong}},
            "mock": a.cheap.startswith("mock") or a.strong.startswith("mock")}
    report = render(res, meta)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(eb._jsonable(_strip({"meta": meta, **res, "games": rows})), fh, ensure_ascii=False, indent=1)
    print(report)
    return res


if __name__ == "__main__":
    main()
