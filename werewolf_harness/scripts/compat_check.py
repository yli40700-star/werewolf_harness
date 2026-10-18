"""Engine compatibility check: does every stored game still replay, and would it under today's rules?

world = f(seed, journal) holds only for a fixed engine. Each stored game is replayed
twice, from nothing but (variant, playbook, seed, journal):

  recorded   under the engine version the game was created with -- what recovery does.
             This MUST reproduce the game; if it does not, an "old" rule branch was
             changed by accident and live games will recover into a different world.
  current    under the current version (or ``--against V``) -- what would happen if the
             game were (wrongly) replayed with today's rules. A divergence here is not a
             bug; it is the list of games a rule change affects, and the step it bites.

Per game it reports identical / diverged at step N (the journal row, what the engine
asked for instead, and the first public state that differs) / crashed. With
``--fingerprint`` the per-step public state (Werewolf.public_state) is hashed and
compared too, so a change that alters the world without moving a turn -- same journal
accepted, same winner, different deaths along the way -- is caught; a corpus file that
carries ``fingerprints`` (tests/corpus/) is also checked against them.

    python -m werewolf_harness.scripts.compat_check '/home/claude/data/runs/*/*.json'
    python -m werewolf_harness.scripts.compat_check werewolf_harness/tests/corpus/*.json --fingerprint
    python -m werewolf_harness.scripts.compat_check --db [--limit 500]      # every game in Postgres
    ... --json report.json      machine-readable rows;   --strict   also fail on current-version divergence

Exit status: 1 if any game fails under its recorded version (or, with --strict, under
the current one), else 0.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any

from ..arena import Record, Terminal, Turn
from ..envs import ENGINE_VERSION, RULE_CHANGES, Rules, UnsupportedEngineVersion, Werewolf, engine_version_of


# ------------------------------------------------------------------ inputs
@dataclass
class StoredGame:
    id: str
    variant: str
    playbook: str
    seed: int
    engine_version: int
    journal: list[Record]
    outcome: dict | None = None            # None: unfinished (a live game) or unknown
    fingerprints: list[str] | None = None  # per-step public-state hashes, when the file carries them


def _norm(x: Any) -> Any:
    """JSON round trip: int dict keys become strings, so a stored and a live value compare."""
    return json.loads(json.dumps(x, ensure_ascii=False, default=str))


def game_from_log(log: dict, gid: str) -> StoredGame:
    """A ``cli play --out`` log, a corpus file, or a server ``full_log`` save (``game`` block)."""
    g = log.get("game") or {}
    src = {**g, **{k: v for k, v in log.items() if k != "game"}} if g else log
    journal = [Record(int(r["actor"]), r["task"], r["action"], bool(r.get("forced"))) for r in log.get("journal") or []]
    return StoredGame(id=gid, variant=src["variant"], playbook=src.get("playbook", "static"), seed=int(src["seed"]),
                      engine_version=engine_version_of(src), journal=journal,
                      outcome=src.get("outcome") or g.get("outcome"), fingerprints=log.get("fingerprints"))


def load_logs(patterns: list[str]) -> list[StoredGame]:
    paths = sorted({p for pat in patterns for p in (glob.glob(pat) or ([pat] if os.path.isfile(pat) else []))})
    root = os.path.commonpath([os.path.dirname(os.path.abspath(p)) for p in paths]) if paths else ""
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            log = json.load(f)
        if not log.get("journal"):
            continue
        out.append(game_from_log(log, os.path.relpath(os.path.abspath(p), root) if root else p))
    return out


async def load_db(url: str | None = None, limit: int = 500) -> list[StoredGame]:
    from ..app.config import get_settings
    from ..app.storage import Store
    store = Store.connect(url or get_settings().database_url, pool_size=2)
    try:
        out = []
        for g in await store.recent_games(limit):
            rows = await store.journal(g["id"])
            out.append(StoredGame(id=g["id"], variant=g["variant"], playbook=(g["config"] or {}).get("playbook", "static"),
                                  seed=int(g["seed"]), engine_version=engine_version_of(g),
                                  journal=[Record(r.actor, r.task, r.action, r.forced) for r in rows],
                                  outcome=g["outcome"] if g["state"] == "over" else None))
        return out
    finally:
        await store.close()


# ------------------------------------------------------------------ one replay
def state_hash(state: dict) -> str:
    body = json.dumps(state, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(body.encode()).hexdigest()[:12]


@dataclass
class Replay:
    """One replay of one journal under one engine version."""
    version: int
    status: str = "ok"                    # ok | diverged | crashed
    step: int | None = None               # journal index where it went wrong
    expected: str | None = None           # what the journal says happens at `step`
    got: str | None = None                # what this engine did instead
    end: str | None = None                # repr of the turn/terminal after the last row
    outcome: dict | None = None
    states: list[dict] = field(default_factory=list)    # public state after reset and after each row
    worlds: list[str] = field(default_factory=list)     # hash of the full private world, same steps

    @property
    def hashes(self) -> list[str]:
        return [state_hash(s) for s in self.states]


def _turn_str(t: Turn | Terminal | None) -> str:
    if isinstance(t, Turn):
        return f"{t.actor} 号 / {t.task}"
    if isinstance(t, Terminal):
        return f"终局 winner={t.outcome.get('winner')}"
    return repr(t)


def replay_under(game: StoredGame, version: int) -> Replay:
    """Feed the journal to the engine at ``version``; stop at the first row it will not take."""
    r = Replay(version)
    try:
        env = Werewolf(game.variant, playbook_mode=game.playbook, engine_version=version)
        turn = env.reset(game.seed)
    except UnsupportedEngineVersion as exc:
        r.status, r.step, r.got = "crashed", 0, f"不支持的引擎版本：{exc}"
        return r
    except Exception as exc:                                     # noqa: BLE001 -- a report, not a raise
        r.status, r.step, r.got = "crashed", 0, f"{type(exc).__name__}: {exc}"
        return r

    def snap():
        r.states.append(_norm(env.public_state()))
        r.worlds.append(state_hash(env.world_state()))
    snap()
    for i, rec in enumerate(game.journal):
        if not isinstance(turn, Turn) or (turn.actor, turn.task) != (rec.actor, rec.task):
            r.status, r.step = "diverged", i
            r.expected, r.got = f"{rec.actor} 号 / {rec.task}", _turn_str(turn)
            r.end = _turn_str(turn)
            return r
        try:
            turn = env.step(rec.actor, rec.task, rec.action)
        except Exception as exc:                                 # noqa: BLE001
            r.step, r.expected = i, f"{rec.actor} 号 / {rec.task}: {_action_str(rec.action)}"
            if isinstance(exc, RuntimeError) and "illegal action" in str(exc):
                # the turn is the same but this engine no longer allows what was done in it
                r.status, r.got = "diverged", f"动作不合法：{str(exc).split(': ', 1)[-1]}"
                return r
            r.status = "crashed"
            r.expected = f"{rec.actor} 号 / {rec.task}: {_action_str(rec.action)}"
            r.got = f"{type(exc).__name__}: {exc} @ {traceback.extract_tb(exc.__traceback__)[-1].lineno}"
            return r
        snap()
    r.end = _turn_str(turn)
    r.outcome = _norm(turn.outcome) if isinstance(turn, Terminal) else None
    if game.outcome is not None and r.outcome != _norm(game.outcome):
        r.status, r.step = "diverged", len(game.journal)
        r.expected = f"终局 winner={game.outcome.get('winner')} round={game.outcome.get('round')}"
        r.got = r.end if r.outcome is None else f"终局 winner={r.outcome.get('winner')} round={r.outcome.get('round')}"
    return r


def _action_str(action: Any) -> str:
    if isinstance(action, dict):
        args = {k: v for k, v in (action.get("args") or {}).items() if k not in ("thought", "content", "reason")}
        return f"{action.get('name')}{args or ''}"
    return str(action)[:60]


# ------------------------------------------------------------------ the comparison
@dataclass
class GameReport:
    id: str
    variant: str
    engine_version: int
    against: int
    steps: int
    recorded: str                          # identical | diverged | crashed (under the recorded version)
    current: str                           # identical | diverged | crashed (under `against`)
    detail: dict = field(default_factory=dict)

    @property
    def broken(self) -> bool:
        return self.recorded != "identical"


def _first_state_diff(a: list[dict], b: list[dict]) -> dict | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            keys = sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))
            return {"after_step": i, "keys": keys,
                    "recorded": {k: _short(x.get(k)) for k in keys}, "current": {k: _short(y.get(k)) for k in keys}}
    return None


def _short(v: Any, n: int = 120) -> Any:
    s = json.dumps(v, ensure_ascii=False, default=str)
    return v if len(s) <= n else s[:n] + "…"


def _classify(rep: Replay, base: Replay | None, fingerprint: bool, stored: list[str] | None) -> tuple[str, dict]:
    """``base`` is the recorded-version replay the current one is measured against (None: this IS it)."""
    detail: dict = {}
    if rep.status != "ok":
        detail = {"step": rep.step, "expected": rep.expected, "got": rep.got}
        status = rep.status
    else:
        status = "identical"
    if base is not None and status == "identical" and rep.end != base.end:
        status, detail = "diverged", {"step": len(rep.states) - 1, "expected": base.end, "got": rep.end}
    if fingerprint and status == "identical":
        ref = base.hashes if base is not None else stored
        if ref is not None:
            mine = rep.hashes
            bad = next((i for i, (x, y) in enumerate(zip(ref, mine, strict=False)) if x != y), None)
            if bad is None and len(ref) != len(mine):
                bad = min(len(ref), len(mine))
            if bad is not None:
                status = "diverged"
                detail = {"step": max(0, bad - 1), "expected": f"公开状态 {ref[bad] if bad < len(ref) else '-'}",
                          "got": f"公开状态 {mine[bad] if bad < len(mine) else '-'}", "by": "fingerprint"}
        if status == "identical" and base is not None and base.worlds != rep.worlds:
            bad = next((i for i, (x, y) in enumerate(zip(base.worlds, rep.worlds, strict=False)) if x != y), 0)
            status, detail = "diverged", {"step": max(0, bad - 1), "expected": "同一个世界",
                                          "got": "公开状态相同，隐藏状态不同", "by": "world"}
    if base is not None and status != "identical":
        diff = _first_state_diff(base.states, rep.states)
        if diff:
            detail["first_state_diff"] = diff
    return status, detail


def check_game(game: StoredGame, against: int = ENGINE_VERSION, fingerprint: bool = False) -> GameReport:
    rec = replay_under(game, game.engine_version)
    cur = rec if against == game.engine_version else replay_under(game, against)
    rec_status, rec_detail = _classify(rec, None, fingerprint, game.fingerprints)
    if cur is rec:
        cur_status, cur_detail = rec_status, dict(rec_detail)
    else:
        cur_status, cur_detail = _classify(cur, rec if rec.status == "ok" else None, fingerprint, None)
    detail = {}
    if rec_detail:
        detail["recorded"] = rec_detail
    if cur_detail and cur is not rec:
        detail["current"] = cur_detail
        try:
            blame = Rules.at(against).changed_since(game.engine_version) if against > game.engine_version else []
        except UnsupportedEngineVersion:
            blame = []
        if blame:
            detail["current"]["rule_changes"] = [c.flag for c in blame]
    return GameReport(game.id, game.variant, game.engine_version, against, len(game.journal),
                      rec_status, cur_status, detail)


def check(games: list[StoredGame], against: int = ENGINE_VERSION, fingerprint: bool = False) -> list[GameReport]:
    return [check_game(g, against, fingerprint) for g in games]


def summary(reports: list[GameReport]) -> dict:
    def count(attr):
        out = {"identical": 0, "diverged": 0, "crashed": 0}
        for r in reports:
            out[getattr(r, attr)] += 1
        return out
    return {"games": len(reports), "recorded": count("recorded"), "current": count("current"),
            "by_version": {v: sum(1 for r in reports if r.engine_version == v) for v in sorted({r.engine_version for r in reports})}}


# ------------------------------------------------------------------ output
_LABEL = {"identical": "一致", "diverged": "分叉", "crashed": "崩溃"}


def _line(d: dict) -> str:
    s = f"第 {d['step']} 步：期望 {d['expected']}，实际 {d['got']}"
    diff = d.get("first_state_diff")
    if diff:
        s += f"；首个不同的公开状态：执行完前 {diff['after_step']} 步之后的 {', '.join(diff['keys'])}"
    if d.get("rule_changes"):
        s += f"；相关规则变更：{', '.join(d['rule_changes'])}"
    return s


def render(reports: list[GameReport], against: int) -> str:
    w = max([len(r.id) for r in reports] + [4])
    out = [f"{'游戏':<{w}}  记录版本  步数  记录版本回放  当前 v{against} 回放"]
    for r in reports:
        out.append(f"{r.id:<{w}}  v{r.engine_version:<7} {r.steps:>4}  {_LABEL[r.recorded]:<10}  {_LABEL[r.current]}")
        for side in ("recorded", "current"):
            if side in r.detail:
                out.append(f"{'':<{w}}    └ {'记录版本' if side == 'recorded' else '当前版本'}：{_line(r.detail[side])}")
    s = summary(reports)
    fmt = lambda c: " / ".join(f"{_LABEL[k]} {v}" for k, v in c.items())       # noqa: E731
    out += ["", f"汇总：{s['games']} 局（按记录版本：" + "，".join(f"v{v} {n} 局" for v, n in s["by_version"].items()) + "）",
            f"  记录版本回放：{fmt(s['recorded'])}" + ("" if not s["recorded"]["diverged"] + s["recorded"]["crashed"]
                                                     else "   !! 有局在自己的版本下都回放不出来：旧规则分支被改动了"),
            f"  当前 v{against} 回放：{fmt(s['current'])}",
            "  规则变更：" + "；".join(f"v{c.introduced} {c.flag}" for c in RULE_CHANGES)]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="compat_check", description=__doc__.split("\n")[0])
    ap.add_argument("logs", nargs="*", help="game logs or globs (cli play --out, corpus files, server saves)")
    ap.add_argument("--db", nargs="?", const="", default=None, metavar="URL",
                    help="also check every game in Postgres (URL defaults to DATABASE_URL)")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--against", type=int, default=ENGINE_VERSION, help="the version to compare with (default: current)")
    ap.add_argument("--fingerprint", action="store_true", help="also compare the per-step public state hash")
    ap.add_argument("--strict", action="store_true", help="fail when a game diverges under --against too")
    ap.add_argument("--json", default=None, help="write the per-game rows here")
    a = ap.parse_args(argv)
    games = load_logs(a.logs) if a.logs else []
    if a.db is not None:
        games += asyncio.run(load_db(a.db or None, a.limit))
    if not games:
        print("没有找到可检查的对局（日志里没有 journal？）", file=sys.stderr)
        return 2
    reports = check(games, a.against, a.fingerprint)
    print(render(reports, a.against))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary(reports), "games": [asdict(r) for r in reports]}, f, ensure_ascii=False, indent=1)
    bad = any(r.broken for r in reports) or (a.strict and any(r.current != "identical" for r in reports))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
