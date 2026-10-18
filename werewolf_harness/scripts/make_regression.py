"""Turn a recorded game into a regression case.

    python -m werewolf_harness.scripts.make_regression runs/rec/g1001.json --name wolf_votes_itself \\
        [--upto 40] [--expect not_forced:37 --expect action:37:vote] [--allow-drift]

The game is replayed first (through the real LLMPlayer stack, on its cassette); a game
that does not replay cannot be a case. The case keeps the journal and per-turn records
of that replay, and only the calls it used -- with --upto N, a long game shrinks to the
part that matters. tests/test_regressions.py replays every case in the folder and checks
the journal, the turn records and the expectations (see replay_llm.check_expectations).

After a deliberate prompt change every case reports a request mismatch, naming the call
and the first differing message. Either regenerate the case from a new recording, or
mark it --allow-drift if the recorded replies still decide the same game.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

from ..replay_llm import check_expectations, compare, load, replay_game

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEST = os.path.normpath(os.path.join(HERE, "..", "tests", "regressions"))
CASSETTE = "cassette.jsonl.gz"
GAME = "game.json.gz"


def make_case(game: str, name: str, *, upto: int | None = None, expect: list[str] | None = None,
              allow_drift: bool = False, dest: str = DEFAULT_DEST, note: str = "") -> tuple[str, list[str]]:
    """Write the case; return its folder and the expectations that do not hold yet (a case
    may be written to pin a bug before it is fixed -- it then fails until it is)."""
    log, cas = load(game)
    rep = replay_game(log, cas, allow_drift=allow_drift, upto=upto)
    diffs = compare(log, rep, upto=upto)
    if diffs:
        raise SystemExit("这一局回放不出来，不能做成回归用例：\n  " + "\n  ".join(diffs))
    folder = os.path.join(dest, name)
    os.makedirs(folder, exist_ok=True)
    case_log = {k: log[k] for k in ("variant", "seed", "model", "guard", "playbook", "parallel", "settings",
                                    "engine_version") if k in log}
    case_log.update({"journal": rep.journal, "outcome": rep.outcome, "turns": rep.turns, "cassette": CASSETTE,
                     "duration_s": log.get("duration_s")})
    with gzip.open(os.path.join(folder, GAME), "wt", encoding="utf-8") as f:
        json.dump(case_log, f, ensure_ascii=False, indent=1)
    cas.write_subset(os.path.join(folder, CASSETTE), rep.used)
    case = {"name": name, "source": os.path.basename(game), "upto": upto, "expect": list(expect or []),
            "allow_drift": allow_drift, "note": note, "calls": len(rep.used), "steps": len(rep.journal),
            "created": time.strftime("%Y-%m-%d")}
    with open(os.path.join(folder, "case.json"), "w", encoding="utf-8") as f:
        json.dump(case, f, ensure_ascii=False, indent=1)
    return folder, check_expectations(rep, case["expect"])


def run_case(folder: str) -> list[str]:
    """Replay one case; [] when it still plays the same game and meets its expectations."""
    case = json.load(open(os.path.join(folder, "case.json"), encoding="utf-8"))
    log, cas = load(os.path.join(folder, GAME))
    # the case's journal already stops at `upto`, and its outcome says so: replay to the same place
    rep = replay_game(log, cas, allow_drift=case.get("allow_drift", False), upto=case.get("upto"))
    return compare(log, rep) + check_expectations(rep, case.get("expect") or [])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="make_regression")
    p.add_argument("game", help="play --record 写出来的日志（录像在它旁边）")
    p.add_argument("--name", required=True)
    p.add_argument("--upto", type=int, default=None, help="只保留前 N 步")
    p.add_argument("--expect", action="append", default=[], help="例如 not_forced:37、action:37:vote、winner:village")
    p.add_argument("--allow-drift", action="store_true", dest="allow_drift")
    p.add_argument("--note", default="", help="一句话：这个用例防的是什么")
    p.add_argument("--dest", default=DEFAULT_DEST)
    a = p.parse_args(argv)
    folder, failing = make_case(a.game, a.name, upto=a.upto, expect=a.expect, allow_drift=a.allow_drift,
                                dest=a.dest, note=a.note)
    size = sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder))
    print(f"用例写好了：{folder}（{size // 1024} KB）")
    for f in failing:
        print(f"   !! 期望现在不成立（修好之前这个用例会失败）：{f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
