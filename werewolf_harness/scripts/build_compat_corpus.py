"""Build tests/corpus/: the stored journals CI replays under their recorded engine version.

Two kinds of file, both in the compat_check input format:

  real-*   the journals of the real-model logs in /home/claude/data/runs/*/ (played
           before engine versions existed, so they carry no engine_version and are
           read as the baseline). Only what replay needs is kept: variant, playbook,
           seed, outcome, and the journal with each action's private ``thought``
           dropped -- the engine never reads it, and it is most of the bytes.
  mock-*   small mock-model games (MockClient seats) across every variant, played
           under every supported engine version, so each rule branch -- the old one
           and the new one -- has journals that pin it.

Every file also stores ``fingerprints``: the per-step public-state hash under its own
version, as computed when the file was written. A later change to an old branch that
still accepts the journal and still ends with the same winner, but changes the world
on the way (a different death, a different ballot count), fails on those.

    python -m werewolf_harness.scripts.build_compat_corpus [--real '/home/claude/data/runs/*/*.json']

Re-run only to ADD games (e.g. after introducing a new version). Regenerating the
files of an existing version with changed code defeats the point of the corpus.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib

from ..arena import run
from ..envs import ENGINE_VERSION, MIN_ENGINE_VERSION, VARIANTS, Werewolf
from ..players import LLMPlayer, MockClient
from .compat_check import game_from_log, replay_under

CORPUS = pathlib.Path(__file__).resolve().parents[1] / "tests" / "corpus"

# (variant, playbook, seed): small, varied, cheap. 12p/15p with the pack's night channel.
MOCK_GAMES = [("8p", "static", s) for s in (1, 2, 3, 4, 5, 6)] + \
             [("8p+sheriff", "static", s) for s in (1, 2, 3, 4)] + \
             [("12p", "static", s) for s in (1, 2, 3, 4)] + \
             [("12p", "none", s) for s in (5, 6)] + \
             [("15p", "dynamic", s) for s in (1, 2)] + \
             [("8p", "none", s) for s in (7, 8)]


def _strip(action):
    if isinstance(action, dict) and isinstance(action.get("args"), dict):
        return {**action, "args": {k: v for k, v in action["args"].items() if k != "thought"}}
    return action


def _write(name: str, body: dict) -> int:
    game = game_from_log(body, name)
    rep = replay_under(game, game.engine_version)
    assert rep.status == "ok", f"{name} does not replay under v{game.engine_version}: {rep.got}"
    body["fingerprints"] = rep.hashes
    path = CORPUS / f"{name}.json"
    path.write_text(json.dumps(body, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path.stat().st_size


def build_real(pattern: str) -> int:
    total = 0
    for p in sorted(glob.glob(pattern)):
        log = json.load(open(p, encoding="utf-8"))
        run_dir = os.path.basename(os.path.dirname(p))
        body = {"source": f"{run_dir}/{os.path.basename(p)}", "variant": log["variant"], "seed": log["seed"],
                "playbook": log.get("playbook", "static"), "model": log.get("model"), "outcome": log["outcome"],
                "journal": [{"actor": r["actor"], "task": r["task"], "action": _strip(r["action"]),
                             "forced": bool(r.get("forced"))} for r in log["journal"]]}
        if "engine_version" in log:
            body["engine_version"] = log["engine_version"]
        total += _write(f"real-{run_dir}-{os.path.splitext(os.path.basename(p))[0]}", body)
    return total


def build_mock(versions: list[int]) -> int:
    total = 0
    for v in versions:
        for variant, playbook, seed in MOCK_GAMES:
            env = Werewolf(variant, playbook_mode=playbook, engine_version=v)
            n = VARIANTS[variant].num_players
            players = {s: LLMPlayer(MockClient(seed=seed * 100 + s), playbook=playbook != "none")
                       for s in range(1, n + 1)}
            ep = run(env, players, seed)
            body = {"source": "mock", "engine_version": v, "variant": variant, "seed": seed, "playbook": playbook,
                    "outcome": ep.outcome,
                    "journal": [{"actor": r.actor, "task": r.task, "action": _strip(r.action), "forced": r.forced}
                                for r in ep.journal]}
            total += _write(f"mock-v{v}-{variant.replace('+', '_')}-{playbook}-s{seed}", body)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="/home/claude/data/runs/*/*.json")
    ap.add_argument("--versions", default=",".join(str(v) for v in range(MIN_ENGINE_VERSION, ENGINE_VERSION + 1)))
    a = ap.parse_args()
    CORPUS.mkdir(parents=True, exist_ok=True)
    real = build_real(a.real) if a.real else 0
    mock = build_mock([int(x) for x in a.versions.split(",") if x])
    print(f"corpus: {CORPUS}  real {real / 1e3:.0f} kB, mock {mock / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
