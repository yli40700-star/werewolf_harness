"""What model strength and a few rule tweaks do to the game itself (2026-10-16).

Reads `cli play` logs, one directory per arm, and reports per arm:

  village win rate                 Wilson 95% CI
  rounds played
  village day-1 vote accuracy      share of village ballots on day 1 that hit a wolf
  village vote accuracy            the same over every day
  wolves exiled on day 1           games where the day-1 exile was a wolf
  wolf seer claims                 games where a wolf claimed seer (悍跳), and how the
                                   village voted on the day of the claim
  persuasion                       after a speech that names a target (intent / accuse),
                                   the share of later ballots that round, by other seats,
                                   that go to that target -- by the speaker's model and team

Deaths come from replaying each journal through the engine (the log keeps no death list).

    python -m werewolf_harness.scripts.exp_report \
        strong=runs/cascade/strong cheap=runs/cascade/cheap \
        wolfS_goodH=runs/exp/wolfS_goodH wolfH_goodS=runs/exp/wolfH_goodS \
        hantiao=runs/exp/hantiao nomem=runs/exp/nomem --out docs/exp_report.md
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

from ..arena import Record, replay
from ..envs import Werewolf, engine_version_of
from .eval_baseline import WOLF, load_game, say_claims, self_role_claims, wilson


def deaths_of(path: str) -> list:
    d = json.load(open(path, encoding="utf-8"))
    env = Werewolf(d["variant"], playbook_mode=d.get("playbook", "static"), engine_version=engine_version_of(d))
    replay(env, d["seed"], [Record(**r) for r in d["journal"]])
    return list(env._deaths)


def models_of(path: str) -> dict[int, str]:
    d = json.load(open(path, encoding="utf-8"))
    s = d.get("settings") or {}
    roles = {int(k): v for k, v in d["ground_truth"]["roles"].items()}
    return {seat: (s.get("wolf_model") if s.get("wolf_model") and r == WOLF else d["model"]) for seat, r in roles.items()}


def short(model: str) -> str:
    return model.split(":")[-1]


def arm_stats(paths: list[str]) -> dict:
    wins = rounds = 0
    d1_hit = d1_all = all_hit = all_all = 0
    d1_wolf_exiled = 0
    claim_games = 0
    claim_votes = defaultdict(int)                # on the claim day: village ballots on fake / real seer / other
    pers = defaultdict(lambda: [0, 0])            # (speaker model, team) -> [follow, ballots]
    n = 0
    for p in paths:
        g = load_game(p)
        if not g.winner:
            continue
        n += 1
        wins += g.winner == "village"
        deaths = deaths_of(p)
        rounds += max([e.round for e in g.entries] + [1])
        wolves = g.wolves()
        seer = next((s for s, r in g.roles.items() if r == "seer"), None)
        model = models_of(p)
        exiled_d1 = [x.player_id for x in deaths if x.round == 1 and x.cause == "vote"]
        d1_wolf_exiled += any(s in wolves for s in exiled_d1)
        claimers: dict[int, int] = {}             # wolf seat -> round of first seer claim
        speeches = [e for e in g.entries if e.task == "speak"]
        for e in speeches:
            if e.actor in wolves and e.actor not in claimers and "seer" in self_role_claims(e.content):
                claimers[e.actor] = e.round
        if claimers:
            claim_games += 1
        for e in g.entries:
            if e.task != "vote" or e.target is None or e.actor in wolves:
                continue
            hit = e.target in wolves
            all_hit += hit
            all_all += 1
            if e.round == 1:
                d1_hit += hit
                d1_all += 1
            for w, r in claimers.items():
                if e.round == r:
                    claim_votes["fake" if e.target == w else "real" if e.target == seer else "other"] += 1
        # persuasion: a speech naming a target, then the other seats' ballots in that round
        votes_by_round = defaultdict(list)
        for e in g.entries:
            if e.task == "vote" and e.target is not None:
                votes_by_round[e.round].append(e)
        for e in speeches:
            c = say_claims(e.content, e.actor)
            named = {c.intent} if c.intent is not None else set(c.accuse)
            if len(named) != 1:
                continue
            t = next(iter(named))
            key = (short(model.get(e.actor, g.model)), g.team(e.actor))
            for v in votes_by_round[e.round]:
                if v.actor in (e.actor, t) or v.seq < e.seq:
                    continue
                pers[key][0] += v.target == t
                pers[key][1] += 1
    return {"games": n, "village_wins": wins, "rounds": rounds / n if n else None,
            "d1_hit": d1_hit, "d1_all": d1_all, "all_hit": all_hit, "all_all": all_all,
            "d1_wolf_exiled": d1_wolf_exiled, "claim_games": claim_games, "claim_votes": dict(claim_votes),
            "persuasion": {f"{m}/{t}": v for (m, t), v in sorted(pers.items())}}


def pct(k: int, n: int) -> str:
    if not n:
        return "—"
    p, lo, hi = wilson(k, n)
    return f"{p:.0%} [{lo:.0%}, {hi:.0%}] ({k}/{n})"


def render(stats: dict[str, dict]) -> str:
    out = ["| 组 | 局 | 好人胜率 | 平均轮数 | 首日好人投中狼 | 全程好人投中狼 | 首日放逐的是狼 | 狼悍跳的局 |",
           "|---|---|---|---|---|---|---|---|"]
    for arm, s in stats.items():
        out.append(f"| {arm} | {s['games']} | {pct(s['village_wins'], s['games'])} | {s['rounds']:.2f} | "
                   f"{pct(s['d1_hit'], s['d1_all'])} | {pct(s['all_hit'], s['all_all'])} | "
                   f"{pct(s['d1_wolf_exiled'], s['games'])} | {s['claim_games']}/{s['games']} |")
    out += ["", "悍跳当天好人的票（投给悍跳狼 / 投给真预言家 / 其他）：", ""]
    for arm, s in stats.items():
        if s["claim_votes"]:
            cv = s["claim_votes"]
            out.append(f"- {arm}：{cv.get('fake', 0)} / {cv.get('real', 0)} / {cv.get('other', 0)}")
    out += ["", "说服力：发言点名一个目标后，同一轮之后其他人（不含被点名者）投给这个目标的比例，按发言者的模型和阵营：", "",
            "| 组 | 发言者 | 跟票率 |", "|---|---|---|"]
    for arm, s in stats.items():
        for k, (f, n) in s["persuasion"].items():
            out.append(f"| {arm} | {k} | {pct(f, n)} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="name=dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    stats = {}
    for spec in a.arms:
        name, d = spec.split("=", 1)
        stats[name] = arm_stats(sorted(glob.glob(f"{d}/g*.json")))
    md = render(stats)
    print(md)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(md + "\n")
    if a.json:
        json.dump(stats, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
