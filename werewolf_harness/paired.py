"""Read a folder of saved games and put the two rule settings side by side.

Every game here was dealt from a seed, so the same seed under two settings is
the same deal played twice -- which is the only way a handful of games can say
anything at all. Unpaired games are counted but never averaged into a claim.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load(folder: Path) -> dict[tuple[str, int], dict]:
    best: dict[tuple[str, int], dict] = {}
    for f in sorted(folder.glob("g*_12p_*_seed*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        g = d["game"]
        if g["state"] != "over":
            continue
        key = (g["playbook"], g["seed"])
        # a resumed game and the run it continued share a seed; the longer one
        # is the whole game, the shorter one is the stump it was resumed from
        if key not in best or best[key]["game"]["health"]["turns"] < g["health"]["turns"]:
            best[key] = d
    return best


def report(folder: Path) -> str:
    by = load(folder)
    seeds = sorted({s for _, s in by})
    out = [f"{'seed':6}{'static':30}{'dynamic':30}",
           f"{'':6}{'胜者   轮  回合  兜底 分工':30}{'胜者   轮  回合  兜底 分工':30}"]
    pairs = []
    for s in seeds:
        cells = []
        for pb in ("static", "dynamic"):
            d = by.get((pb, s))
            if not d:
                cells.append("—".ljust(30)); continue
            g, o = d["game"], d["game"].get("outcome") or {}
            plans = sum(1 for j in d["journal"] if j["task"] == "pack_plan")
            cells.append(f"{'狼' if o.get('winner')=='werewolf' else '村民':5}{o.get('round'):<4}"
                         f"{g['health']['turns']:<6}{g['health']['forced']:<5}{plans:<4}".ljust(30))
        a, b = by.get(("static", s)), by.get(("dynamic", s))
        if a and b:
            pairs.append((s, (a["game"]["outcome"] or {}).get("winner"),
                          (b["game"]["outcome"] or {}).get("winner")))
        out.append(f"{s:<6}{cells[0]}{cells[1]}")

    same = sum(1 for _, x, y in pairs if x == y)
    to_wolf = sum(1 for _, x, y in pairs if x == "village" and y == "werewolf")
    to_vil = sum(1 for _, x, y in pairs if x == "werewolf" and y == "village")
    out.append(f"\n配对 {len(pairs)} 对：相同 {same}，"
               f"static村民→dynamic狼 {to_wolf}，static狼→dynamic村民 {to_vil}")

    for pb in ("static", "dynamic"):
        rows = [d for (p, _), d in by.items() if p == pb]
        if not rows:
            continue
        turns = sum(d["game"]["health"]["turns"] for d in rows)
        forced = sum(d["game"]["health"]["forced"] for d in rows)
        wolf = sum(1 for d in rows if (d["game"]["outcome"] or {}).get("winner") == "werewolf")
        plans = sum(sum(1 for j in d["journal"] if j["task"] == "pack_plan") for d in rows)
        out.append(f"{pb:8} {len(rows)} 局 · 狼胜 {wolf}/{len(rows)} · {turns} 回合 · "
                   f"兜底 {forced} ({forced/turns:.2%}) · 狼分工 {plans} 次")

    per = defaultdict(lambda: {"turns": 0, "forced": 0, "th": 0, "terms": 0})
    for d in by.values():
        sm = {int(k): v for k, v in d["game"]["seat_models"].items()}
        for j in d["journal"]:
            m = sm.get(j["actor"])
            if not m:
                continue
            a = per[m]
            a["turns"] += 1
            a["forced"] += bool(j.get("forced"))
            a["terms"] += 1
            a["th"] += bool(((j.get("action") or {}).get("args") or {}).get("thought"))
    out.append(f"\n{'模型':22}{'回合':7}{'兜底率':10}{'thought':9}")
    for m, a in sorted(per.items(), key=lambda kv: -kv[1]["turns"]):
        out.append(f"{m:22}{a['turns']:<7}{a['forced']/a['turns']:<10.2%}{a['th']/a['terms']:<9.0%}")
    return "\n".join(out)




def wolves(folder: Path) -> str:
    """How the pack actually played, not just whether it won.

    Win or lose turns on the deal as much as on the play, so a handful of games
    cannot answer "is dynamic better". What it can answer is narrower and more
    useful: did the extra planning change what the wolves DID -- did they vote
    together more, keep their own alive longer, get exiled less often?
    """
    by = load(folder)
    out = [f"{'':10}{'static':>10}{'dynamic':>10}   （每局平均，除非注明）"]
    agg: dict[str, dict] = {}
    for (pb, _), d in by.items():
        g = d["game"]
        roles = {int(k): v for k, v in (d.get("revealed") or {}).items()}
        pack = {s for s, r in roles.items() if r == "werewolf"}
        pub = d.get("public") or {}
        dead = pub.get("dead") or []
        a = agg.setdefault(pb, {"games": 0, "rounds": 0, "wolves_exiled": 0, "villagers_killed": 0,
                                "detonations": 0, "ballots": 0, "united": 0, "plans": 0,
                                "kept": 0, "targeted": 0, "left": 0})
        a["games"] += 1
        a["rounds"] += (g.get("outcome") or {}).get("round") or 0
        a["left"] += len(pack - {x["player_id"] for x in dead})
        for x in dead:
            if x["player_id"] in pack and x["cause"] == "exiled":
                a["wolves_exiled"] += 1
            if x["player_id"] not in pack and x["cause"] == "night":
                a["villagers_killed"] += 1
        a["detonations"] += sum(1 for j in d["journal"]
                                if (j.get("action") or {}).get("name") == "self_detonate")
        a["plans"] += sum(1 for j in d["journal"] if j["task"] == "pack_plan")

        # cohesion: on each day's ballot, did the living wolves name one target?
        ballots: dict[int, dict[int, int]] = {}
        rnd = 1
        for j in d["journal"]:
            if j["task"] == "vote":
                t = ((j.get("action") or {}).get("args") or {}).get("target_id")
                if j["actor"] in pack and t is not None:
                    ballots.setdefault(rnd, {})[j["actor"]] = t
            if j["task"] == "night":
                rnd += 1
        for cast in ballots.values():
            if len(cast) < 2:
                continue
            a["ballots"] += 1
            a["united"] += len(set(cast.values())) == 1

        # did the pack's agreed vote_target actually get their votes?
        agreed = [((j.get("action") or {}).get("args") or {}).get("vote_target")
                  for j in d["journal"] if j["task"] == "pack_plan"]
        agreed = [x for x in agreed if x is not None]
        for cast in ballots.values():
            for want in agreed:
                if want in cast.values():
                    a["kept"] += 1
                a["targeted"] += 1
                break

    def row(label, f, pct=False):
        cells = []
        for pb in ("static", "dynamic"):
            a = agg.get(pb)
            cells.append("—".rjust(10) if not a else
                         (f"{f(a):>9.0%}" if pct else f"{f(a):>10.2f}"))
        return f"{label:10}{cells[0]}{cells[1]}"

    out.append(row("局数", lambda a: a["games"]))
    out.append(row("狼分工次数", lambda a: a["plans"] / a["games"]))
    out.append(row("打了几轮", lambda a: a["rounds"] / a["games"]))
    out.append(row("狼被票出", lambda a: a["wolves_exiled"] / a["games"]))
    out.append(row("刀掉好人", lambda a: a["villagers_killed"] / a["games"]))
    out.append(row("自爆", lambda a: a["detonations"] / a["games"]))
    out.append(row("终局存活狼", lambda a: a["left"] / a["games"]))
    out.append(row("票型一致", lambda a: a["united"] / a["ballots"] if a["ballots"] else 0, pct=True))
    out.append(row("按约定投", lambda a: a["kept"] / a["targeted"] if a["targeted"] else 0, pct=True))
    return "\n".join(out)


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent.parent / "games" / "resume"
    print(report(folder))
    print("\n== 狼队打法")
    print(wolves(folder))
