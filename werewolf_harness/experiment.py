"""Run the same table N times and report what actually differed.

One game tells you almost nothing: who won turns on the deal as much as on the
play. The point of a batch is the numbers that survive being averaged -- how
often each model's seat had to be filled in by the harness, how long it took to
answer, how much it cost -- and those only mean something when every game was
dealt from a different seed and played by the same six models.

Games run one at a time on purpose. Several vendors rate-limit hard, and a
batch that trips a limit measures the limit rather than the models.
"""
from __future__ import annotations

import json
import statistics as st
import time
from pathlib import Path

from .server.service import Game


def _row(g: Game, seed: int, wall: float) -> dict:
    h = g.health()
    per: dict[str, dict] = {}
    for seat, model in g.seat_models.items():
        p = g.players.get(seat)
        hist = list(getattr(p, "history", []) or [])
        row = per.setdefault(model, {"seats": [], "turns": 0, "forced": 0,
                                     "latency_ms": [], "thoughts": 0, "terminals": 0})
        row["seats"].append(seat)
        row["turns"] += len(hist)
        for t in hist:
            rec = getattr(t, "recovery", None) or {}
            if rec.get("fallback_used"):
                row["forced"] += 1
            if getattr(t, "latency_ms", 0):
                row["latency_ms"].append(t.latency_ms)
            act = getattr(t, "action", None)
            if isinstance(act, dict):
                row["terminals"] += 1
                if (act.get("args") or {}).get("thought"):
                    row["thoughts"] += 1
    for row in per.values():
        lat = row.pop("latency_ms")
        row["latency_median_s"] = round(st.median(lat) / 1000, 1) if lat else None
    return {
        "seed": seed,
        "state": g.state,
        "error": g.error,
        "winner": (g.outcome or {}).get("winner"),
        "rounds": (g.outcome or {}).get("round"),
        "wall_s": round(wall, 1),
        "turns": h["turns"],
        "forced": h["forced"],
        "forced_rate": h["forced_rate"],
        "usable": h["usable"],
        "transport_failures": h["transport_failures"],
        "tokens": h["tokens"],
        "roles": {str(s): r for s, r in (g.env._roles or {}).items()},
        "review_remarks": len(g.env.review_log()) if hasattr(g.env, "review_log") else 0,
        "pack_chat": len(g.env.pack_chat_log()) if hasattr(g.env, "pack_chat_log") else 0,
        "by_model": per,
    }


def run_batch(*, games: int, variant: str, models: list[str], env_keys: dict,
              guard: str = "L1+L2+L3", playbook: str = "static",
              out: Path, seed0: int = 1, max_tokens: int = 4_000_000) -> dict:
    rows: list[dict] = []
    out.mkdir(parents=True, exist_ok=True)
    for i in range(games):
        seed = seed0 + i
        started = time.time()
        print(f"[{i + 1}/{games}] seed={seed} 开局…", flush=True)
        g = Game(variant=variant, model=models, guard=guard, playbook=playbook,
                 human_seats=[], turn_seconds=None, seed=seed,
                 max_tokens=max_tokens, env_keys=env_keys)
        g.start()
        g._thread.join()
        row = _row(g, seed, time.time() - started)
        rows.append(row)
        (out / f"game_{seed}.json").write_text(
            json.dumps({**row, "journal": [
                {"n": n + 1, "actor": r.actor, "task": r.task, "action": r.action, "forced": r.forced}
                for n, r in enumerate(g.journal)],
                "review": g.env.review_log() if hasattr(g.env, "review_log") else []},
                ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"    {row['state']} · {row['winner']} 胜 · {row['rounds']} 轮 · {row['turns']} 回合 · "
              f"兜底 {row['forced']} ({row['forced_rate']:.1%}) · {row['tokens'] // 1000}k tok · "
              f"{row['wall_s']:.0f}s", flush=True)
        (out / "summary.json").write_text(
            json.dumps(summarise(rows), ensure_ascii=False, indent=1), encoding="utf-8")
    return summarise(rows)


def summarise(rows: list[dict]) -> dict:
    done = [r for r in rows if r["state"] == "over"]
    per: dict[str, dict] = {}
    for r in rows:
        for model, m in r["by_model"].items():
            a = per.setdefault(model, {"turns": 0, "forced": 0, "terminals": 0, "thoughts": 0,
                                       "latencies": [], "wolf_games": 0, "wolf_wins": 0,
                                       "village_games": 0, "village_wins": 0})
            a["turns"] += m["turns"]
            a["forced"] += m["forced"]
            a["terminals"] += m["terminals"]
            a["thoughts"] += m["thoughts"]
            if m["latency_median_s"] is not None:
                a["latencies"].append(m["latency_median_s"])
            # A model sat several seats; count each seat's side separately, because
            # winning as a wolf and winning as a villager are not the same skill.
            for seat in m["seats"]:
                role = r["roles"].get(str(seat))
                if role is None:
                    continue
                side = "wolf" if role == "werewolf" else "village"
                a[f"{side}_games"] += 1
                if r["winner"] == ("werewolf" if side == "wolf" else "village"):
                    a[f"{side}_wins"] += 1
    for a in per.values():
        lat = a.pop("latencies")
        a["latency_median_s"] = round(st.median(lat), 1) if lat else None
        a["forced_rate"] = round(a["forced"] / a["turns"], 4) if a["turns"] else 0.0
        a["thought_rate"] = round(a["thoughts"] / a["terminals"], 3) if a["terminals"] else 0.0
        for side in ("wolf", "village"):
            n = a[f"{side}_games"]
            a[f"{side}_winrate"] = round(a[f"{side}_wins"] / n, 3) if n else None
    return {
        "games": len(rows),
        "finished": len(done),
        "aborted": [r["seed"] for r in rows if r["state"] != "over"],
        "village_wins": sum(1 for r in done if r["winner"] == "village"),
        "wolf_wins": sum(1 for r in done if r["winner"] == "werewolf"),
        "median_rounds": st.median([r["rounds"] for r in done]) if done else None,
        "median_turns": st.median([r["turns"] for r in done]) if done else None,
        "total_tokens": sum(r["tokens"] for r in rows),
        "median_wall_s": round(st.median([r["wall_s"] for r in rows]), 1) if rows else None,
        "forced_rate_overall": round(sum(r["forced"] for r in rows) / max(1, sum(r["turns"] for r in rows)), 4),
        "transport_failures": sum(r["transport_failures"] for r in rows),
        "by_model": per,
        "games_detail": rows,
    }
