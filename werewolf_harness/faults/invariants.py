"""What must hold after every scenario, checked against Postgres (read directly,
not through the fault proxy), the model-call log and what the spectators saw.

    journal       seq 0..n-1 with no gap and no duplicate, per game
    fencing       the fencing token sampled during the run never went down
    no_recall     no model call started for a turn after that turn was committed
    drafts        every draft a WebSocket / SSE reader was shown ended committed with
                  exactly the journal's text, or retracted; no event seq skipped
    terminal      every game ended: finished, or stopped with an explicit error -- none stuck
    replay        rebuilding a finished game from (seed, journal) ends where it ended live
    memory        each seat-memory row belongs to a journal row that seat committed
"""
from __future__ import annotations

import json
from collections import defaultdict

from sqlalchemy import text

from ..app.runtime.runtime import Runtime
from ..arena import Record, Terminal

NAMES = ["journal", "fencing", "no_recall", "drafts", "terminal", "replay", "memory"]
LABELS = {"journal": "日志连续无重复", "fencing": "fencing 单调", "no_recall": "已提交回合不再调模型",
          "drafts": "客户端草稿闭合", "terminal": "每局终结", "replay": "重放一致", "memory": "座位记忆对齐"}


async def check(store, gids: list[str], *, fencing_samples: dict[str, list[int]], calls: list[dict],
                watchers: list, deadline_hit: set[str]) -> dict[str, dict]:
    """{invariant: {"ok": bool, "detail": str}}"""
    out: dict[str, dict] = {}
    async with store.engine.connect() as c:
        games = {r["id"]: dict(r) for r in (await c.execute(text(
            "SELECT id, seed, config, state, outcome, error, engine_version, fencing_token, created_at, resumed_from "
            "FROM games WHERE id = ANY(:ids)"), {"ids": gids})).mappings().all()}
        jrows = (await c.execute(text(
            "SELECT game_id, seq, actor, task, action, forced, extract(epoch from created_at) AS t FROM journal "
            "WHERE game_id = ANY(:ids) ORDER BY game_id, seq"), {"ids": gids})).mappings().all()
        mem = (await c.execute(text("SELECT game_id, seat, seq FROM agent_state WHERE game_id = ANY(:ids)"),
                               {"ids": gids})).mappings().all()
    journals: dict[str, list] = defaultdict(list)
    for r in jrows:
        journals[r["game_id"]].append(r)

    # journal
    bad = [g for g in gids if [r["seq"] for r in journals[g]] != list(range(len(journals[g])))]
    out["journal"] = _res(not bad, f"{sum(len(journals[g]) for g in gids)} 行" if not bad else f"不连续: {bad}")

    # fencing
    down = {g: s for g, s in fencing_samples.items() if any(b < a for a, b in zip(s, s[1:], strict=False))}
    owners = sum(games[g]["fencing_token"] for g in gids if g in games)
    out["fencing"] = _res(not down, f"{owners} 次持有（token 合计）" if not down else f"token 回退: {down}")

    # no_recall: a turn's first model step must start before that turn was committed
    viol = []
    by_game: dict[str, list] = defaultdict(list)
    for x in calls:
        if x.get("g") in games and x.get("kind") == "step" and x.get("step") == 0 and x.get("v") is not None:
            by_game[x["g"]].append(x)
    for g, xs in by_game.items():
        rows = journals[g]
        for x in xs:
            row = next((r for r in rows[x["v"]:] if r["actor"] == x["s"] and r["task"] == x["task"]), None)
            if row is not None and x["t"] > float(row["t"]) + 0.05:
                viol.append(f"{g}#{row['seq']} seat {x['s']} asked {x['t'] - float(row['t']):.2f}s after commit")
    n_calls = sum(len(v) for v in by_game.values())
    out["no_recall"] = _res(not viol, f"{n_calls} 次回合首调用" if not viol else f"{len(viol)} 次: {viol[:3]}")

    # drafts
    probs = []
    shown = 0
    for w in watchers:
        rows = [Record(r["actor"], r["task"], _j(r["action"]), r["forced"]) for r in journals[w.gid]]
        v = w.mirror.violations(rows, final=True)
        shown += w.mirror.counts()["shown"]
        if w.ended is None and games.get(w.gid, {}).get("state") in ("over", "aborted", "crashed"):
            probs.append(f"{w.kind}:{w.gid} 没收到结束事件（连接停在不再驱动这局的节点上）")
        if w.gaps:
            probs.append(f"{w.kind}:{w.gid} 事件序号跳了 {w.gaps} 次")
        probs += [f"{w.kind}:{w.gid} {p}" for p in v[:3]]
    polled = sum(1 for w in watchers if w.ended == "poll")
    out["drafts"] = _res(not probs, (f"{len(watchers)} 个客户端看到 {shown} 条草稿"
                                     + (f"，{polled} 个在节点间时对局结束、按网页的做法轮询到结局" if polled else ""))
                         if not probs else "; ".join(probs[:4]))

    # terminal
    states = {g: games.get(g, {}).get("state") for g in gids}
    stuck = [g for g, s in states.items() if s not in ("over", "aborted", "crashed") or g in deadline_hit]
    errs = [f"{g}:{games[g]['error']}" for g, s in states.items() if s in ("aborted", "crashed")]
    out["terminal"] = _res(not stuck, (f"{sum(s == 'over' for s in states.values())}/{len(gids)} 打完"
                                       + (f"，明确报错 {errs}" if errs else "")) if not stuck else f"卡住: {stuck} {states}")

    # replay
    mism = []
    for g in gids:
        row = games.get(g)
        if row is None or row["state"] != "over":
            continue
        try:
            t = Runtime.table_from_row({**row, "config": _j(row["config"])})
            Runtime.replay(t, [Record(r["actor"], r["task"], _j(r["action"]), r["forced"]) for r in journals[g]])
            ok = isinstance(t.turn, Terminal) and _norm(t.turn.outcome) == _norm(_j(row["outcome"]))
        except Exception as exc:                           # noqa: BLE001
            ok = False
            mism.append(f"{g}: {exc}")
            continue
        if not ok:
            mism.append(f"{g}: outcome differs")
    out["replay"] = _res(not mism, "终局一致" if not mism else "; ".join(mism))

    # memory
    misal = []
    for r in mem:
        rows = journals[r["game_id"]]
        if r["seq"] >= len(rows) or rows[r["seq"]]["actor"] != r["seat"]:
            misal.append(f"{r['game_id']} seat {r['seat']} seq {r['seq']}")
    out["memory"] = _res(not misal, f"{len(mem)} 行" if not misal else f"{len(misal)} 行错位: {misal[:3]}")
    return out


def _res(ok: bool, detail: str) -> dict:
    return {"ok": bool(ok), "detail": detail}


def _j(v):
    return json.loads(v) if isinstance(v, str) else v


def _norm(v):
    return json.loads(json.dumps(v, default=str))
