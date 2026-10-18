"""python -m werewolf_harness.faults run [--scenario X ...] [--games N] [--quick]

Starts Postgres and Redis behind fault proxies, two API nodes and two model
workers as real processes, runs each scenario, checks every invariant, and
writes a scenario x invariant matrix to faults/report.md (+ report.json).
Exits non-zero if any cell is ✗.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sys
import tempfile
import time

from . import invariants
from .cluster import TIMING, Cluster
from .infra import Postgres, Redis, can_own_pg, can_own_redis
from .run import QUICK, SCENARIOS, TITLES, Run

HERE = os.path.dirname(os.path.abspath(__file__))


async def main_async(a) -> int:
    names = a.scenario or (QUICK if a.quick else SCENARIOS)
    games = a.games or (1 if a.quick else 2)
    run_dir = tempfile.mkdtemp(prefix="wh-faults-")
    own_pg = a.infra == "own" or (a.infra == "auto" and can_own_pg())
    own_rd = a.infra == "own" or (a.infra == "auto" and can_own_redis())
    url_pg = os.environ.get("FAULTS_DATABASE_URL") or os.environ.get("TEST_DATABASE_URL")
    url_rd = os.environ.get("FAULTS_REDIS_URL") or os.environ.get("TEST_REDIS_URL")
    if (not own_pg and not url_pg) or (not own_rd and not url_rd):
        print("需要本机的 initdb/pg_ctl 与 redis-server，或者 FAULTS_DATABASE_URL / FAULTS_REDIS_URL", file=sys.stderr)
        return 2
    pg, rd = Postgres(own_pg, url_pg), Redis(own_rd, url_rd)
    t0 = time.monotonic()
    print(f"== 故障注入：{len(names)} 个场景 × {games} 局；Postgres {'自起' if own_pg else '共享'}，"
          f"Redis {'自起' if own_rd else '共享'}；进程日志 {run_dir}")
    await pg.start()
    await rd.start()
    cluster = Cluster(pg.url, rd.url, run_dir)
    results = []
    run = None
    try:
        await cluster.start()
        print(f"   集群就绪 {time.monotonic() - t0:.1f}s：{', '.join(p.name for p in cluster.procs)}")
        run = Run(cluster, pg, rd, scale=0.5 if a.quick else 1.0)
        for i, name in enumerate(names):
            print(f"-- {name}：{TITLES[name]}")
            res = await run.scenario(name, games, seed0=7000 + 100 * i)
            results.append(res)
            cells = " ".join(("✓" if v["ok"] else "✗") for v in res["invariants"].values())
            print(f"   {cells}  {res['seconds']}s  步数 {res['stats']['steps']}，兜底 {res['stats']['forced']}，"
                  f"接管 {res['stats']['takeovers']}")
            for k, v in res["invariants"].items():
                if not v["ok"]:
                    print(f"   ✗ {invariants.LABELS[k]}: {v['detail']}")
    finally:
        if run is not None:
            await run.close()
        cluster.stop()
        await rd.stop()
        await pg.stop()
    total = round(time.monotonic() - t0, 1)
    meta = {"total_s": total, "quick": a.quick, "games_per_scenario": games, "own_postgres": own_pg,
            "own_redis": own_rd, "timing": TIMING, "python": platform.python_version(), "platform": platform.platform(),
            "when": time.strftime("%Y-%m-%d %H:%M"), "logs": run_dir}
    ok = all(v["ok"] for r in results for v in r["invariants"].values())
    out_md = a.out or os.path.join(HERE, "report.md")
    with open(os.path.splitext(out_md)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "ok": ok, "results": results}, f, ensure_ascii=False, indent=2, default=str)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(render(results, meta, ok))
    print(f"== {'全部通过' if ok else '有失败'}，用时 {total}s → {out_md}")
    return 0 if ok else 1


def render(results: list[dict], meta: dict, ok: bool) -> str:
    cols = invariants.NAMES
    lines = [f"# 故障注入报告{'（quick）' if meta['quick'] else ''}", "",
             f"{meta['when']}，{len(results)} 个场景 × {meta['games_per_scenario']} 局，总用时 **{meta['total_s']} s**，"
             f"结论：**{'全部通过' if ok else '有失败'}**。",
             f"Postgres {'自起（真实崩溃重启）' if meta['own_postgres'] else '共享（重启用代理模拟）'}，"
             f"Redis {'自起（真实重启、无持久化）' if meta['own_redis'] else '共享（重启用代理模拟 + FLUSHDB）'}；"
             f"2 个 API 节点 + 2 个模型 worker，全是独立进程；座位是确定性的 faults-mock。",
             f"时钟：租约 {meta['timing']['LEASE_TTL_MS']} ms，续约 {meta['timing']['LEASE_RENEW_S']} s，"
             f"恢复扫描 {meta['timing']['RECOVERY_SCAN_S']} s，回合期限 {meta['timing']['AGENT_TIMEOUT_S']} s。", "",
             "| 场景 | " + " | ".join(invariants.LABELS[c] for c in cols) + " | 用时 s | 步数 | 兜底 | 接管 |",
             "|---|" + "---|" * (len(cols) + 4)]
    for r in results:
        cells = ["✓" if r["invariants"][c]["ok"] else "✗" for c in cols]
        s = r["stats"]
        lines.append(f"| {r['title']} | " + " | ".join(cells) + f" | {r['seconds']} | {s['steps']} | {s['forced']} | "
                     f"{s['takeovers']} |")
    lines += ["", "兜底 = 期限到了用默认动作提交的回合；接管 = fencing token 增加的次数（换了 owner 或同节点重新领回）。", ""]
    for r in results:
        lines.append(f"## {r['title']}（`{r['scenario']}`）")
        lines.append("")
        lines += [f"- 注入：{n}" for n in r["notes"]]
        for c in cols:
            v = r["invariants"][c]
            lines.append(f"- {'✓' if v['ok'] else '✗'} {invariants.LABELS[c]}：{v['detail']}")
        lines.append("")
    lines += ["## 怎么复现", "", "```", "make faults          # 全部场景", "make faults-quick    # CI 用的子集",
              "python -m werewolf_harness.faults run --scenario node_kill --games 3", "```", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m werewolf_harness.faults")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--scenario", action="append", choices=SCENARIOS)
    r.add_argument("--games", type=int, default=0, help="每个场景同时跑几局（默认 2，--quick 为 1）")
    r.add_argument("--quick", action="store_true", help=f"CI 子集：{', '.join(QUICK)}")
    r.add_argument("--infra", choices=["auto", "own", "shared"], default="auto")
    r.add_argument("--out", help="报告路径（默认 faults/report.md，旁边写同名 .json）")
    a = ap.parse_args()
    sys.exit(asyncio.run(main_async(a)))


if __name__ == "__main__":
    main()
