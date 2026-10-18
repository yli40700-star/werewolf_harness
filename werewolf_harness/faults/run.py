"""Scenarios: each starts its games on the running cluster, injects one kind of
fault while they play, waits for them to end, and checks every invariant.

    worker_kill    kill -9 a model worker while it runs turns (speeches streaming)
    node_kill      kill -9 the API node that owns the games; another adopts them
    redis_restart  Redis goes away and comes back empty (no persistence)
    pg_restart     Postgres crashes while commits are in flight, then recovers
    model_faults   a window of 503s, then of timeouts, then of slow answers
    network        latency + jitter, segment loss, a partition, then every connection reset
    dup_zombie     jobs delivered twice; the owner frozen past its lease, woken after takeover
"""
from __future__ import annotations

import asyncio
import json
import os
import time

from redis.asyncio import Redis
from sqlalchemy import text

from ..app.llm.queue import LANES, jobs_key
from ..app.storage import Store
from . import invariants
from .clients import SSEWatcher, WSWatcher
from .cluster import TIMING, Cluster
from .infra import Postgres
from .infra import Redis as RedisInfra

LEASE_S = int(TIMING["LEASE_TTL_MS"]) / 1000
SCENARIOS = ["worker_kill", "node_kill", "redis_restart", "pg_restart", "model_faults", "network", "dup_zombie"]
QUICK = ["worker_kill", "node_kill", "model_faults", "dup_zombie"]
TITLES = {"worker_kill": "worker 回合中被 kill -9", "node_kill": "owner 节点被 kill -9",
          "redis_restart": "Redis 重启（无持久化）", "pg_restart": "Postgres 提交中崩溃重启",
          "model_faults": "模型报错 / 超时 / 变慢窗口", "network": "网络延迟 / 丢包 / 分区 / 断连",
          "dup_zombie": "任务重复投递 + 被 fence 的旧 owner 醒来"}


class Run:
    def __init__(self, cluster: Cluster, pg: Postgres, redis: RedisInfra, log=print, scale: float = 1.0):
        self.c, self.pg, self.redis = cluster, pg, redis
        self.scale = scale                                 # fault windows are this many times as long (quick: 0.5)
        self.store = Store.connect(pg.direct_url)          # the checks bypass the proxies
        self.r = Redis.from_url(redis.direct_url)
        self.log = log
        self.token: str | None = None
        self.notes: list[str] = []
        self.create_retries = 0

    async def close(self) -> None:
        await self.store.close()
        await self.r.aclose()

    # ------------------------------------------------------------ helpers
    async def _node(self):
        while True:
            for n in self.c.up_nodes():
                try:
                    if (await self.c.get(n, "/readyz")).get("ok"):
                        return n
                except Exception:                          # noqa: BLE001
                    pass
            await asyncio.sleep(0.2)

    async def create(self, k: int, seed0: int) -> list[str]:
        n = await self._node()
        if self.token is None:
            self.token = (await self.c.post(n, "/api/auth/guest", {}))["token"]
        out = []
        for i in range(k):
            body = {"variant": "8p", "models": ["faults-mock"], "human_seats": [], "seed": seed0 + i}
            for attempt in range(5):                       # a client retries a 5xx, as the web page does
                try:
                    g = await self.c.post(n, "/api/games", body, self.token)
                    break
                except Exception:                          # noqa: BLE001
                    if attempt == 4:
                        raise
                    self.create_retries += 1
                    await asyncio.sleep(0.5)
                    n = await self._node()
            out.append(g["id"])
        return out

    async def versions(self, gids: list[str]) -> dict[str, int]:
        async with self.store.engine.connect() as c:
            rows = (await c.execute(text("SELECT game_id, count(*) FROM journal WHERE game_id = ANY(:i) GROUP BY 1"),
                                    {"i": gids})).all()
        d = {g: 0 for g in gids}
        d.update({r[0]: int(r[1]) for r in rows})
        return d

    async def states(self, gids: list[str]) -> dict[str, dict]:
        async with self.store.engine.connect() as c:
            rows = (await c.execute(text("SELECT id, state, owner_node, fencing_token FROM games WHERE id = ANY(:i)"),
                                    {"i": gids})).mappings().all()
        return {r["id"]: dict(r) for r in rows}

    async def until_version(self, gids: list[str], v: int, timeout: float = 60) -> None:
        """Until every game has `v` steps (or has ended: a short game must not stall the scenario)."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            vs, st = await self.versions(gids), await self.states(gids)
            if all(vs[g] >= v or st[g]["state"] not in ("waiting", "running") for g in gids):
                return
            await asyncio.sleep(0.1)

    async def until_draft_open(self, watchers, timeout: float = 30) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if any(d.state == "open" and d.text for w in watchers for d in w.mirror.drafts.values()):
                return True
            await asyncio.sleep(0.02)
        return False

    async def owner(self, gids: list[str]):
        st = await self.states(gids)
        counts: dict[str, int] = {}
        for s in st.values():
            counts[s["owner_node"]] = counts.get(s["owner_node"], 0) + 1
        name = max(counts, key=counts.get)
        return next(n for n in self.c.nodes if n.name == name)

    def note(self, msg: str) -> None:
        self.notes.append(msg)
        self.log(f"    · {msg}")

    # ------------------------------------------------------------ one scenario
    async def scenario(self, name: str, games: int, seed0: int, timeout: float = 150) -> dict:
        self.notes = []
        self.c.set_control({})
        calls_from = os.path.getsize(self.c.call_log) if os.path.exists(self.c.call_log) else 0
        t0 = time.monotonic()
        gids = await self.create(games, seed0)
        watchers = [cls(self.c, g) for g in gids for cls in (WSWatcher, SSEWatcher)]
        for w in watchers:
            w.start()
        samples: dict[str, list[int]] = {g: [] for g in gids}
        stop = asyncio.Event()

        async def sample() -> None:
            while not stop.is_set():
                try:
                    for g, s in (await self.states(gids)).items():
                        samples[g].append(int(s["fencing_token"]))
                except Exception:                          # noqa: BLE001
                    pass
                await asyncio.sleep(0.1)

        sampler = asyncio.create_task(sample())
        try:
            await getattr(self, f"_{name}")(gids, watchers)
        except Exception as exc:                           # noqa: BLE001 -- the injection itself failed: say so
            self.note(f"注入失败: {type(exc).__name__}: {exc}")
        self.c.set_control({})
        for p in self.redis.proxy, self.pg.proxy:
            p.clear()
        # every injected process is back (a scenario may leave one dead on purpose for a while)
        for p in self.c.procs:
            if not p.alive:
                await self.c.restart(p)
        deadline_hit = await self._settle(gids, watchers, t0 + timeout)
        stop.set()
        await sampler
        for w in watchers:
            await w.close()
        calls = _read_calls(self.c.call_log, calls_from)
        res = await invariants.check(self.store, gids, fencing_samples=samples, calls=calls, watchers=watchers,
                                     deadline_hit=deadline_hit)
        st = await self.states(gids)
        async with self.store.engine.connect() as c:
            forced = await c.scalar(text("SELECT count(*) FROM journal WHERE game_id = ANY(:i) AND forced"),
                                    {"i": gids})
            steps = await c.scalar(text("SELECT count(*) FROM journal WHERE game_id = ANY(:i)"), {"i": gids})
        return {"scenario": name, "title": TITLES[name], "games": gids, "seconds": round(time.monotonic() - t0, 1),
                "invariants": res, "notes": list(self.notes),
                "stats": {"steps": int(steps or 0), "forced": int(forced or 0),
                          "takeovers": sum(int(s["fencing_token"]) - 1 for s in st.values()),
                          "states": {g: s["state"] for g, s in st.items()},
                          "watchers": [w.report() for w in watchers]}}

    async def _settle(self, gids, watchers, deadline: float) -> set[str]:
        """Wait until every game is terminal in the database and its readers have seen the end."""
        while time.monotonic() < deadline:
            st = await self.states(gids)
            done = all(s["state"] in ("over", "aborted", "crashed") for s in st.values())
            if done and all(w.ended for w in watchers):
                return set()
            if done:
                # the readers get a few seconds more to catch the last events
                end = min(deadline, time.monotonic() + 8)
                while time.monotonic() < end and not all(w.ended for w in watchers):
                    await asyncio.sleep(0.1)
                return set()
            await asyncio.sleep(0.2)
        st = await self.states(gids)
        return {g for g, s in st.items() if s["state"] not in ("over", "aborted", "crashed")}

    # ------------------------------------------------------------ the faults
    async def _worker_kill(self, gids, watchers) -> None:
        for i in range(2):
            await self.until_version(gids, 8 + 8 * i)
            mid = await self.until_draft_open(watchers, 20)
            w = self.c.workers[i % len(self.c.workers)]
            w.kill()
            self.note(f"kill -9 {w.name}{'（有草稿正在流式输出）' if mid else ''}")
            await asyncio.sleep(2)
            w.start()

    async def _node_kill(self, gids, watchers) -> None:
        await self.until_version(gids, 10)
        await self.until_draft_open(watchers, 10)
        n = await self.owner(gids)
        before = await self.states(gids)
        n.kill()
        t_kill = time.monotonic()
        self.note(f"kill -9 {n.name}（owner），其它节点等租约过期（{LEASE_S:.0f}s）后接管")
        while time.monotonic() - t_kill < 30:
            st = await self.states(gids)
            if all(st[g]["fencing_token"] > before[g]["fencing_token"] and st[g]["owner_node"] != n.name
                   for g in gids if st[g]["state"] == "running"):
                self.note(f"全部被接管，用时 {time.monotonic() - t_kill:.1f}s")
                break
            await asyncio.sleep(0.1)
        n.start()
        await self.c.ready(n)

    async def _redis_restart(self, gids, watchers) -> None:
        await self.until_version(gids, 10)
        how = await self.redis.restart(down_s=1.0)
        self.note(how)
        await self.until_version(gids, 16)
        how = await self.redis.restart(down_s=0.3)
        self.note("第二次：" + how)

    async def _pg_restart(self, gids, watchers) -> None:
        await self.until_version(gids, 10)
        how = await self.pg.restart(down_s=2.0)
        self.note(how + "，停 2s")
        await self.until_version(gids, 16)
        how = await self.pg.restart(down_s=12.0)
        self.note("第二次停 12s（超过 actor 的提交重试窗口）")

    async def _model_faults(self, gids, watchers) -> None:
        await self.until_version(gids, 6)
        for ctl, secs, what in (({"error_rate": 0.5}, 4, "50% 调用报 503"),
                                ({"timeout_rate": 0.5, "timeout_s": 1.0}, 4, "50% 调用 1s 后超时"),
                                ({"error_rate": 1.0}, 3, "100% 报错"),
                                ({"slow_ms": 2500}, 5, "每次调用慢 2.5s")):
            secs *= self.scale
            self.c.set_control(ctl)
            self.note(f"{what} × {secs:g}s")
            await asyncio.sleep(secs)
        self.c.set_control({})

    async def _network(self, gids, watchers) -> None:
        await self.until_version(gids, 6)
        px = (self.pg.proxy, self.redis.proxy)
        for p in px:
            p.latency(150, 100)
        self.note("Postgres/Redis 链路延迟 150±100ms × 5s")
        await asyncio.sleep(5)
        for p in px:
            p.latency(0)
            p.loss(0.2)
        self.note("20% 分段被延迟重传 200–600ms × 5s")
        await asyncio.sleep(5)
        for p in px:
            p.loss(0)
            p.blackhole(True)
        self.note("分区：所有链路不通 × 2.5s")
        await asyncio.sleep(2.5)
        for p in px:
            p.blackhole(False)
        await asyncio.sleep(2)
        n = sum(p.cut() for p in px)
        self.note(f"重置全部 {n} 条连接")

    async def _dup_zombie(self, gids, watchers) -> None:
        await self.until_version(gids, 6)
        dup = 0
        for _ in range(6):                                 # copies of recent jobs: delivered twice
            for lane in LANES:
                for _eid, f in await self.r.xrevrange(jobs_key(lane), count=4):
                    await self.r.xadd(jobs_key(lane), f)
                    dup += 1
            await asyncio.sleep(0.4)
        self.note(f"重复投递 {dup} 个模型任务")
        await self.until_draft_open(watchers, 10)
        n = await self.owner(gids)
        before = await self.states(gids)
        n.pause()
        self.note(f"SIGSTOP {n.name}（owner），超过租约 {LEASE_S:.0f}s")
        t = time.monotonic()
        while time.monotonic() - t < 30:
            st = await self.states(gids)
            if all(st[g]["fencing_token"] > before[g]["fencing_token"] for g in gids if st[g]["state"] == "running"):
                break
            await asyncio.sleep(0.1)
        await asyncio.sleep(1.5)                           # the new owner commits a few steps first
        n.resume()
        self.note(f"其它节点已接管，SIGCONT {n.name}：它手里的答案只能被 fencing 拒绝")


def _read_calls(path: str, start: int) -> list[dict]:
    out = []
    try:
        with open(path, "rb") as f:
            f.seek(start)
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out
