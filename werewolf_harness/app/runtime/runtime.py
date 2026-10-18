"""All games on this node: creating, adopting (recovery), renewing, draining.

A node drives a game only while it holds that game's lease. Three loops:

  renew     every lease_renew_s: extend every lease we hold; a lease we failed to
            extend is gone, so that actor is stopped before it can write again
            (and if it tries anyway, the fencing token stops it in the database)
  scan      every recovery_scan_s: any live game in Postgres that nobody holds is
            adopted -- replayed from its journal, model memory restored, and the
            current turn asked again
  (startup) is just the first scan
  load      every load_tick_s: sample the model layer's load (load.py) and start
            queued games while it has room (admission.py, docs/adr/0011)

Replay is cheap enough to do on every adoption: a whole 12-seat game (~150 steps)
replays in about 3 ms, so there are no snapshots (docs/adr/0003).
"""
from __future__ import annotations

import asyncio
import inspect
import secrets
import time
from collections.abc import Callable
from typing import Any

from redis.asyncio import Redis

from ...arena import Record
from ...envs import UnsupportedEngineVersion, engine_version_of
from ...players.providers import retired
from ..config import Settings
from ..obs import logging as olog
from ..storage import Store
from .actor import ActorHooks, GameActor
from .admission import ADMIT, QUEUE, REJECT, Admission, Overloaded, Ticket
from .agents import LocalExecutor
from .lease import Leases
from .load import LoadMonitor, LoadSample, Thresholds
from .table import Table, TableConfig
from .timers import TimingWheel

log = olog.get("runtime")


class Runtime:
    def __init__(self, settings: Settings, store: Store, redis: Redis, *, executor=None,
                 env_keys: dict | None = None, hooks: ActorHooks | None = None, hub=None):
        self.s = settings
        self.store = store
        self.redis = redis
        self.node = settings.node_id
        self.leases = Leases(redis, self.node, settings.lease_ttl_ms)
        self.executor = executor or LocalExecutor(settings.agent_workers, fair=settings.fair_queuing)
        self.wheel = TimingWheel(tick=0.1)
        self.env_keys = env_keys or {}
        self.hub = hub                              # app.llm.gateway.Hub (model layer), optional
        self.hooks = hooks or ActorHooks()
        self.actors: dict[str, GameActor] = {}
        self.finished: dict[str, Table] = {}        # recently ended here, kept for readers (bounded)
        self.draining = False
        self._tasks: list[asyncio.Task] = []
        self._adopting: set[str] = set()
        self.incompatible: dict[str, str] = {}      # live games whose engine version this code cannot replay
        self.counters = {"adopted": 0, "lost_leases": 0, "created": 0}
        self.listeners: list[Callable[[str, dict], Any]] = []
        self.monitor = LoadMonitor(Thresholds(
            wait_s=settings.overload_wait_s, demand_per_slot=settings.overload_demand_per_slot,
            latency_ms=settings.overload_latency_ms, enter=settings.overload_enter, exit=settings.overload_exit,
            hold_s=settings.overload_hold_s, per_game_tau_s=settings.overload_per_game_tau_s),
            gateway_latency=getattr(hub, "latency", None))
        self.admission = Admission(max_queue=settings.max_queued_games, max_active=settings.max_active_games,
                                   probe_s=settings.admission_probe_s)
        self._ended_seen: set[str] = set()

    # ------------------------------------------------------------ lifecycle
    async def start(self, scan: bool = True) -> None:
        self.wheel.start()
        start = getattr(self.executor, "start", None)
        if start:
            await start()
        self._tasks.append(asyncio.create_task(self._renew_loop(), name="lease-renew"))
        if self.hub is not None:
            self._tasks.append(asyncio.create_task(self._flush_calls(), name="llm-call-flush"))
        if scan:
            await self.scan()
            self._tasks.append(asyncio.create_task(self._scan_loop(), name="recovery-scan"))
        await self._requeue()
        self._tasks.append(asyncio.create_task(self._load_loop(), name="load"))

    async def shutdown(self, drain: bool = True) -> None:
        """Stop driving games and hand them over. With drain, leases are released
        so another node adopts them at its next scan instead of waiting out the TTL."""
        self.draining = True
        for t in self._tasks:
            t.cancel()
        for a in list(self.actors.values()):
            await a.stop("shutdown")
            if drain:
                await self.leases.release(a.id)
        self.actors.clear()
        await self.wheel.stop()
        await self.flush_calls()
        close = getattr(self.executor, "close", None)
        if close:
            await close()

    # ------------------------------------------------------------ games
    async def create(self, cfg: TableConfig, *, seed: int | None = None, created_by: str | None = None,
                     start: bool | None = None) -> GameActor:
        """Start a game now, whatever the load (tests, benches). The public door is
        `request`, which goes through admission control."""
        table = self._new_table(cfg, seed)
        await self.store.create_game(game_id=table.id, variant=cfg.variant, seed=table.seed, config=cfg.to_dict(),
                                     state=table.state, seats=table.seat_rows(), created_by=created_by,
                                     engine_version=table.engine_version)
        return await self._launch(table)

    async def request(self, cfg: TableConfig, *, seed: int | None = None,
                      created_by: str | None = None) -> tuple[str, GameActor | None, dict]:
        """Create a game through admission control: ("admit", actor, {}) when it started,
        ("queue", None, {game_id, position, eta_s, ...}) when it waits in line.
        Raises Overloaded when the line is full."""
        if not self.s.admission:
            return ADMIT, await self.create(cfg, seed=seed, created_by=created_by), {}
        self._validate(cfg)
        await self.refresh_load()
        d = self.admission.decide(overloaded=self.monitor.overloaded, active=self.active_games(),
                                  busy=self.monitor.level == "busy")
        if d == REJECT:
            raise Overloaded(self.admission.retry_after_s(self.active_games()))
        if d == ADMIT:
            return ADMIT, await self.create(cfg, seed=seed, created_by=created_by), {}
        table = self._new_table(cfg, seed)
        await self.store.create_game(game_id=table.id, variant=cfg.variant, seed=table.seed, config=cfg.to_dict(),
                                     state="queued", seats=table.seat_rows(), created_by=created_by,
                                     owner_node=self.node, engine_version=table.engine_version)
        self.admission.push(Ticket(table.id, table))
        olog.kv(log, 20, "game queued", game_id=table.id, position=len(self.admission),
                pressure=round(self.monitor.pressure, 2))
        return QUEUE, None, self.queue_info(table.id) or {}

    def _validate(self, cfg: TableConfig) -> None:
        for m in cfg.models:
            why = retired(m)
            if why:
                raise ValueError(f"{m} 已经下架（{why}）")
        if self.draining:
            raise RuntimeError("this node is draining; try again")

    def _new_table(self, cfg: TableConfig, seed: int | None) -> Table:
        self._validate(cfg)
        gid = "g" + secrets.token_hex(4)
        seed = seed if seed is not None else int.from_bytes(secrets.token_bytes(2), "big")
        return Table(gid, seed, cfg, self.env_keys, hub=self.hub)

    async def _launch(self, table: Table) -> GameActor:
        gid = table.id
        if not await self.leases.acquire(gid):
            raise RuntimeError("could not take the lease on a brand-new game")
        token = await self.store.bump_fencing(gid, self.node)
        table.reset()
        actor = self._actor(table, token)
        self.counters["created"] += 1
        if table.state == "running":
            table.emit("phase", phase="start", seed=table.seed)
        actor.launch()
        return actor

    # ------------------------------------------------------------ load + admission
    def active_games(self) -> int:
        return sum(1 for a in self.actors.values() if a.stopped is None)

    def queue_info(self, gid: str) -> dict | None:
        info = self.admission.info(gid, self.active_games())
        return {"game_id": gid, **info} if info else None

    async def refresh_load(self) -> bool:
        """Sample the executor now and update the overload state."""
        fn = getattr(self.executor, "load_sample", None)
        sample = fn() if fn else LoadSample(inflight=getattr(self.executor, "inflight", 0))
        if inspect.isawaitable(sample):
            sample = await sample
        sample.games = self.active_games()
        return self.monitor.observe(sample)

    def load_view(self) -> dict:
        n = self.active_games()
        return {**self.monitor.view(), "node": self.node, "active_games": n,
                "admission": {"enabled": self.s.admission, "queued_games": len(self.admission),
                              "max_queued_games": self.admission.max_queue,
                              "max_active_games": self.admission.max_active,
                              "probe_s": self.admission.probe_s,
                              "next_eta_s": self.admission.eta_s(len(self.admission) + 1, n),
                              "mean_game_s": round(self.admission.mean_game_s, 1),
                              "counts": dict(self.admission.counts)}}

    async def admit_queued(self, limit: int = 1) -> int:
        """Start up to `limit` queued games, oldest first, while there is room."""
        n = 0
        while n < limit and self.admission.can_start_next(overloaded=self.monitor.overloaded,
                                                          active=self.active_games(),
                                                          busy=self.monitor.level == "busy"):
            t = self.admission.pop()
            table = t.table
            try:
                await self.store.set_state(table.id, table.state)        # queued -> waiting | running
                await self._launch(table)
            except Exception:                                  # noqa: BLE001 -- one bad start must not block the line
                log.exception("could not start queued game %s", table.id)
                continue
            olog.kv(log, 20, "game admitted from queue", game_id=table.id,
                    waited_s=round(self.admission.waits[-1], 1))
            n += 1
        return n

    async def _requeue(self) -> None:
        """After a restart: this node's queued games go back in line, oldest first."""
        try:
            rows = await self.store.queued_games(self.node)
        except Exception:                                      # noqa: BLE001 -- no such state yet: nothing queued
            log.exception("could not read queued games")
            return
        for r in rows:
            if r["id"] in self.admission:
                continue
            try:
                # pinned when it was queued: a deploy while it waited does not change its rules
                table = self.table_from_row(r, self.env_keys, hub=self.hub)
            except UnsupportedEngineVersion:
                log.exception("queued game %s has rules this code cannot run", r["id"])
                continue
            self.admission.push(Ticket(r["id"], table, enqueued=r["created_at"].timestamp()))

    def _note_ended(self) -> None:
        for gid, a in list(self.actors.items()):
            if a.stopped == "finished" and gid not in self._ended_seen:
                self._ended_seen.add(gid)
                self.admission.game_ended(time.time() - a.t.created)
        self._ended_seen &= set(self.actors)

    async def _load_loop(self) -> None:
        while True:
            await asyncio.sleep(self.s.load_tick_s)
            try:
                await self.refresh_load()
                self._note_ended()
                await self.admit_queued(1)
            except asyncio.CancelledError:
                raise
            except Exception:                                  # noqa: BLE001 -- a failed tick is retried next tick
                log.exception("load tick failed")

    async def import_log(self, log: dict, *, max_tokens: int | None = None, created_by: str | None = None) -> GameActor:
        """Continue a game from an old JSON save (the pre-database format).

        The saved journal is written into a new game's journal, then the game is
        started exactly the way a recovered one is: replay, then ask the next turn.
        """
        g = log.get("game") or {}
        records = [Record(r["actor"], r["task"], r["action"], bool(r.get("forced"))) for r in log.get("journal") or []]
        if not records:
            raise ValueError("这份日志里没有回合，直接重开就行")
        if any(r.action is None or "name" not in (r.action or {}) for r in records):
            raise ValueError("这份日志的动作不全（多半是没权限那半份），换一份完整的")
        cfg = TableConfig(variant=g.get("variant", "12p"), models=list(g.get("models") or ["mock"]),
                          guard=g.get("guard", "L1+L2+L3"), playbook=g.get("playbook", "static"),
                          human_seats=list(g.get("human_seats") or []), turn_seconds=None,
                          max_tokens=max_tokens or 6_000_000, xray=bool(g.get("xray")),
                          # the save's own rules; a save from before versioning played the baseline
                          engine_version=engine_version_of(g))
        gid = "g" + secrets.token_hex(4)
        table = Table(gid, int(g["seed"]), cfg, self.env_keys, hub=self.hub)
        self.replay(table, records)                       # validate before writing anything
        await self.store.create_game(game_id=gid, variant=cfg.variant, seed=table.seed, config=cfg.to_dict(),
                                     state="running", seats=table.seat_rows(), created_by=created_by,
                                     resumed_from=g.get("id", "存档"), engine_version=table.engine_version)
        if not await self.leases.acquire(gid):
            raise RuntimeError("could not take the lease on a brand-new game")
        token = await self.store.bump_fencing(gid, self.node)
        for i, r in enumerate(records):
            await self.store.append(gid, token=token, seq=i, actor=r.actor, task=r.task, action=r.action, forced=r.forced)
        table.state = "running"
        table.resumed_from = g.get("id", "存档")
        table.emit("phase", phase="resume", round=getattr(table.env, "_round", 0), turns=len(table.journal))
        actor = self._actor(table, token)
        actor.launch()
        return actor

    def get(self, gid: str) -> GameActor | None:
        return self.actors.get(gid)

    def table(self, gid: str) -> Table | None:
        a = self.actors.get(gid)
        return a.t if a else self.finished.get(gid)

    async def load_readonly(self, gid: str) -> Table | None:
        """A game this node does not drive (ended, or held elsewhere), rebuilt for reading."""
        t = self.table(gid)
        if t is not None:
            return t
        row = await self.store.get_game(gid)
        if row is None:
            return None
        table = self.table_from_row(row, self.env_keys)
        table.state = row["state"]
        rows = await self.store.journal(gid, archived=row.get("archived_at") is not None)
        self.replay(table, [Record(r.actor, r.task, r.action, r.forced) for r in rows])
        table.outcome, table.error = row["outcome"], row["error"]
        if table.state in ("over", "aborted", "crashed"):
            self._remember(table)
        return table

    def _remember(self, table: Table) -> None:
        self.finished[table.id] = table
        while len(self.finished) > 200:
            self.finished.pop(next(iter(self.finished)))

    def _actor(self, table: Table, token: int) -> GameActor:
        table.draft_epoch = token                   # draft ids are unique per incarnation
        a = GameActor(table, store=self.store, executor=self.executor, wheel=self.wheel, token=token,
                      agent_timeout_s=self.s.agent_timeout_s, hooks=self.hooks)
        self.actors[table.id] = a
        return a

    # ------------------------------------------------------------ recovery
    async def adopt(self, gid: str) -> GameActor | None:
        """Take over a live game nobody holds: lease, fence, replay, restore memory, continue."""
        if gid in self.actors or gid in self._adopting or self.draining:
            return None
        self._adopting.add(gid)
        try:
            if not await self.leases.acquire(gid):
                return None
            t0 = time.monotonic()
            token = await self.store.bump_fencing(gid, self.node)
            row = await self.store.get_game(gid)
            if row is None or row["state"] not in ("waiting", "running"):
                await self.leases.release(gid)
                return None
            rows = await self.store.journal(gid)
            try:
                return self._resume(gid, token, row, rows, await self.store.agent_memory(gid), t0)
            except UnsupportedEngineVersion as exc:
                await self._leave_incompatible(gid, exc)
                return None
        finally:
            self._adopting.discard(gid)

    async def adopt_many(self, gids: list[str]) -> int:
        """adopt() for a batch: the same steps, but each one is a single round trip for
        the whole batch (leases in one pipeline, fencing bumped and game rows returned by
        one UPDATE, journals and memory in one query each). One by one, each adoption was
        ~8 ms of driver round trips around ~2 ms of replay (docs/db_scale.md)."""
        gids = [g for g in dict.fromkeys(gids) if g not in self.actors and g not in self._adopting]
        if not gids or self.draining:
            return 0
        self._adopting.update(gids)
        try:
            t0 = time.monotonic()
            won = await self.leases.acquire_many(gids)
            claimed = await self.store.claim_live(won, self.node) if won else {}
            for g in won:
                if g not in claimed:                         # ended (or vanished) since the scan read it
                    await self.leases.release(g)
            journals = await self.store.journals(list(claimed))
            memories = await self.store.agent_memories(list(claimed))
            n = 0
            for gid, (token, row) in claimed.items():
                try:
                    self._resume(gid, token, row, journals[gid], memories.get(gid, {}), t0)
                    n += 1
                except UnsupportedEngineVersion as exc:
                    await self._leave_incompatible(gid, exc)
                except Exception as exc:                      # noqa: BLE001 -- one broken game must not keep the rest down
                    olog.kv(log, 40, "adoption failed", game_id=gid, error=repr(exc))
                    await self.leases.release(gid)
            return n
        finally:
            self._adopting.difference_update(gids)

    async def _leave_incompatible(self, gid: str, exc: Exception) -> None:
        """Written by a newer node (rolling deploy) or under retired rules: replaying it
        here would rebuild a different world. Leave it for a node that can, and stop
        trying it on every scan."""
        await self.leases.release(gid)
        self.incompatible[gid] = str(exc)
        olog.kv(log, 30, "cannot adopt: engine version", game_id=gid, error=str(exc))

    def _resume(self, gid: str, token: int, row: dict, rows: list, memory: dict, t0: float) -> GameActor:
        # under the rules the game was created with (the row's engine_version column), so a
        # batch adopted after a deploy still rebuilds each game in its own world
        table = self.table_from_row(row, self.env_keys, hub=self.hub)
        table.state = row["state"]
        table.resumed_from = row.get("resumed_from")
        self.replay(table, [Record(r.actor, r.task, r.action, r.forced) for r in rows],
                    request_ids={r.request_id: r.seq for r in rows if r.request_id})
        table.memory = memory
        table.recoveries += 1
        table.emit("phase", phase="resume", round=getattr(table.env, "_round", 0), turns=len(table.journal),
                   node=self.node)
        actor = self._actor(table, token)
        # Drafts are not persisted: whatever the last owner was streaming died with it. Say
        # so first, then the open turns are asked again (docs/adr/0009).
        table.draft_reset("recovered")
        actor.launch()
        self.counters["adopted"] += 1
        olog.kv(log, 20, "adopted", game_id=gid, steps=len(rows), ms=round((time.monotonic() - t0) * 1000, 1),
                token=token, memory_seats=len(table.memory))
        for fn in self.listeners:
            fn("adopted", {"game_id": gid, "ms": (time.monotonic() - t0) * 1000})
        return actor

    @staticmethod
    def table_from_row(row: dict, env_keys: dict | None = None, hub=None) -> Table:
        """A Table for a stored game, under the rules it was created with.

        The ``engine_version`` column decides (a row from before the column defaults to
        the baseline); the config copy is overwritten with it so the two cannot disagree.
        Raises UnsupportedEngineVersion when this code cannot reproduce that version.
        """
        cfg = TableConfig.from_dict(row["config"])
        cfg.engine_version = engine_version_of(row)
        created = row.get("created_at")
        return Table(row["id"], row["seed"], cfg, env_keys, created=created.timestamp() if created else None, hub=hub)

    @staticmethod
    def replay(table: Table, journal: list[Record], request_ids: dict | None = None) -> None:
        """Rebuild world and events from the journal. No model is asked anything."""
        table.reset()
        if journal or table.state == "running":
            table.emit("phase", phase="start", seed=table.seed)
        for rec in journal:
            t = table.turn
            if not hasattr(t, "actor") or (t.actor, t.task) != (rec.actor, rec.task):
                raise ValueError(f"journal diverged at step {len(table.journal)}: env expects {t}, row is "
                                 f"{rec.actor}/{rec.task}")
            table.phase_event()
            record, news = table.apply(rec.action, rec.forced)
            table.step_events(record, news)
        table.request_ids = dict(request_ids or {})

    async def scan(self) -> int:
        """Adopt every live game nobody holds.

        At startup after a full outage that is every live game: the scan asks Redis about
        all of them in one MGET, then adopts them `recovery_batch` at a time (adopt_many).
        """
        live = [g["id"] for g in await self.store.live_games()
                if g["id"] not in self.actors and g["id"] not in self.incompatible]
        holders = await self.leases.holders(live)
        orphans = [g for g in live if holders.get(g) is None]
        step = max(1, self.s.recovery_batch)
        n = 0
        for i in range(0, len(orphans), step):
            n += await self.adopt_many(orphans[i:i + step])
        return n

    async def _scan_loop(self) -> None:
        while True:
            await asyncio.sleep(self.s.recovery_scan_s)
            try:
                await self.scan()
            except Exception:                              # noqa: BLE001 -- a failed scan is retried next tick
                log.exception("recovery scan failed")

    async def _flush_calls(self) -> None:
        """Model-call records are written in batches, off the hot path."""
        while True:
            await asyncio.sleep(1.0)
            await self.flush_calls()

    async def flush_calls(self) -> int:
        if self.hub is None:
            return 0
        rows = [r.to_row() for r in self.hub.sink.drain(5000)]
        if rows:
            try:
                await self.store.record_calls(rows)
            except Exception:                              # noqa: BLE001 -- accounting must not stop games
                log.exception("could not write llm_calls")
        return len(rows)

    async def _renew_loop(self) -> None:
        while True:
            await asyncio.sleep(self.s.lease_renew_s)
            for gid, a in list(self.actors.items()):
                if a.stopped is not None:
                    # finished, fenced or out of db: hand the lease back and forget it
                    self.actors.pop(gid, None)
                    if a.stopped in ("finished", "aborted", "crashed"):
                        self._remember(a.t)
                    await self.leases.release(gid)
                    continue
                try:
                    ok = await self.leases.renew(gid)
                except Exception:                          # noqa: BLE001 -- redis blip: try again next tick
                    continue
                if not ok:
                    self.counters["lost_leases"] += 1
                    olog.kv(log, 30, "lease lost", game_id=gid)
                    self.actors.pop(gid, None)
                    await a.stop("lease_lost")
