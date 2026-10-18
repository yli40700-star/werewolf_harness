"""Model turns through Redis Streams, run by separate worker processes.

    node (actor)                    Redis                                 worker process(es)
    ------------                    -----                                 ------------------
    submit(job) ── XADD ──▶ llm:jobs:human | online | offline ──XREADGROUP──▶ rebuild the world from
                                                                              Postgres, run the turn
    results loop ◀── XREAD ── llm:results:{node} ◀──────────── XADD ────────── then XACK the job

Why a queue at all: model turns are slow (seconds) and bursty; the game loop
must not care how many there are or where they run. Workers scale on their own,
a slow provider backs up the queue instead of the API, and a worker crash loses
nothing -- its unacknowledged jobs are claimed by another worker (XAUTOCLAIM).

Delivery is at-least-once. A job can run twice (a worker died after the model
answered but before XACK), so results are idempotent at the receiving end: the
actor accepts a result only for the exact `version` it is waiting on.

Lanes are strict priority: a turn with a person waiting at the table goes
before an AI-only game, which goes before offline evaluation. Fairness *within*
a lane: since the actor asks every open turn at once (a vote asks 8 seats), one
game can put 8 jobs in a lane in one instant, and FIFO would serve all 8 before
another game's single turn. A stream cannot be reordered, so the node bounds it
at the source instead: a game has at most `per_game_asks` jobs in the stream;
the rest wait on the node and go out as that game's answers come back. FIFO over
games that each hold at most k places is round-robin within a factor of k.
(The in-process LocalExecutor can do better -- true per-game fair queuing --
because it owns its queue; see runtime/agents.py and docs/adr/0011.)

Load: `load_sample()` reads each lane's consumer-group lag (jobs not yet taken
by any worker), the age of the oldest of them, the pending count (taken, not
acknowledged = in flight), and the capacity the live workers announce in
`llm:workers` -- the inputs of the overload signal (runtime/load.py).

Jobs carry only (game, seat, version, task, memory). The worker rebuilds the
exact view and legal actions by replaying the journal (a few ms), so nothing
un-serialisable crosses the wire and a job can never act on a world that moved on.

Streaming (docs/adr/0009): a speech job's draft pieces go back on the same
results stream as its result (field "d" instead of "r"), written by one sender
per job in order, and the result only after the last piece -- so the node sees
them in the order they were generated and never a piece after its result. A
node that no longer wants a job (deadline, stopped game) PUBLISHes on
``llm:cancel``; the worker running it pulls the plug on its model call.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import socket
import time
from collections import deque
from collections.abc import Callable

from redis.asyncio import Redis

from ...arena import Record, Turn
from ..config import Settings
from ..obs import logging as olog
from ..obs import metrics, tracing
from ..runtime.agents import AgentJob, AgentResult, DraftMsg, run_agent_turn
from ..runtime.load import LatencyWindow, LoadSample

LANES = ("human", "online", "offline")
GROUP = "workers"
WORKERS_KEY = "llm:workers"           # hash: consumer -> {"capacity", "ts"}, refreshed by each worker
CANCEL_CHANNEL = "llm:cancel"
log = olog.get("llm.queue")


def jobs_key(lane: str) -> str:
    return f"llm:jobs:{lane}"


def results_key(node: str) -> str:
    return f"llm:results:{node}"


class StreamExecutor:
    """The node side: publish jobs, receive results, hand them to the actors."""

    def __init__(self, redis: Redis, settings: Settings, per_game_asks: int | None = None):
        self.r = redis
        self.s = settings
        self.node = settings.node_id
        self.deliver: dict[str, Callable[[AgentResult], None]] = {}
        self.inflight = 0
        self.per_game_asks = settings.per_game_asks if per_game_asks is None else per_game_asks
        # per game: jobs in the stream (seat, version) -> when sent, and jobs held back on the node
        self.out: dict[str, dict[tuple[int, int], float]] = {}
        self.held: dict[str, deque[AgentJob]] = {}
        self.latency = LatencyWindow()                     # per-turn service time on the workers
        self.send_retries = 0
        # jobs in the stream, kept so a dead-lettered one can be asked again (once)
        self.jobs: dict[tuple[str, int, int], AgentJob] = {}
        self.reasked: set[tuple[str, int, int]] = set()
        self.reasks = 0
        self._task: asyncio.Task | None = None
        self._last_id = "$"

    async def start(self) -> None:
        if self._task is None:
            self._last_id = "$"
            self._task = asyncio.create_task(self._results(), name="llm-results")

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None

    def submit(self, job: AgentJob, deliver: Callable[[AgentResult], None]) -> None:
        self.deliver[job.game_id] = deliver
        self.held.setdefault(job.game_id, deque()).append(job)
        self._release(job.game_id)

    def _release(self, gid: str) -> None:
        """Send this game's held jobs while it has fewer than per_game_asks in the stream."""
        out = self.out.setdefault(gid, {})
        # a job whose answer never comes back (dead-lettered, or the worker dropped it
        # and is from before stale replies) must not hold its slot for ever
        ttl = self.s.agent_timeout_s + 5
        now = time.monotonic()
        for k in [k for k, t in out.items() if now - t > ttl]:
            del out[k]
        q = self.held.get(gid)
        while q and (not self.per_game_asks or len(out) < self.per_game_asks):
            job = q.popleft()
            out[(job.seat, job.version)] = now
            self._publish(job)
        if not q:
            self.held.pop(gid, None)
        if not out:
            self.out.pop(gid, None)

    def _settle(self, gid: str, seat: int, version: int) -> None:
        self.jobs.pop((gid, seat, version), None)
        out = self.out.get(gid)
        if out is not None:
            out.pop((seat, version), None)
        self._release(gid)

    def _publish(self, job: AgentJob) -> None:
        self.inflight += 1
        self.jobs[(job.game_id, job.seat, job.version)] = job
        fields = {"game_id": job.game_id, "seat": job.seat, "version": job.version, "task": job.task,
                  "memory": json.dumps(job.memory, ensure_ascii=False) if job.memory else "",
                  "reply_to": self.node, "enq": time.time(), "lane": job.lane,
                  "stream": "1" if job.stream else "0", "job_id": job.job_id,
                  "tp": json.dumps(tracing.to_carrier(job.trace_ctx)) if job.trace_ctx is not None else ""}
        asyncio.get_running_loop().create_task(self._send(job, fields))

    async def _send(self, job: AgentJob, fields: dict) -> None:
        """XADD the job, retrying through a Redis blip. A job that never reached the stream
        used to be noticed only by the turn's deadline (agent_timeout_s, then the default
        action). Gives up once the job is no longer wanted (cancelled: its slot is gone)."""
        delay = 0.2
        while True:
            try:
                await self.r.xadd(jobs_key(job.lane), fields, maxlen=100_000, approximate=True)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:                       # noqa: BLE001
                if (job.seat, job.version) not in self.out.get(job.game_id, {}):
                    return
                self.send_retries += 1
                log.warning("could not queue a model job (%s): retrying in %.1fs", str(exc)[:120], delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 2.0)

    def cancel(self, job: AgentJob, reason: str) -> None:
        """Stop a job. One still held on this node (per-game cap) never goes out. One in
        the stream: tell whichever worker runs it to stop it -- fire-and-forget; a worker
        that has not taken it yet remembers the cancel and drops it with a stale reply,
        which gives the game's slot back. If nobody hears it, the result is dropped as
        stale anyway."""
        q = self.held.get(job.game_id)
        if q and any(j is job for j in q):
            kept = deque(j for j in q if j is not job)
            if kept:
                self.held[job.game_id] = kept
            else:
                self.held.pop(job.game_id, None)
            return
        msg = json.dumps({"job_id": job.job_id, "game_id": job.game_id, "seat": job.seat, "version": job.version,
                          "reason": reason})
        asyncio.get_running_loop().create_task(self._publish_cancel(msg))
        # Its answer is not wanted, so it no longer holds one of the game's places in the
        # stream. Waiting for the worker's reply to give it back would block the game's next
        # asks for agent_timeout_s whenever that reply never comes (Redis lost the job).
        self._settle(job.game_id, job.seat, job.version)

    async def _publish_cancel(self, msg: str) -> None:
        try:
            await self.r.publish(CANCEL_CHANNEL, msg)
        except Exception:                                  # noqa: BLE001 -- a lost cancel costs tokens, not correctness
            log.warning("could not publish a cancel")

    async def _results(self) -> None:
        key = results_key(self.node)
        while True:
            try:
                got = await self.r.xread({key: self._last_id}, count=100, block=1000)
            except asyncio.CancelledError:
                raise
            except Exception as exc:                       # noqa: BLE001 -- redis blip: back off and retry
                # logged: a bug here (a bad argument) looks exactly like a blip and stalls every game
                log.warning("reading results failed (%s): retrying", str(exc)[:200])
                await asyncio.sleep(0.5)
                continue
            for _stream, entries in got or []:
                for eid, f in entries:
                    self._last_id = eid
                    piece = f.get(b"d", f.get("d"))
                    if piece is not None:                  # a draft piece, not a result
                        m = json.loads(_s(piece))
                        fn = self.deliver.get(m.get("game_id"))
                        if fn:
                            fn(DraftMsg(**{k: m[k] for k in DraftMsg.__dataclass_fields__ if k in m}))
                        continue
                    d = json.loads(_s(f[b"r"] if b"r" in f else f["r"]))
                    self.inflight = max(0, self.inflight - 1)
                    job_key = (d["game_id"], int(d["seat"]), int(d["version"]))
                    job = self.jobs.get(job_key)
                    self._settle(*job_key)
                    if d.get("dead"):
                        # delivered too often and dead-lettered: ask once more instead of
                        # leaving the turn to its deadline; a second death waits it out
                        if job is not None and job_key not in self.reasked and d["game_id"] in self.deliver:
                            self.reasked.add(job_key)
                            self.reasks += 1
                            self.submit(job, self.deliver[d["game_id"]])
                        continue
                    if d.get("stale"):
                        continue                           # the worker dropped it: only its slot comes back
                    if (d.get("meta") or {}).get("ms") is not None and not d["meta"].get("cancelled"):
                        self.latency.record(d["meta"]["ms"])
                    fn = self.deliver.get(d["game_id"])
                    if fn:
                        mem = d.get("memory")
                        fn(AgentResult(d["game_id"], d["seat"], d["version"], d.get("action"), d.get("forced", False),
                                       tuple(mem) if mem else None, d.get("meta") or {}, d.get("error")))


    async def load_sample(self) -> LoadSample:
        """Queue depth / oldest wait per lane from the consumer group, capacity from live workers."""
        depth: dict[str, int] = {}
        oldest_ms: int | None = None
        pending = 0
        for lane in LANES:
            key = jobs_key(lane)
            try:
                groups = await self.r.xinfo_groups(key)
            except Exception:                              # noqa: BLE001 -- no stream yet
                depth[lane] = 0
                continue
            g = next((g for g in groups if _s(g.get("name")) == GROUP), None)
            if g is None:
                depth[lane] = 0
                continue
            lag = g.get("lag")
            pending += int(g.get("pending") or 0)
            last = _s(g.get("last-delivered-id") or "0-0")
            nxt = await self.r.xrange(key, min="(" + last, max="+", count=1)
            if lag is None:                                # Redis < 7: count what is past the cursor
                lag = len(await self.r.xrange(key, min="(" + last, max="+", count=10_000))
            depth[lane] = int(lag)
            if nxt:
                ms = int(_s(nxt[0][0]).split("-")[0])
                oldest_ms = ms if oldest_ms is None else min(oldest_ms, ms)
        held = sum(len(q) for q in self.held.values())
        if held:
            depth["held"] = held
        cap = 0
        now = time.time()
        for _k, v in (await self.r.hgetall(WORKERS_KEY)).items():
            try:
                w = json.loads(_s(v))
            except ValueError:
                continue
            if now - float(w.get("ts", 0)) < 10:
                cap += int(w.get("capacity", 0))
        p50, _p95, _n = self.latency.quantiles()
        return LoadSample(depth=depth, oldest_wait_s=0.0 if oldest_ms is None else max(0.0, now - oldest_ms / 1000),
                          inflight=pending, capacity=max(1, cap), turn_p50_ms=p50)


class Worker:
    """A worker process: pulls jobs by lane priority, runs them, replies, acks."""

    def __init__(self, settings: Settings, store, redis: Redis, hub, *, consumer: str | None = None,
                 concurrency: int = 8, env_keys: dict | None = None, claim_idle_ms: int | None = None,
                 max_deliveries: int = 3, store_calls: bool = False):
        self.s, self.store, self.r, self.hub = settings, store, redis, hub
        self.consumer = consumer or f"{socket.gethostname()}-{os.getpid()}"
        self.sem = asyncio.Semaphore(concurrency)
        self.concurrency = concurrency
        # its own pool: asyncio.to_thread's default pool is min(32, cpus + 4) threads, which
        # load testing found capping a 2-core worker at ~6 turns in flight (~8 commits/s)
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="turn")
        self.env_keys = env_keys or {}
        # A job a worker took and stopped working on is claimed by another after it sat this
        # long unacknowledged. Workers keep the jobs they are running fresh (_keepalive_loop),
        # so this measures how long a worker has been dead, not how long its turn takes.
        self.claim_idle_ms = claim_idle_ms or getattr(settings, "claim_idle_ms", 0) or int(settings.agent_timeout_s * 1000)
        self.store_calls = store_calls                    # write the hub's model-call records to Postgres
        self.held: dict[bytes | str, str] = {}            # entry id -> lane, for jobs running here
        self.redis_errors = 0
        self.max_deliveries = max_deliveries
        self.done = 0
        self.stale = 0
        self.dead = 0
        self.order: list[str] = []          # lanes in the order jobs were taken (tests read this)
        self.waits: dict[str, list[float]] = {x: [] for x in LANES}   # ms each job sat in the queue
        self._tasks: set[asyncio.Task] = set()
        self._stop = False
        self.running: dict[str, AgentJob] = {}            # job_id -> the job this worker is running
        self.cancelled = 0
        self._cancel_early: dict[str, float] = {}         # cancels for jobs not started (here) yet
        self.drafts_sent = 0

    async def ensure_groups(self) -> None:
        for lane in LANES:
            try:
                await self.r.xgroup_create(jobs_key(lane), GROUP, id="0", mkstream=True)
            except Exception as e:                         # noqa: BLE001 -- BUSYGROUP: already there
                if "BUSYGROUP" not in str(e):
                    raise

    async def run(self) -> None:
        await self.ensure_groups()
        reclaim = asyncio.create_task(self._reclaim_loop())
        beat = asyncio.create_task(self._announce_loop())
        self._tasks.add(asyncio.create_task(self._cancel_loop()))
        self._tasks.add(asyncio.create_task(self._keepalive_loop()))
        if self.store_calls:
            self._tasks.add(asyncio.create_task(self._flush_loop()))
        delay = 0.2
        try:
            while not self._stop:
                await self.sem.acquire()
                try:
                    got = await self._next()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:                   # noqa: BLE001 -- Redis restarted or unreachable
                    # A worker outlives Redis: it waits and reads again. A Redis that came back
                    # empty has no consumer group (NOGROUP), so make it again.
                    self.sem.release()
                    self.redis_errors += 1
                    log.warning("reading jobs failed (%s): retrying in %.1fs", str(exc)[:120], delay)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 3.0)
                    try:
                        await self.ensure_groups()
                    except Exception:                      # noqa: BLE001 -- still down: next round
                        pass
                    continue
                delay = 0.2
                if got is None:
                    self.sem.release()
                    continue
                lane, eid, fields = got
                t = asyncio.create_task(self._handle(lane, eid, fields))
                self._tasks.add(t)
                t.add_done_callback(self._tasks.discard)
        finally:
            reclaim.cancel()
            beat.cancel()
            try:
                await self.r.hdel(WORKERS_KEY, self.consumer)
            except Exception:                              # noqa: BLE001
                pass

    async def _announce_loop(self) -> None:
        """Tell nodes how many jobs this worker can run at once (their load signal's capacity)."""
        while True:
            try:
                await self.r.hset(WORKERS_KEY, self.consumer,
                                  json.dumps({"capacity": self.concurrency, "ts": time.time()}))
            except Exception:                              # noqa: BLE001 -- redis blip
                pass
            await asyncio.sleep(2.0)

    async def stop(self) -> None:
        self._stop = True
        for job in list(self.running.values()):
            job.cancel.cancel("worker stopping")      # unacked: another worker claims it and runs it again
        for t in list(self._tasks):
            t.cancel()
        self.pool.shutdown(wait=False, cancel_futures=True)

    async def _next(self):
        # strict priority: look at each lane without blocking, highest first...
        for lane in LANES:
            got = await self.r.xreadgroup(GROUP, self.consumer, {jobs_key(lane): ">"}, count=1)
            if got:
                eid, f = got[0][1][0]
                return lane, eid, f
        # ...then wait on all of them for a while
        got = await self.r.xreadgroup(GROUP, self.consumer, {jobs_key(x): ">" for x in LANES}, count=1, block=500)
        if not got:
            return None
        stream, entries = got[0]
        lane = _s(stream).rsplit(":", 1)[1]
        eid, f = entries[0]
        return lane, eid, f

    async def _handle(self, lane: str, eid, raw: dict) -> None:
        self.held[eid] = lane
        try:
            f = {_s(k): _s(v) for k, v in raw.items()}
            self.order.append(lane)
            wait_ms = (time.time() - float(f["enq"])) * 1000
            self.waits.setdefault(lane, []).append(wait_ms)
            metrics.QUEUE_WAIT_MS.labels(lane=lane).observe(wait_ms)
            reply = await self._run(f)
            if reply is None:
                # tell the node it was dropped, so the game's slot in the stream comes back
                reply = {"game_id": f["game_id"], "seat": int(f["seat"]), "version": int(f["version"]), "stale": True}
            if f.get("reply_to"):
                body = {"r": json.dumps(reply, ensure_ascii=False, default=str)}
                await _retry(lambda: self.r.xadd(results_key(f["reply_to"]), body, maxlen=10_000, approximate=True))
            await _retry(lambda: self.r.xack(jobs_key(lane), GROUP, eid))
        except asyncio.CancelledError:
            raise
        except Exception as exc:                           # noqa: BLE001 -- not acked: another worker runs it again
            log.warning("job %s failed, left for a retry: %s", _s(eid), str(exc)[:200])
        finally:
            self.held.pop(eid, None)
            self.sem.release()

    async def _run(self, f: dict) -> dict | None:
        gid, seat, version = f["game_id"], int(f["seat"]), int(f["version"])
        row = await self.store.get_game(gid)
        if row is None or row["state"] not in ("running", "waiting"):
            self.stale += 1
            return None
        rows = await self.store.journal(gid)
        if len(rows) < version:
            self.stale += 1                               # written by an owner whose steps were never committed
            return None
        # A job may have been asked ahead of its turn (an open vote), so the world may have
        # moved since `version`. It is still wanted if the seat's turn is still open now; the
        # actor checks at commit time that the seat sees what it saw when it was asked.
        from ..runtime.runtime import Runtime
        table = Runtime.table_from_row(row, self.env_keys, hub=self.hub)      # under the game's own engine version
        Runtime.replay(table, [Record(r.actor, r.task, r.action, r.forced) for r in rows])
        t = table.turn
        opener = getattr(table.env, "open_turns", None)
        opened = ([t] if isinstance(t, Turn) else []) + (list(opener()) if callable(opener) and isinstance(t, Turn) else [])
        t = next((o for o in opened if o.actor == seat and o.task == f["task"]), None)
        if t is None:
            self.stale += 1                               # the world moved on; the actor no longer wants this
            return None
        mem = json.loads(f["memory"]) if f.get("memory") else None
        key = f.get("job_id") or f"{gid}:{seat}:{version}"
        if self._cancel_early.pop(key, None) is not None:
            self.stale += 1                               # cancelled while it sat in the queue
            return None
        job = AgentJob(gid, seat, version, t.task, table.env.observe(seat), table.env.action_space(seat, t.task),
                       tuple(mem) if mem else None,
                       make_player=lambda on_step: table.make_player(seat, task=t.task, version=version),
                       trace_ctx=tracing.from_carrier(json.loads(f["tp"])) if f.get("tp") else None,
                       stream=f.get("stream") == "1", job_id=key)
        loop = asyncio.get_running_loop()
        sender = None
        if job.stream:
            pieces: asyncio.Queue = asyncio.Queue()
            job.publish = lambda m: loop.call_soon_threadsafe(pieces.put_nowait, m)
            sender = asyncio.create_task(self._send_drafts(pieces, results_key(f["reply_to"])))
        self.running[key] = job
        try:
            res = await loop.run_in_executor(self.pool, run_agent_turn, job)
        finally:
            self.running.pop(key, None)
            if sender is not None:
                # the turn's own thread flushed its last piece before returning, and those
                # call_soon_threadsafe puts ran before this await resumed: the queue is complete
                pieces.put_nowait(None)
                await sender
        self.done += 1
        res.meta["remote"] = True
        res.meta["worker"] = self.consumer
        return {"game_id": gid, "seat": seat, "version": version, "action": res.action, "forced": res.forced,
                "memory": list(res.memory) if res.memory else None, "meta": res.meta, "error": res.error}

    async def _send_drafts(self, pieces: asyncio.Queue, key: str) -> None:
        """One job's draft pieces onto the results stream, strictly in order."""
        while True:
            m = await pieces.get()
            if m is None:
                return
            try:
                await self.r.xadd(key, {"d": json.dumps(m.to_dict(), ensure_ascii=False)}, maxlen=10_000,
                                  approximate=True)
                self.drafts_sent += 1
            except Exception:                              # noqa: BLE001 -- a lost piece leaves a gap the node refuses
                log.warning("could not send a draft piece")

    async def _cancel_loop(self) -> None:
        """Cancels from the nodes. Pub/sub, not a stream: a cancel nobody hears costs tokens, not correctness."""
        while not self._stop:
            ps = self.r.pubsub()
            try:
                await ps.subscribe(CANCEL_CHANNEL)
                while not self._stop:
                    msg = await ps.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if msg and msg.get("type") == "message":
                        self._on_cancel(json.loads(_s(msg["data"])))
            except asyncio.CancelledError:
                raise
            except Exception:                              # noqa: BLE001 -- redis blip: resubscribe
                await asyncio.sleep(0.5)
            finally:
                try:
                    await ps.aclose()
                except Exception:                          # noqa: BLE001
                    pass

    def _on_cancel(self, d: dict) -> None:
        key = d.get("job_id") or f"{d['game_id']}:{d['seat']}:{d['version']}"
        job = self.running.get(key)
        if job is not None:
            self.cancelled += 1
            job.cancel.cancel(d.get("reason") or "cancelled")
            return
        now = time.monotonic()
        self._cancel_early[key] = now
        for k in [k for k, t0 in self._cancel_early.items() if now - t0 > 600]:
            del self._cancel_early[k]

    async def _keepalive_loop(self) -> None:
        """Reset the idle time of every job running here (XCLAIM to ourselves, JUSTID), so
        only a dead worker's jobs ever get idle enough to be claimed by another."""
        while True:
            await asyncio.sleep(max(0.2, self.claim_idle_ms / 3000))
            by_lane: dict[str, list] = {}
            for eid, lane in list(self.held.items()):
                by_lane.setdefault(lane, []).append(eid)
            for lane, ids in by_lane.items():
                try:
                    await self.r.xclaim(jobs_key(lane), GROUP, self.consumer, 0, ids, justid=True)
                except Exception:                          # noqa: BLE001 -- redis blip: next round
                    pass

    async def _flush_loop(self) -> None:
        """Model-call records made here go to Postgres, as an API node's do (runtime.flush_calls)."""
        while True:
            await asyncio.sleep(1.0)
            await self.flush_calls()

    async def flush_calls(self) -> int:
        rows = [r.to_row() for r in self.hub.sink.drain(5000)] if self.hub is not None else []
        if rows:
            try:
                await self.store.record_calls(rows)
            except Exception:                              # noqa: BLE001 -- accounting must not stop turns
                log.exception("could not write llm_calls")
        return len(rows)

    async def _reclaim_loop(self) -> None:
        """Jobs a dead worker took and never acknowledged: claim them after they sit idle."""
        while True:
            await asyncio.sleep(max(0.2, self.claim_idle_ms / 4000))
            for lane in LANES:
                try:
                    res = await self.r.xautoclaim(jobs_key(lane), GROUP, self.consumer, self.claim_idle_ms,
                                                  start_id="0-0", count=20)
                except Exception:                          # noqa: BLE001
                    continue
                for eid, fields in (res[1] if res else []):
                    if not fields or eid in self.held:
                        continue
                    try:
                        info = await self.r.xpending_range(jobs_key(lane), GROUP, min=eid, max=eid, count=1)
                        tries = info[0]["times_delivered"] if info else 1
                        if tries > self.max_deliveries:
                            self.dead += 1
                            await self.r.xadd("llm:dead", {k: v for k, v in fields.items()}, maxlen=10_000)
                            await self.r.xack(jobs_key(lane), GROUP, eid)
                            f = {_s(k): _s(v) for k, v in fields.items()}
                            if f.get("reply_to"):          # its node asks again rather than wait for the deadline
                                body = {"r": json.dumps({"game_id": f["game_id"], "seat": int(f["seat"]),
                                                         "version": int(f["version"]), "dead": True})}
                                await self.r.xadd(results_key(f["reply_to"]), body, maxlen=10_000, approximate=True)
                            continue
                    except Exception:                      # noqa: BLE001 -- redis blip: it is claimed again later
                        continue
                    await self.sem.acquire()
                    t = asyncio.create_task(self._handle(lane, eid, fields))
                    self._tasks.add(t)
                    t.add_done_callback(self._tasks.discard)


async def _retry(op, tries: int = 6, delay: float = 0.2):
    """A Redis write that must not be lost to a blip: a few tries with backoff (about 6 s),
    then the error goes up (the job stays unacknowledged and is claimed and run again)."""
    for i in range(tries):
        try:
            return await op()
        except asyncio.CancelledError:
            raise
        except Exception:                                  # noqa: BLE001
            if i == tries - 1:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 2.0)


def _s(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


async def main() -> None:
    """python -m werewolf_harness.app.llm.worker (see worker.py)."""
    from ...server.http import read_env
    from ..config import get_settings
    from ..storage import Store
    from .gateway import Hub
    s = get_settings()
    olog.setup()
    tracing.setup("worker")
    if os.environ.get("WORKER_METRICS_PORT"):
        from prometheus_client import start_http_server
        start_http_server(int(os.environ["WORKER_METRICS_PORT"]))
    store = Store.connect(s.database_url)
    r = Redis.from_url(s.redis_url)
    env = read_env()
    hub = Hub(s.redis_url, env_keys=env, tpm=s.llm_tpm, default_tpm=s.llm_default_tpm)
    w = Worker(s, store, r, hub, concurrency=s.agent_workers, env_keys=env, store_calls=True)
    log.info("worker %s up", w.consumer)
    try:
        await w.run()
    finally:
        await w.flush_calls()
        await store.close()
        await r.aclose()
