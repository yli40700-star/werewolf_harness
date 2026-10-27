"""Model turns through Redis Streams, run by separate worker processes.

    node (actor)                    Redis                                 worker process(es)
    ------------                    -----                                 ------------------
    submit(job) ── XADD ──▶ llm:jobs:human | online | offline ──XREADGROUP──▶ run the turn on the
                                                                              view + spec in the job
    results loop ◀── XREAD ── llm:results:{node} ◀──────────── XADD ────────── then XACK the job
    call relay   ◀── XREADGROUP ── llm:calls ◀─────────────────── XADD ────── model-call records
      └─▶ Postgres llm_calls

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

A job carries everything the turn needs, computed by the actor: the seat's view
(already cut to what that seat may see), its action spec (the legal actions,
arena.ActionSpec.to_dict), its memory, the version, the task, and the seat's
player settings (model, cascade tier, budgets, guard, language:
runtime/table.py Table.seat_settings). The worker builds the player from those
(runtime/seats.py) and runs the ReAct loop; it reads no journal, replays
nothing and has no Postgres connection. Whether the answer is still wanted is
the actor's call, made where the world is: at commit it recomputes the seat's
view and spec, compares them with what it sent, and checks the action against
the live spec (runtime/actor.py `_drain`) -- a stale answer is dropped and the
turn asked again.

Jobs the actor no longer wants (deadline, turn gone, game stopped) are closed:
the node sets ``llm:closed:{job_id}`` and PUBLISHes on ``llm:cancel``. A worker
also closes a job once it has delivered its answer. A worker checks the key
before it starts a job, so a job it never heard cancelled -- it subscribed
later, or the job is a duplicate delivery of one already answered -- costs no
model call.

Model-call records (llm_calls: model, tokens, cache, latency, fallback reason)
are made by the gateway in whichever process ran the call. A worker XADDs them
to ``llm:calls`` in batches; every API process runs a CallRelay that reads the
stream through a consumer group and writes the rows to Postgres, acknowledging
a batch only after it is written. A failed write leaves the batch pending for
a retry (by this or any other API process); it never touches a turn.

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

from ...arena import ActionSpec, wire
from ...players import TokenBudget, TransportBudget
from ..config import Settings
from ..obs import logging as olog
from ..obs import metrics, tracing
from ..runtime.agents import AgentJob, AgentResult, DraftMsg, run_agent_turn
from ..runtime.load import LatencyWindow, LoadSample
from ..runtime.seats import build_player, guard_for

LANES = ("human", "online", "offline")
GROUP = "workers"
WORKERS_KEY = "llm:workers"           # hash: consumer -> {"capacity", "ts"}, refreshed by each worker
CANCEL_CHANNEL = "llm:cancel"
CLOSED_TTL_S = 3600                   # how long a closed job's marker outlives it (re-deliveries come sooner)
CALLS_KEY = "llm:calls"               # model-call records from the workers, for the API processes to store
CALLS_GROUP = "recorders"
log = olog.get("llm.queue")


def jobs_key(lane: str) -> str:
    return f"llm:jobs:{lane}"


def results_key(node: str) -> str:
    return f"llm:results:{node}"


def closed_key(job_id: str) -> str:
    """Set when a job is no longer to be run: cancelled by its node, or answered."""
    return f"llm:closed:{job_id}"


def job_payload(job: AgentJob) -> dict:
    """The parts of a job that are the turn itself, as stream fields (JSON strings; "" for
    a part the job lacks -- a worker drops such a job as one it cannot run)."""
    return {"view": json.dumps(wire.encode(job.view), ensure_ascii=False) if job.view is not None else "",
            "spec": json.dumps(job.spec.to_dict(), ensure_ascii=False) if job.spec is not None else "",
            "player": json.dumps(job.settings, ensure_ascii=False) if job.settings is not None else ""}


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

    def call_relay(self, store) -> CallRelay:
        """What writes the workers' model-call records to Postgres from this process (Runtime.start)."""
        return CallRelay(self.r, store, consumer=self.node)

    def _publish(self, job: AgentJob) -> None:
        try:
            payload = job_payload(job)
        except Exception:                                  # noqa: BLE001 -- a bug, not a blip: the deadline answers the turn
            log.exception("could not serialise the job for seat %s at %s; left to the turn's deadline",
                          job.seat, job.version)
            self.out.get(job.game_id, {}).pop((job.seat, job.version), None)
            return
        self.inflight += 1
        self.jobs[(job.game_id, job.seat, job.version)] = job
        fields = {"game_id": job.game_id, "seat": job.seat, "version": job.version, "task": job.task, **payload,
                  "memory": json.dumps(job.memory, ensure_ascii=False) if job.memory else "",
                  "reply_to": self.node, "enq": time.time(), "lane": job.lane,
                  "stream": "1" if job.stream else "0", "job_id": job.job_id,
                  "tp": json.dumps(tracing.to_carrier(job.trace_ctx)) if job.trace_ctx is not None else ""}
        asyncio.get_running_loop().create_task(self._send(job, fields))

    async def _send(self, job: AgentJob, fields: dict) -> None:
        """XADD the job, retrying through a Redis blip. A job that never reached the stream
        used to be noticed only by the turn's deadline (agent_timeout_s, then the default
        action). Gives up once the job is no longer wanted (cancelled: its slot is gone) --
        checked before every try, so a job cancelled while Redis was away is not sent
        when Redis comes back, where its turn may be committed by then (a worker no
        longer looks that up)."""
        delay = 0.2
        while True:
            if (self.jobs.get((job.game_id, job.seat, job.version)) is not job
                    or (job.seat, job.version) not in self.out.get(job.game_id, {})):
                return
            try:
                await self.r.xadd(jobs_key(job.lane), fields, maxlen=100_000, approximate=True)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:                       # noqa: BLE001
                self.send_retries += 1
                log.warning("could not queue a model job (%s): retrying in %.1fs", str(exc)[:120], delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 2.0)

    def cancel(self, job: AgentJob, reason: str) -> None:
        """Stop a job. One still held on this node (per-game cap) never goes out. One in
        the stream: mark it closed (llm:closed:{job_id}) and tell whichever worker runs it
        to stop it -- fire-and-forget; a worker that has not taken it yet finds the mark
        (or remembers the cancel) and drops it with a stale reply. If the mark is lost
        too, the job runs and its result is dropped by the actor as stale."""
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
        asyncio.get_running_loop().create_task(self._publish_cancel(job.job_id, msg))
        # Its answer is not wanted, so it no longer holds one of the game's places in the
        # stream. Waiting for the worker's reply to give it back would block the game's next
        # asks for agent_timeout_s whenever that reply never comes (Redis lost the job).
        self._settle(job.game_id, job.seat, job.version)

    async def _publish_cancel(self, job_id: str, msg: str) -> None:
        # The key first, and through a Redis blip: it is what a worker that never hears the
        # PUBLISH (subscribed later, reconnecting after the same blip) still finds before it
        # starts the job.
        try:
            await _retry(lambda: self.r.set(closed_key(job_id), "1", ex=CLOSED_TTL_S))
        except Exception:                                  # noqa: BLE001 -- a lost cancel costs tokens, not correctness
            log.warning("could not mark a cancelled job closed")
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
    """A worker process: pulls jobs by lane priority, runs them, replies, acks.

    Needs Redis and the model providers only: the turn (view, spec, memory, the seat's
    player settings) is in the job, and the model-call records go back over Redis."""

    def __init__(self, settings: Settings, redis: Redis, hub, *, consumer: str | None = None,
                 concurrency: int = 8, env_keys: dict | None = None, claim_idle_ms: int | None = None,
                 max_deliveries: int = 3):
        self.s, self.r, self.hub = settings, redis, hub
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
        self._unsent: list[dict] = []                    # model-call records not yet on llm:calls
        self.calls_sent = 0
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
        if self.hub is not None:
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
            if not reply.get("stale") and not (reply.get("meta") or {}).get("cancelled"):
                await self._close(_job_key(f))           # answered: a second delivery of it is not run again
            await _retry(lambda: self.r.xack(jobs_key(lane), GROUP, eid))
        except asyncio.CancelledError:
            raise
        except Exception as exc:                           # noqa: BLE001 -- not acked: another worker runs it again
            log.warning("job %s failed, left for a retry: %s", _s(eid), str(exc)[:200])
        finally:
            self.held.pop(eid, None)
            self.sem.release()

    async def _run(self, f: dict) -> dict | None:
        """Run one job's turn; None = dropped without asking the model (the node gets a stale reply)."""
        gid, seat, version = f["game_id"], int(f["seat"]), int(f["version"])
        key = _job_key(f)
        if self._cancel_early.pop(key, None) is not None:
            self.stale += 1                               # cancelled while it sat in the queue
            return None
        if not (f.get("view") and f.get("spec") and f.get("player")):
            self.stale += 1                               # nothing to run (a job from a node that predates payloads)
            log.warning("job %s carries no view/spec/player settings: dropped", key)
            return None
        try:
            closed = await self.r.exists(closed_key(key))
        except Exception:                                  # noqa: BLE001 -- unsure: run it; the actor drops what it does not want
            closed = False
        if closed:
            self.stale += 1                               # cancelled, or already answered (a re-delivery)
            return None
        view = wire.decode(json.loads(f["view"]))
        spec = ActionSpec.from_dict(json.loads(f["spec"]))
        st = json.loads(f["player"])
        mem = json.loads(f["memory"]) if f.get("memory") else None

        def make_player(on_step):
            # a fresh guard and fresh per-turn budgets, as a worker has always had (it used to
            # rebuild the table for every job); the game-wide token cap is the gateway's (GameBudget)
            return build_player(gid, seat, version, st, hub=self.hub, env_keys=self.env_keys,
                                guard=guard_for(st["guard"]), transport=TransportBudget(),
                                tokens=TokenBudget(max_tokens=st["max_tokens"]), on_step=on_step)
        job = AgentJob(gid, seat, version, f["task"], view, spec, tuple(mem) if mem else None,
                       make_player=make_player, settings=st,
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

    async def _close(self, key: str) -> None:
        try:
            await self.r.set(closed_key(key), "1", ex=CLOSED_TTL_S)
        except Exception:                                  # noqa: BLE001 -- a re-delivery would just run again
            log.warning("could not mark job %s closed", key)

    async def _flush_loop(self) -> None:
        """Model-call records made here go to llm:calls; an API process stores them (CallRelay)."""
        while True:
            await asyncio.sleep(1.0)
            await self.flush_calls()

    async def flush_calls(self) -> int:
        """Send the hub's model-call records to llm:calls. Ones Redis did not take are kept
        (bounded) and sent with the next flush. Never raises: accounting must not stop turns."""
        if self.hub is not None:
            self._unsent += [r.to_row() for r in self.hub.sink.drain(5000)]
        if not self._unsent:
            return 0
        rows, self._unsent = self._unsent, []
        sent = 0
        for i in range(0, len(rows), 500):
            part = rows[i:i + 500]
            try:
                await self.r.xadd(CALLS_KEY, {"c": json.dumps(part, ensure_ascii=False, default=str)},
                                  maxlen=100_000, approximate=True)
                sent += len(part)
            except Exception:                              # noqa: BLE001 -- redis blip: next flush
                self._unsent = (rows[i:] + self._unsent)[-50_000:]
                log.warning("could not send %d model-call records: kept for the next flush", len(rows) - i)
                break
        self.calls_sent += sent
        return sent

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


class CallRelay:
    """API side of the model-call records the workers make: ``llm:calls`` -> Postgres.

    Every API process runs one, all in one consumer group, so each batch is written by
    one of them. A batch is acknowledged only once its rows are in Postgres; a write
    that fails leaves it pending, and it is claimed again (by this process or another,
    e.g. after this one died mid-write) once it has sat `claim_idle_ms`. A batch that
    has failed `max_deliveries` times is dropped with a log line -- a row Postgres will
    never take must not be retried for ever. Runs beside the games; never stops a turn.
    """

    def __init__(self, redis: Redis, store, consumer: str, *, claim_idle_ms: int = 10_000,
                 max_deliveries: int = 5, batch: int = 50):
        self.r, self.store, self.consumer = redis, store, consumer
        self.claim_idle_ms, self.max_deliveries, self.batch = claim_idle_ms, max_deliveries, batch
        self.written = 0
        self.failed = 0
        self.dropped = 0
        self._group = False

    async def _ensure_group(self) -> None:
        if self._group:
            return
        try:
            await self.r.xgroup_create(CALLS_KEY, CALLS_GROUP, id="0", mkstream=True)
        except Exception as e:                             # noqa: BLE001 -- BUSYGROUP: already there
            if "BUSYGROUP" not in str(e):
                raise
        self._group = True

    async def run(self) -> None:
        while True:
            try:
                await self.drain_once(block=1000)
            except asyncio.CancelledError:
                raise
            except Exception as exc:                       # noqa: BLE001 -- redis blip (or it came back empty)
                self._group = False
                log.warning("relaying model-call records failed (%s): retrying", str(exc)[:200])
                await asyncio.sleep(1.0)

    async def drain_once(self, block: int | None = None) -> int:
        """Write what is waiting: batches left pending long enough, then new ones. Returns rows written."""
        await self._ensure_group()
        n = 0
        res = await self.r.xautoclaim(CALLS_KEY, CALLS_GROUP, self.consumer, self.claim_idle_ms,
                                      start_id="0-0", count=self.batch)
        n += await self._write(res[1] if res else [], again=True)
        got = await self.r.xreadgroup(CALLS_GROUP, self.consumer, {CALLS_KEY: ">"}, count=self.batch, block=block)
        for _stream, entries in got or []:
            n += await self._write(entries)
        return n

    async def _write(self, entries, again: bool = False) -> int:
        rows: list[dict] = []
        ids, junk = [], []
        for eid, f in entries:
            try:
                rows += json.loads(_s(f[b"c"] if b"c" in f else f["c"]))
                ids.append(eid)
            except (KeyError, TypeError, ValueError):
                junk.append(eid)                           # not a batch of records (or trimmed away): nothing to keep
        if junk:
            await self.r.xack(CALLS_KEY, CALLS_GROUP, *junk)
        if not ids:
            return 0
        try:
            await self.store.record_calls(rows)
        except Exception:                                  # noqa: BLE001 -- accounting must not stop games
            self.failed += 1
            log.exception("could not write %d llm_calls rows: left pending for a retry", len(rows))
            if again:
                await self._drop_hopeless(ids)
            return 0
        await self.r.xack(CALLS_KEY, CALLS_GROUP, *ids)
        self.written += len(rows)
        return len(rows)

    async def _drop_hopeless(self, ids: list) -> None:
        for eid in ids:
            info = await self.r.xpending_range(CALLS_KEY, CALLS_GROUP, min=eid, max=eid, count=1)
            if info and info[0]["times_delivered"] >= self.max_deliveries:
                self.dropped += 1
                log.error("dropping model-call batch %s after %d failed writes", _s(eid), info[0]["times_delivered"])
                await self.r.xack(CALLS_KEY, CALLS_GROUP, eid)


def _job_key(f: dict) -> str:
    return f.get("job_id") or f"{f['game_id']}:{f['seat']}:{f['version']}"


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
    from .gateway import Hub
    s = get_settings()
    olog.setup()
    tracing.setup("worker")
    if os.environ.get("WORKER_METRICS_PORT"):
        from prometheus_client import start_http_server
        start_http_server(int(os.environ["WORKER_METRICS_PORT"]))
    r = Redis.from_url(s.redis_url)
    env = read_env()
    hub = Hub(s.redis_url, env_keys=env, tpm=s.llm_tpm, default_tpm=s.llm_default_tpm)
    w = Worker(s, r, hub, concurrency=s.agent_workers, env_keys=env)
    log.info("worker %s up", w.consumer)
    try:
        await w.run()
    finally:
        await w.flush_calls()
        await r.aclose()
