"""One game, one asyncio task, one mailbox.

Only this task changes the Table. Everything else -- an HTTP request with a
person's move, a model result coming back from a worker, a clock firing --
becomes a message in the mailbox, and the loop takes them one at a time. That
is the whole concurrency story inside a game: a single writer, so no locks.

Every step is committed journal-first:

    ① INSERT the journal row (+ the model seat's memory) with our fencing token
    ② apply it to the in-memory world, version += 1
    ③ publish the events

Crash between ① and ②: recovery replays the row, so the step happened. Crash
before ①: the step never happened and the turn is asked again. Crash after ②
but before ③: events are derived from the journal and regenerated on replay;
a reconnecting client catches up from its last event seq. See docs/adr/0002.

Streaming speech (docs/adr/0009): while a model seat generates a speech, its
text arrives here as DraftMsg pieces and is shown as a draft. A draft lives and
dies with the ask it belongs to: it is committed in step ③ of the commit that
journals that ask's answer -- with the journal's text -- and retracted on every
other way an ask can end (guard retry, miss, deadline, dropped turn, stop,
crash). Ending an ask early also cancels its model call.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.exc import DBAPIError

from ...arena import Terminal, Turn
from ...arena.driver import fingerprint, spec_key
from ...players.human import Pending
from ...players.llm.streaming import SPEECH_TOOLS
from ...server.service import _in_plain_words
from ..obs import logging as olog
from ..obs import tracing
from ..storage import DuplicateStep, Fenced, Store
from .agents import AgentJob, AgentResult, DraftMsg
from .table import DRAFT_AUDIENCE, Draft, Table
from .timers import Timeout, TimingWheel

log = olog.get("runtime.actor")


# ---------------------------------------------------------------- messages
@dataclass
class HumanAction:
    seat: int
    seq: int
    action: Any
    request_id: str | None
    reply: asyncio.Future


@dataclass
class TimerFired:
    version: int
    kind: str                    # human | agent


@dataclass
class Start:
    reply: asyncio.Future | None = None


@dataclass
class Stop:
    reason: str


@dataclass
class ActorHooks:
    on_finish: Callable[[Table], Any] | None = None      # leaderboard, metrics
    on_commit: Callable[[Table, float], Any] | None = None
    on_stale: Callable[[Table], Any] | None = None


@dataclass
class ActorStats:
    commits: int = 0
    commit_ms_total: float = 0.0
    db_retries: int = 0
    extra: dict = field(default_factory=dict)


class GameActor:
    def __init__(self, table: Table, *, store: Store, executor, wheel: TimingWheel, token: int,
                 agent_timeout_s: float, hooks: ActorHooks | None = None):
        self.t = table
        self.store = store
        self.executor = executor
        self.wheel = wheel
        self.token = token
        self.agent_timeout_s = agent_timeout_s
        self.hooks = hooks or ActorHooks()
        self.mailbox: asyncio.Queue = asyncio.Queue()
        self.bell = asyncio.Condition()
        self.task: asyncio.Task | None = None
        self.timer: Timeout | None = None
        self.stopped: str | None = None           # why the loop ended (fenced, shutdown, db, finished)
        self.stats = ActorStats()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._turn_span = None
        # Model seats asked and not yet committed, one per seat. Besides the seat whose
        # turn it is, every seat whose turn the engine declares open (a sealed vote, a
        # review round, the seer and the guard at night) is asked at once; answers are
        # committed one by one in journal order, each only if the seat still sees what it
        # saw when it was asked. A model job is stateless (memory travels with it), so an
        # answer that went stale is simply not used -- there is nothing to undo.
        self.asks: dict[int, dict] = {}

    @property
    def id(self) -> str:
        return self.t.id

    @property
    def version(self) -> int:
        return len(self.t.journal)

    # ------------------------------------------------------------ lifecycle
    def launch(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.task = self._loop.create_task(self._run(), name=f"game-{self.id}")

    async def stop(self, reason: str = "shutdown") -> None:
        if self.task and not self.task.done():
            self.mailbox.put_nowait(Stop(reason))
            try:
                await asyncio.wait_for(self.task, 5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self.task.cancel()
        self._cancel_timer()

    def post(self, msg: Any) -> None:
        """Thread-safe: executors and timers call this from wherever they run."""
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._enqueue, msg)

    def _enqueue(self, msg: Any) -> None:
        if self.stopped is not None and isinstance(msg, AgentResult) and msg.meta.get("cancelled"):
            # The turn we cancelled on the way out: nobody reads the mailbox any more,
            # but what the cancel saved still belongs in the ledger (adr/0009).
            self.t.note_cancelled(msg.meta.get("tokens_saved"))
            return
        self.mailbox.put_nowait(msg)

    async def _run(self) -> None:
        tok = olog.bind(game_id=self.id)
        try:
            if self.t.state == "running":
                await self._prompt()
            while self.stopped is None:
                msg = await self.mailbox.get()
                await self._handle(msg)
        except Exception as exc:                          # noqa: BLE001 -- a crash is recorded, not swallowed
            log.exception("actor crashed")
            self.t.state, self.t.error = "crashed", f"{type(exc).__name__}: {exc}"
            self.t.acting = None
            await self._safe_set_state()
            await self._notify(self.t.emit("error", message=self.t.error))
            self.stopped = "crashed"
        finally:
            self._cancel_timer()
            self._close_all(self.stopped or "stopped")
            olog.unbind(tok)

    async def _handle(self, msg: Any) -> None:
        if isinstance(msg, Stop):
            self.stopped = msg.reason
        elif isinstance(msg, Start):
            await self._start()
            if msg.reply and not msg.reply.done():
                msg.reply.set_result(self.t.summary())
        elif isinstance(msg, HumanAction):
            res = await self._human(msg)
            if not msg.reply.done():
                msg.reply.set_result(res)
        elif isinstance(msg, DraftMsg):
            await self._draft_msg(msg)
        elif isinstance(msg, AgentResult):
            await self._agent_result(msg)
        elif isinstance(msg, TimerFired):
            await self._timer(msg)

    # ------------------------------------------------------------ public API (called from request handlers)
    async def submit(self, seat: int, seq: int, action: Any, request_id: str | None = None) -> dict:
        fut = asyncio.get_running_loop().create_future()
        self.mailbox.put_nowait(HumanAction(seat, seq, action, request_id, fut))
        return await asyncio.wait_for(fut, 30)

    async def start(self) -> dict:
        fut = asyncio.get_running_loop().create_future()
        self.mailbox.put_nowait(Start(fut))
        return await asyncio.wait_for(fut, 30)

    async def wait_events(self, after: int, timeout: float = 20.0) -> list:
        async with self.bell:
            if len(self.t.events) <= after:
                try:
                    await asyncio.wait_for(self.bell.wait(), timeout)
                except asyncio.TimeoutError:
                    pass
        return self.t.events[after:]

    async def wait_any(self, after: int, dlast: int, timeout: float = 20.0) -> None:
        """Wait until there is an event at or past `after` or a draft event past `dlast`."""
        async with self.bell:
            if len(self.t.events) <= after and self.t.dseq - 1 <= dlast:
                try:
                    await asyncio.wait_for(self.bell.wait(), timeout)
                except asyncio.TimeoutError:
                    pass

    # ------------------------------------------------------------ turns
    async def _start(self) -> None:
        if self.t.state != "waiting":
            return
        self.t.state = "running"
        await self._safe_set_state()
        await self._notify(self.t.emit("phase", phase="start", seed=self.t.seed))
        await self._prompt()

    async def _prompt(self) -> None:
        """Ask whoever's turn it is. Never waits for the answer."""
        turn = self.t.turn
        if isinstance(turn, Terminal):
            await self._finish(turn)
            return
        assert isinstance(turn, Turn)
        view = self.t.env.observe(turn.actor)
        spec = self.t.env.action_space(turn.actor, turn.task)
        human = turn.actor in self.t.human_seats
        self.t.acting = {"actor": turn.actor, "task": turn.task, "since": time.time(), "human": human,
                         "model": self.t.seat_models.get(turn.actor, "真人"), "steps": []}
        ev = self.t.phase_event()
        v = self.version
        self._end_turn_span()
        self._turn_span = tracing.get_tracer().start_span(
            "turn", attributes={"game.id": self.id, "game.seat": turn.actor, "game.task": turn.task,
                                "game.version": v, "game.seat_kind": "human" if human else "agent"})
        turn_ctx = tracing.trace.set_span_in_context(self._turn_span)
        if human:
            secs = self.t.cfg.turn_seconds
            deadline = None if secs is None else time.time() + secs
            self.t.pending = Pending(v, turn.task, view, spec, deadline)
            self.t.pending_seat = turn.actor
            if secs is not None:
                self._arm(secs, TimerFired(v, "human"))
        else:
            self._arm(self.agent_timeout_s, TimerFired(v, "agent"))
        opened = self._open_turns(turn)
        for seat in [s for s, a in self.asks.items() if a["turn"] not in opened]:
            self._drop_ask(seat, "turn_gone")       # its turn is gone (a wolf ended the day): the answer is dropped
            self.t.stale_results += 1
        for t in opened:
            if t.actor in self.t.human_seats:
                continue
            a = self.asks.get(t.actor)
            if a is None or a["turn"] != t:
                if a is not None:
                    self._drop_ask(t.actor, "turn_gone")
                self._ask(t, turn_ctx, ahead=t != turn)
        await self._notify(ev)
        await self._drain()

    def _open_turns(self, turn: Turn) -> list[Turn]:
        opener = getattr(self.t.env, "open_turns", None)
        opened = (opener() if callable(opener) and self.t.cfg.parallel else []) or [turn]
        return opened if opened[0] == turn else [turn]

    def _ask(self, t: Turn, turn_ctx, ahead: bool) -> None:
        view = self.t.env.observe(t.actor)
        spec = self.t.env.action_space(t.actor, t.task)
        v = self.version
        job = AgentJob(self.id, t.actor, v, t.task, view, spec, self.t.memory.get(t.actor),
                       make_player=lambda on_step, s=t.actor, task=t.task, v=v:
                       self.t.make_player(s, on_step=on_step, task=task, version=v),
                       on_step=self._step_from_thread,
                       lane=self.t.cfg.lane or ("human" if self.t.human_seats else "online"),
                       trace_ctx=turn_ctx, stream=t.task in DRAFT_AUDIENCE)
        self.asks[t.actor] = {"turn": t, "version": v, "view": fingerprint(view), "spec": spec_key(spec),
                              "result": None, "since": time.time(), "job": job,
                              # drafts of this ask, keyed (run, attempt); runs superseded by a re-run
                              "drafts": {}, "closed": set(), "run": None, "dead_runs": set()}
        if ahead:
            self.t.asked_ahead += 1
        self.executor.submit(job, self.post)

    async def _drain(self) -> None:
        """Commit the current turn if its answer is already in."""
        turn = self.t.turn
        if self.stopped or not isinstance(turn, Turn) or turn.actor in self.t.human_seats:
            return
        a = self.asks.get(turn.actor)
        if a is None or a["turn"] != turn or a["result"] is None:
            return
        r: AgentResult = a["result"]
        view = self.t.env.observe(turn.actor)
        spec = self.t.env.action_space(turn.actor, turn.task)
        if fingerprint(view) != a["view"] or spec_key(spec) != a["spec"] or spec.check(r.action) is not None:
            self.t.misses += 1                     # asked too early after all: ask again, now
            self._drop_ask(turn.actor, "stale", cancel=False)
            await self._notify(None)
            self._ask(turn, tracing.trace.set_span_in_context(self._turn_span) if self._turn_span else None, ahead=False)
            return
        # The draft that answer streamed is the one to commit; any other is retracted now.
        said = self._said(r)
        keep = a["drafts"].pop((r.meta.get("run"), r.meta.get("draft_attempt")), None) if said is not None else None
        self._drop_ask(turn.actor, "not_said", cancel=False)
        if r.meta.get("remote") and r.meta.get("trace"):
            for st in r.meta["trace"]:          # a worker in another process cannot stream steps; they come with the result
                self.t.traces.setdefault(r.seat, []).append({**st, "round": getattr(self.t.env, "_round", 0)})
            self.t.traces[r.seat] = self.t.traces[r.seat][-40:]
        self.t.note_cascade(r.meta)                # counted once, for the answer that is committed
        agent_states = [(r.seat, r.memory[0], r.memory[1])] if r.memory else []
        await self._commit(r.action, forced=r.forced, agent_states=agent_states, memory=(r.seat, r.memory),
                           draft=keep)

    @staticmethod
    def _said(r: AgentResult) -> str | None:
        """The speech this answer journals, if it is the model's own speech (not a fallback)."""
        act = r.action
        if r.forced or not isinstance(act, dict) or act.get("name") not in SPEECH_TOOLS:
            return None
        content = (act.get("args") or {}).get("content")
        return content if isinstance(content, str) else None

    async def _human(self, m: HumanAction) -> dict:
        if m.seat not in self.t.human_seats:
            return {"status": "not_a_human_seat"}
        if m.request_id and m.request_id in self.t.request_ids:
            seq = self.t.request_ids[m.request_id]
            return {"status": "duplicate", "seq": seq, "action": self.t.journal[seq].action}
        if m.seq < self.version and self.t.journal[m.seq].actor == m.seat:
            return {"status": "duplicate", "seq": m.seq, "action": self.t.journal[m.seq].action}
        p = self.t.pending
        if p is None or self.t.pending_seat != m.seat:
            return {"status": "not_your_turn"}
        if m.seq != p.seq:
            return {"status": "stale", "expected": p.seq}
        reason = p.spec.check(m.action)
        if reason:
            p.feedback = reason
            return {"status": "illegal", "reason": reason}
        ok = await self._commit(m.action, forced=False, request_id=m.request_id or f"h-{uuid.uuid4().hex[:12]}")
        return {"status": "accepted"} if ok else {"status": "unavailable", "reason": self.stopped}

    async def _agent_result(self, r: AgentResult) -> None:
        if r.meta.get("cancelled"):
            self.t.note_cancelled(r.meta.get("tokens_saved"))      # counted whenever it lands
        a = self.asks.get(r.seat)
        if self.stopped or a is None or a["version"] != r.version or a["result"] is not None:
            # the world moved on (deadline fired, game recovered, the turn was dropped, a duplicate delivery)
            self.t.stale_results += 1
            if self.hooks.on_stale:
                self.hooks.on_stale(self.t)
            return
        if r.error:
            self._cancel_timer()
            self.t.state, self.t.error = "aborted", _in_plain_words(r.error)
            self.t.acting = None
            self._close_all("aborted")
            await self._safe_set_state()
            await self._notify(self.t.emit("error", message=self.t.error))
            self.stopped = "aborted"
            return
        if r.meta.get("cancelled"):
            return                                 # a cancel we did not ask for (worker shutting down): wait for the clock
        a["result"] = r
        await self._drain()

    async def _timer(self, m: TimerFired) -> None:
        if m.version != self.version or not isinstance(self.t.turn, Turn):
            return
        spec = self.t.env.action_space(self.t.turn.actor, self.t.turn.task)
        self._drop_ask(self.t.turn.actor, "timeout")     # retract what it showed, and stop it costing tokens
        await self._commit(spec.fallback, forced=True)

    # ------------------------------------------------------------ drafts
    async def _draft_msg(self, m: DraftMsg) -> None:
        """A piece of a speech being generated. Shown only while its ask is live."""
        a = self.asks.get(m.seat)
        if self.stopped or a is None or a["version"] != m.version or a["result"] is not None \
                or m.run in a["dead_runs"]:
            return
        if a["run"] != m.run:
            if a["run"] is not None:
                # The same job running again (claimed after its worker went quiet): the
                # earlier run's text is void, and anything more it sends is ignored.
                for key in [k for k in a["drafts"] if k[0] == a["run"]]:
                    self.t.draft_retract(a["drafts"].pop(key), "rerun")
                a["dead_runs"].add(a["run"])
            a["run"] = m.run
        key = (m.run, m.attempt)
        d = a["drafts"].get(key)
        if m.kind == "abandon":
            a["closed"].add(key)
            if d is not None:
                self.t.draft_retract(a["drafts"].pop(key), m.text or "retry")
                await self._notify(None)
            return
        if key in a["closed"]:
            return
        if d is None:
            if m.offset != 0:
                return                             # we missed its start: show nothing rather than a torn text
            d = a["drafts"][key] = self.t.draft_open(m.seat, a["version"], a["turn"].task, m.run, m.attempt,
                                                     a["since"])
        if self.t.draft_delta(d, m.offset, m.text):
            await self._notify(None)

    def _drop_ask(self, seat: int, reason: str, cancel: bool = True) -> None:
        """End an ask: retract every draft it showed and, if it is still running, cancel it."""
        a = self.asks.pop(seat, None)
        if a is None:
            return
        for d in list(a["drafts"].values()):
            self.t.draft_retract(d, reason)
        a["drafts"].clear()
        if cancel and a["result"] is None:
            stop = getattr(self.executor, "cancel", None)
            if stop is not None:
                try:
                    stop(a["job"], reason)
                except Exception:                          # noqa: BLE001 -- best effort: the result is ignored anyway
                    log.exception("cancel failed")

    def _close_all(self, reason: str) -> None:
        """The game stopped driving here: no draft stays open, no model call keeps running."""
        for seat in list(self.asks):
            self._drop_ask(seat, reason)
        for d in list(self.t.drafts.values()):            # one whose commit never landed (fenced, db)
            self.t.draft_retract(d, reason)
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.create_task(self._notify(None))
            except RuntimeError:
                pass

    # ------------------------------------------------------------ commit
    async def _commit(self, action: Any, *, forced: bool, request_id: str | None = None,
                      agent_states: list | None = None, memory: tuple | None = None,
                      draft: Draft | None = None) -> bool:
        turn = self.t.turn
        assert isinstance(turn, Turn)
        t0 = time.monotonic()
        delay = 0.2
        span = tracing.get_tracer().start_span(
            "commit", context=tracing.trace.set_span_in_context(self._turn_span) if self._turn_span else None,
            attributes={"game.id": self.id, "game.seq": self.version, "game.forced": forced})
        try:
            ok = await self._commit_inner(action, forced, request_id, agent_states, memory, turn, t0, delay, draft)
        finally:
            span.end()
            self._end_turn_span()
        if ok:
            await self._prompt()          # the next turn gets its own span
        return ok

    def _end_turn_span(self) -> None:
        if self._turn_span is not None:
            self._turn_span.end()
            self._turn_span = None

    async def _commit_inner(self, action, forced, request_id, agent_states, memory, turn, t0, delay,
                            draft: Draft | None = None) -> bool:
        for attempt in range(7):
            try:
                await self.store.append(self.id, token=self.token, seq=self.version, actor=turn.actor, task=turn.task,
                                        action=action, forced=forced, request_id=request_id,
                                        agent_states=agent_states or [])
                break
            except Fenced:
                # someone else owns this game now; our in-memory world must not advance
                log.warning("fenced: stopping")
                self.stopped = "fenced"
                return False
            except DuplicateStep:
                # our seq is already taken: another incarnation wrote it. Same answer as fenced.
                log.warning("duplicate step at %s: stopping", self.version)
                self.stopped = "fenced"
                return False
            except (DBAPIError, OSError, asyncio.TimeoutError, ConnectionError):
                self.stats.db_retries += 1
                if attempt == 6:
                    log.error("database unavailable: releasing the game for recovery")
                    self.stopped = "db"
                    return False
                await asyncio.sleep(delay)
                delay = min(delay * 2, 3.0)
        # ② apply
        self._cancel_timer()
        if request_id:
            self.t.request_ids[request_id] = self.version
        if memory and memory[1] is not None:
            self.t.memory[memory[0]] = memory[1]
        self.t.pending, self.t.pending_seat, self.t.acting = None, None, None
        record, news = self.t.apply(action, forced)
        ms = (time.monotonic() - t0) * 1000
        self.stats.commits += 1
        self.stats.commit_ms_total += ms
        if self.hooks.on_commit:
            self.hooks.on_commit(self.t, ms)
        # ③ publish -- the draft first, so a client sees it become the speech before the turn lands
        if draft is not None:
            said = (record.action.get("args") or {}).get("content") if isinstance(record.action, dict) else None
            if isinstance(said, str):
                self.t.draft_commit(draft, said, len(self.t.journal))
            else:
                self.t.draft_retract(draft, "not_said")
        self.t.step_events(record, news)
        await self._notify(None)
        return True

    async def _finish(self, term: Terminal) -> None:
        self.t.acting = None
        self.t.outcome = term.outcome
        self.t.state = "over"
        await self._safe_set_state()
        await self._notify(self.t.emit("end", outcome=term.outcome, health=self.t.health()))
        if self.hooks.on_finish:
            try:
                res = self.hooks.on_finish(self.t)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:                              # noqa: BLE001
                log.exception("on_finish hook failed")
        self.stopped = "finished"

    # ------------------------------------------------------------ helpers
    def _arm(self, delay: float, msg: TimerFired) -> None:
        self._cancel_timer()
        self.timer = self.wheel.schedule(delay, lambda: self.mailbox.put_nowait(msg))

    def _cancel_timer(self) -> None:
        if self.timer:
            self.timer.cancel()
            self.timer = None

    def _step_from_thread(self, seat: int, step) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._on_step, seat, step)

    def _on_step(self, seat: int, step) -> None:
        row = {"n": step.step, "action": step.action, "args": step.args,
               "observation": (step.observation or "")[:300], "thought": (step.thought or "")[:300],
               "blocked": step.block_reason, "ms": step.latency_ms, "round": getattr(self.t.env, "_round", 0)}
        self.t.traces.setdefault(seat, []).append(row)
        self.t.traces[seat] = self.t.traces[seat][-40:]
        if self.t.acting and self.t.acting.get("actor") == seat:
            self.t.acting.setdefault("steps", []).append(step.action)

    async def _notify(self, _ev) -> None:
        async with self.bell:
            self.bell.notify_all()

    async def _safe_set_state(self) -> None:
        try:
            await self.store.set_state(self.id, self.t.state, outcome=self.t.outcome, error=self.t.error)
        except Exception:                                  # noqa: BLE001 -- state is re-derivable; never kill the loop for it
            logging.getLogger(__name__).exception("set_state failed")
