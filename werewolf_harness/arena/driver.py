"""The loop. Environment-agnostic: grep this file for any game word and you will find none."""
from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .protocol import ActionSpec, Environment, Illegal, Player, Terminal, Turn  # noqa: F401


@dataclass
class Record:
    actor: str
    task: str
    action: Any
    forced: bool = False       # True when the actor exhausted retries and the fallback was used


@dataclass
class Health:
    """What a batch needs before its numbers mean anything.

    A run whose turns were mostly filled in by the harness is not a record of
    what the models did. This is computed for every episode, not on request,
    because the failure mode it guards against is a summary that never looked.
    """
    turns: int = 0
    forced: int = 0
    replays: int = 0

    @property
    def forced_rate(self) -> float:
        return self.forced / self.turns if self.turns else 0.0

    @property
    def usable(self) -> bool:
        return self.forced_rate <= 0.05

    def to_dict(self) -> dict:
        return {"turns": self.turns, "forced": self.forced, "replays": self.replays,
                "forced_rate": round(self.forced_rate, 4), "usable": self.usable}


@dataclass
class Episode:
    journal: list[Record] = field(default_factory=list)
    outcome: dict[str, Any] | None = None
    health: Health = field(default_factory=Health)
    # one row per journal record: when it was asked, decided and committed (seconds
    # since the start of the episode), and whether it was decided ahead of its turn
    timings: list[dict] = field(default_factory=list)
    misses: int = 0            # decided ahead, then thrown away because its turn changed


class StepLedger:
    """Idempotent state advance: the same submission applied twice advances once.

    The world is a function of the journal, so a duplicate step is not a
    duplicate write -- it is a *different world*. In-process that never happens;
    over a network it happens constantly (a retried POST, a human double-click,
    a queue redelivering a message). The key is everything the framework knows
    about a turn -- game, position in the journal, actor, task -- and a repeat
    returns the first outcome instead of touching the environment.
    """

    def __init__(self, game: str = ""):
        self.game = game
        self.seq = 0
        self.replays = 0
        self._done: dict[str, tuple[Record, Turn | Terminal]] = {}

    def key(self, turn: Turn, seq: int | None = None) -> str:
        return f"{self.game}:{self.seq if seq is None else seq}:{turn.actor}:{turn.task}"

    def apply(self, env: Environment, turn: Turn, action: Any, *,
              seq: int | None = None, forced: bool = False) -> tuple[Record, Turn | Terminal]:
        k = self.key(turn, seq)
        if k in self._done:
            self.replays += 1
            return self._done[k]
        record = Record(turn.actor, turn.task, action, forced)
        nxt = env.step(turn.actor, turn.task, action)
        self._done[k] = (record, nxt)
        self.seq += 1
        return record, nxt


def run(env: Environment, players: dict[str, Player], seed: int, max_retries: int = 3,
        ledger: StepLedger | None = None, parallel: int = 1) -> Episode:
    """Play one episode.

    parallel=1 asks one turn at a time. parallel>1 also asks, in the background, every
    turn the environment declares open (``open_turns``, optional): several seats think
    at once, and their answers are committed one by one in the order ``step`` asks for
    them. Each answer is checked before it is committed -- the actor must still see the
    same view and have the same legal actions as when it was asked -- so the journal is
    the one the sequential loop would have written for those decisions, and replay
    does not know the difference.
    """
    if parallel > 1 and callable(getattr(env, "open_turns", None)):
        return _run_parallel(env, players, seed, max_retries, ledger or StepLedger(), parallel)
    ep = Episode()
    ledger = ledger or StepLedger()
    t0 = time.monotonic()
    turn = env.reset(seed)
    while isinstance(turn, Turn):
        asked = time.monotonic() - t0
        view = env.observe(turn.actor)
        spec = env.action_space(turn.actor, turn.task)
        action, forced = _elicit(players[turn.actor], view, turn.task, spec, max_retries)
        done = time.monotonic() - t0
        record, nxt = ledger.apply(env, turn, action, forced=forced)
        ep.journal.append(record)
        ep.timings.append({"seq": len(ep.journal) - 1, "actor": turn.actor, "task": turn.task, "asked": round(asked, 3),
                           "done": round(done, 3), "committed": round(time.monotonic() - t0, 3), "ahead": False})
        turn = nxt
    _flush(players)
    ep.outcome = turn.outcome
    ep.health = Health(turns=len(ep.journal), forced=sum(r.forced for r in ep.journal),
                       replays=ledger.replays)
    return ep


# ── parallel decide, serial commit ─────────────────────────────────────
def fingerprint(view: Any) -> str:
    """What an actor saw, as one comparable string."""
    return json.dumps(view, sort_keys=True, ensure_ascii=False, default=str)


def spec_key(spec: ActionSpec) -> str:
    """What an actor was allowed to do, as one comparable string. ``validate`` callables
    cannot be compared, which is why the answer is also re-checked against the live spec."""
    tools = [(t.name, sorted((k, p.type, p.choices, p.required, p.max_len, p.range, p.non_empty)
                             for k, p in t.params.items())) for t in spec.tools]
    return json.dumps([spec.kind, list(spec.options), spec.max_len, sorted(spec.fields.items()), tools, spec.fallback],
                      sort_keys=True, ensure_ascii=False, default=str)


@dataclass
class _Ask:
    turn: Turn
    view_fp: str
    spec_fp: str
    future: Future
    asked: float
    ahead: bool
    done: list = field(default_factory=list)   # filled by the worker: [seconds]


def _run_parallel(env, players, seed, max_retries, ledger: StepLedger, parallel: int) -> Episode:
    ep = Episode()
    t0 = time.monotonic()
    clock = lambda: time.monotonic() - t0        # noqa: E731
    pool = ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="seat")
    asks: dict[Turn, _Ask] = {}

    def ask(t: Turn, ahead: bool) -> None:
        # observe and action_space run here, on the one thread that also steps the world
        view = env.observe(t.actor)
        spec = env.action_space(t.actor, t.task)
        a = _Ask(t, fingerprint(view), spec_key(spec), None, clock(), ahead)    # type: ignore[arg-type]

        def work():
            out = _elicit(players[t.actor], view, t.task, spec, max_retries)
            a.done.append(clock())
            return out
        a.future = pool.submit(work)
        asks[t] = a

    def drop(t: Turn) -> None:
        """An answer whose turn changed: wait for it, then undo what it did to the seat."""
        a = asks.pop(t)
        try:
            a.future.result()
        finally:
            ep.misses += 1
            undo = getattr(players[t.actor], "discard_last", None)
            if callable(undo):
                undo()

    try:
        turn = env.reset(seed)
        while isinstance(turn, Turn):
            opened = env.open_turns() or [turn]
            if opened[0] != turn:                 # the environment's list must start where step() is
                opened = [turn]
            for t in [t for t in asks if t not in opened]:
                drop(t)
            busy = {t.actor for t in asks}
            for t in opened:
                if t not in asks and t.actor not in busy:
                    ask(t, ahead=t != turn)
                    busy.add(t.actor)
            if turn not in asks:                  # its actor was busy with a turn that is no longer open
                for t in [t for t in asks if t.actor == turn.actor]:
                    drop(t)
                ask(turn, ahead=False)
            a = asks.pop(turn)
            action, forced = a.future.result()
            view = env.observe(turn.actor)
            spec = env.action_space(turn.actor, turn.task)
            if fingerprint(view) != a.view_fp or spec_key(spec) != a.spec_fp or spec.check(action) is not None:
                # asked too early after all: ask again, now, one at a time
                ep.misses += 1
                undo = getattr(players[turn.actor], "discard_last", None)
                if callable(undo):
                    undo()
                a = _Ask(turn, "", "", None, clock(), False)    # type: ignore[arg-type]
                action, forced = _elicit(players[turn.actor], view, turn.task, spec, max_retries)
                a.done.append(clock())
            record, nxt = ledger.apply(env, turn, action, forced=forced)
            ep.journal.append(record)
            ep.timings.append({"seq": len(ep.journal) - 1, "actor": turn.actor, "task": turn.task,
                               "asked": round(a.asked, 3), "done": round(a.done[0] if a.done else clock(), 3),
                               "committed": round(clock(), 3), "ahead": a.ahead})
            turn = nxt
        for t in list(asks):
            drop(t)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    _flush(players)
    ep.outcome = turn.outcome
    ep.health = Health(turns=len(ep.journal), forced=sum(r.forced for r in ep.journal),
                       replays=ledger.replays)
    return ep


def _flush(players: dict) -> None:
    """Let every seat finish what it does after its turn (an LLM seat's note to itself)."""
    for p in players.values():
        f = getattr(p, "flush", None)
        if callable(f):
            f()


def _elicit(player: Player, view: dict, task: str, spec: ActionSpec, max_retries: int) -> tuple[Any, bool]:
    """Ask until the action is legal; (action, forced).

    forced is True when the driver fell back to spec.fallback, and also when the player
    itself reports that the legal action it returned was its own fallback
    (``last_forced``: an LLM seat whose model never produced a usable answer, a human
    who timed out). Without the second half the health line counted 0 forced turns
    while the seats were filling in moves the model never chose. Read right after
    act(), on the same thread, so the parallel path (which calls this per seat on the
    pool) sees the flag of this turn and not another one's.
    """
    feedback = None
    for _ in range(max_retries):
        action = player.act(view, task, spec, feedback)
        reason = spec.check(action)
        if reason is None:
            return action, bool(getattr(player, "last_forced", False))
        feedback = reason                      # the reason goes back to the actor
    return spec.fallback, True


def replay(env: Environment, seed: int, journal: list[Record], upto: int | None = None) -> Turn | Terminal:
    """world == f(seed, journal). Feed the recorded actions back; end up at the same state."""
    turn = env.reset(seed)
    for rec in journal[:upto]:
        assert isinstance(turn, Turn) and (turn.actor, turn.task) == (rec.actor, rec.task), \
            f"journal diverged at {rec}: env expected {turn}"
        turn = env.step(rec.actor, rec.task, rec.action)
    return turn
