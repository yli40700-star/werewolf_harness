"""A model turn run by a worker from what the job carries, and the actor's check of the answer.

The worker gets the seat's view, action spec, memory and player settings in the job
(llm/queue.py) and must not need Postgres, the journal or the engine to run the turn.
Whatever it sends back, the actor decides at commit time whether it is still usable.
No database or Redis here: the worker's one Redis read is faked.
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from ..app.config import Settings
from ..app.llm.queue import Worker, job_payload
from ..app.runtime.actor import GameActor
from ..app.runtime.agents import AgentJob, AgentResult, run_agent_turn
from ..app.runtime.runtime import Runtime
from ..app.runtime.table import Table, TableConfig
from ..app.runtime.timers import TimingWheel
from ..app.storage import Store
from ..arena import Turn
from ..envs import Werewolf


class _Redis:
    """All a worker asks Redis before a turn: is this job closed?"""

    def __init__(self, closed=()):
        self.closed = set(closed)
        self.asked: list[str] = []

    async def exists(self, key):
        self.asked.append(key)
        return int(key.rsplit(":", 1)[1] in self.closed)


def _job(t: Table, turn: Turn) -> AgentJob:
    """The job the actor builds (actor.py _ask), with the in-process player."""
    v = len(t.journal)
    return AgentJob(t.id, turn.actor, v, turn.task, t.env.observe(turn.actor), t.env.action_space(turn.actor, turn.task),
                    t.memory.get(turn.actor), make_player=lambda on_step: t.make_player(turn.actor, task=turn.task, version=v),
                    settings=t.seat_settings(turn.actor, turn.task), job_id=f"{t.id}-{v}-{turn.actor}")


def _fields(job: AgentJob) -> dict:
    """The stream entry as a worker reads it: every field a string."""
    f = {"game_id": job.game_id, "seat": job.seat, "version": job.version, "task": job.task,
         "memory": json.dumps(job.memory, ensure_ascii=False) if job.memory else "", "reply_to": "n1",
         "enq": time.time(), "lane": job.lane, "stream": "0", "job_id": job.job_id, "tp": "", **job_payload(job)}
    return {k: str(v) for k, v in f.items()}


def _forbid(monkeypatch, *targets):
    def boom(*_a, **_k):
        raise AssertionError("the worker touched the game's state or storage")
    for owner, name in targets:
        monkeypatch.setattr(owner, name, boom)


@pytest.mark.parametrize("cfg", [
    TableConfig(variant="8p", models=["mock"]),
    TableConfig(variant="12p", models=["mock"], playbook="static"),
    TableConfig(variant="8p", cascade={"cheap": "mock-flaky:0.5", "strong": "mock-strong", "mode": "both"}),
], ids=["8p", "12p-pack", "cascade"])
async def test_a_worker_plays_a_whole_game_from_the_jobs_alone(cfg, monkeypatch):
    """Every turn of a game twice: in-process (the Table's own player) and by a Worker given
    only the stream entry, with the engine and the store made to blow up if touched. The
    answers, the memory and the cascade record are the same."""
    t = Table("gW", 17, cfg)
    t.reset()
    w = Worker(Settings(), _Redis(), None, consumer="w")
    turns = 0
    while isinstance(t.turn, Turn):
        job = _job(t, t.turn)
        here = run_agent_turn(job)
        fields = _fields(job)
        with monkeypatch.context() as m:
            _forbid(m, (Werewolf, "observe"), (Werewolf, "action_space"), (Werewolf, "step"), (Werewolf, "reset"),
                    (Werewolf, "__init__"), (Table, "__init__"), (Runtime, "replay"), (Runtime, "table_from_row"),
                    (Store, "connect"), (Store, "journal"), (Store, "get_game"), (Store, "__init__"))
            there = await w._run(fields)
        assert there is not None
        assert there["action"] == here.action and there["forced"] == here.forced, t.turn
        assert there["memory"] == (list(here.memory) if here.memory else None)
        assert there["meta"].get("cascade") == here.meta.get("cascade")
        assert there["meta"]["remote"] and not there.get("error")
        t.memory[t.turn.actor] = here.memory
        t.apply(here.action, here.forced)
        turns += 1
    assert turns > 20 and w.done == turns and w.stale == 0
    w.pool.shutdown(wait=False)


async def test_a_closed_job_costs_no_model_call(monkeypatch):
    t = Table("gC", 3, TableConfig(variant="8p", models=["mock"]))
    t.reset()
    job = _job(t, t.turn)
    w = Worker(Settings(), _Redis(closed={job.job_id}), None, consumer="w")
    import werewolf_harness.app.llm.queue as q
    _forbid(monkeypatch, (q, "build_player"), (q, "run_agent_turn"))
    assert await w._run(_fields(job)) is None and w.stale == 1          # cancelled, or answered already
    w._on_cancel({"job_id": "early", "game_id": "gC", "seat": 1, "version": 0})
    assert await w._run({**_fields(job), "job_id": "early"}) is None and w.stale == 2   # heard cancelled while queued
    bare = {k: v for k, v in _fields(job).items() if k not in ("view", "spec", "player")}
    assert await w._run({**bare, "job_id": "old"}) is None and w.stale == 3   # nothing to run it on
    w.pool.shutdown(wait=False)


# ------------------------------------------------------------------ the actor's check at commit
class _Store:
    def __init__(self):
        self.rows: list[dict] = []

    async def append(self, game_id, **kw):
        self.rows.append(kw)

    async def set_state(self, *a, **kw):
        pass


class _Executor:
    def __init__(self):
        self.jobs: list[AgentJob] = []
        self.cancelled: list[tuple[AgentJob, str]] = []

    def submit(self, job, deliver):
        self.jobs.append(job)

    def cancel(self, job, reason):
        self.cancelled.append((job, reason))


async def _until(cond, timeout=5.0):
    t0 = time.monotonic()
    while not cond():
        assert time.monotonic() - t0 < timeout, "timed out"
        await asyncio.sleep(0.01)


async def test_the_actor_rejects_an_illegal_or_stale_answer_and_asks_again():
    t = Table("gA", 5, TableConfig(variant="8p", models=["mock"]))
    t.reset()
    ex, store, wheel = _Executor(), _Store(), TimingWheel(tick=0.05)
    wheel.start()
    a = GameActor(t, store=store, executor=ex, wheel=wheel, token=1, agent_timeout_s=60)
    a.launch()
    turn = t.turn
    await _until(lambda: any(j.seat == turn.actor for j in ex.jobs))
    job = next(j for j in ex.jobs if j.seat == turn.actor)
    asked = len([j for j in ex.jobs if j.seat == turn.actor])

    def answer(action, version=job.version, forced=False):
        a.post(AgentResult(t.id, turn.actor, version, action, forced, None, {"remote": True}))

    # not a legal action for this turn: never committed, the seat is asked again
    answer({"name": "no_such_tool", "args": {}})
    await _until(lambda: len([j for j in ex.jobs if j.seat == turn.actor]) == asked + 1)
    assert t.misses == 1 and not store.rows and not t.journal
    # legal in shape but for a world that has moved on (another version): dropped
    answer(job.spec.fallback, version=job.version + 3)
    await _until(lambda: t.stale_results == 1)
    assert not store.rows
    # the seat would now see something other than what it was sent: not used, asked again
    a.asks[turn.actor]["view"] = "the view at ask time differed"
    answer(job.spec.fallback)
    await _until(lambda: len([j for j in ex.jobs if j.seat == turn.actor]) == asked + 2)
    assert t.misses == 2 and not store.rows
    # a legal answer to the live ask is committed, journal first
    legal = job.spec.fallback
    answer(legal)
    await _until(lambda: len(store.rows) == 1)
    assert store.rows[0]["action"] == legal and store.rows[0]["actor"] == turn.actor and len(t.journal) == 1
    assert not [c for c in ex.cancelled if c[0].seat == turn.actor and c[1] == "stale"]   # answered: nothing to cancel
    await a.stop()
    await wheel.stop()


@pytest.mark.parametrize("mode", ["on-fail", "decisive", "both"])
def test_the_seat_settings_rebuild_the_same_cascade_policy(mode):
    from ..players.llm.cascade import CascadePolicy
    t = Table("gP", 2, TableConfig(variant="8p", cascade={"cheap": "mock-flaky", "strong": "mock-strong", "mode": mode,
                                                          "decisive": ["vote"], "low_confidence": True}))
    st = json.loads(json.dumps(t.seat_settings(1, "vote")))
    assert CascadePolicy.from_dict(st["cascade"]) == t.cascade_policy
    assert t.make_player(1, task="vote").cascade == t.cascade_policy
