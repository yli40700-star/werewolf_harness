"""Streaming drafts in the online service: what a client is shown while a seat generates, and
that it always ends as the journal's text or an explicit retract (docs/adr/0009)."""
from __future__ import annotations

import asyncio
import random

import pytest

from werewolf_harness.app.llm.gateway import Hub
from werewolf_harness.app.llm.queue import StreamExecutor, Worker
from werewolf_harness.app.runtime.drafts import DraftMirror, outbox
from werewolf_harness.app.runtime.table import Table, TableConfig
from werewolf_harness.players.providers import base as pbase
from werewolf_harness.players.providers.base import Delta, ProviderError
from werewolf_harness.players.providers.mock import STREAM_PROFILES, MockClient

from .conftest import REDIS_URL
from .helpers import until
from .test_runtime import _rt


class Reader:
    """One client: a seat (or a spectator) reading the merged stream through `outbox`."""

    def __init__(self, actor, seat=None):
        self.a, self.seat = actor, seat
        self.mirror = DraftMirror()
        self.cur = {"next": 0, "epoch": None, "dlast": -1, "known": {}}
        self.order: list[tuple[str, dict]] = []
        self.task = asyncio.create_task(self._run())

    def switch(self, actor):
        """Reconnect to another node, carrying the cursor the way app.js does."""
        self.a = actor
        self.cur["known"] = {d.draft_id: len(d.text) for d in self.mirror.drafts.values() if d.state == "open"}

    def pump(self):
        for op, body in outbox(self.a.t, self.seat, self.cur):
            self.order.append((op, body))
            if op == "draft":
                self.mirror.apply(body)

    async def _run(self):
        # not just cancel(): on 3.10 asyncio.wait_for can swallow a cancel that lands as the
        # condition it waits on fires, and a bare `while True` would then go round forever
        me = asyncio.current_task()
        while self.task is me:
            await self.a.wait_any(self.cur["next"], self.cur["dlast"], timeout=0.5)
            self.pump()

    def halt(self):
        """Stop reading (a dropped connection). Returns the task to await, if any."""
        t, self.task = self.task, None
        if t is not None:
            t.cancel()
        return t

    async def close(self):
        t = self.halt()
        try:
            if t is not None:
                await t
        except asyncio.CancelledError:
            pass
        self.pump()


class Faulty(MockClient):
    """A streaming mock that sometimes drops the connection mid-speech or stalls until cancelled."""

    def __init__(self, seed, fail=0.0, stall=0.0, **kw):
        super().__init__(seed=seed, **kw)
        self.fail, self.stall = fail, stall
        self.rng = random.Random(seed * 31 + 7)
        self.cancelled = 0

    def chat_stream(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0, cancel=None):
        cancel = cancel or pbase.CancelToken()
        r = self.rng.random()
        n = 0
        for item in super().chat_stream(messages, tools, temperature, max_tokens, timeout, cancel=cancel):
            if isinstance(item, Delta):
                n += 1
                if n == 3 and r < self.fail:
                    raise ProviderError("connection dropped mid-stream", status=502)
                if n == 3 and r < self.fail + self.stall:
                    cancel.wait(60)
                    self.cancelled += 1
                    cancel.raise_if(n)
            yield item


@pytest.fixture
def profiles(monkeypatch):
    monkeypatch.setitem(STREAM_PROFILES, "mock-slow-speech",
                        {"ttft_s": 0.05, "tokens_per_s": 30.0, "chunk_tokens": 3, "speech_tokens": 300})
    monkeypatch.setitem(STREAM_PROFILES, "mock-mid-speech",
                        {"ttft_s": 0.02, "tokens_per_s": 300.0, "chunk_tokens": 3, "speech_tokens": 120})
    return monkeypatch


def _faulty(monkeypatch, fail, stall):
    orig = Table.client_for

    def client_for(self, model, seat):
        if model == "faulty":
            return Faulty(self.seed * 100 + seat, fail=fail, stall=stall, ttft_s=0.0, tokens_per_s=2000.0,
                          chunk_tokens=3, speech_tokens=60)
        return orig(self, model, seat)
    monkeypatch.setattr(Table, "client_for", client_for)


def _commits_match_journal(t: Table):
    """Server side: every commit event carries the journal's text, before its turn event."""
    turn_ev = {e.data.get("n"): e.seq for e in t.events if e.kind == "turn"}
    n_commit = 0
    for ev in t.draft_log:
        if ev.kind == "draft_commit":
            n_commit += 1
            rec = t.journal[ev.data["n"] - 1]
            assert rec.actor == ev.data["seat"] and rec.action["args"]["content"] == ev.data["text"]
            assert ev.at <= turn_ev[ev.data["n"]], "a draft committed after its turn was published"
    return n_commit


# ---------------------------------------------------------------- the happy path
async def test_drafts_stream_and_commit_with_the_journal_text(settings, store, redis):
    rt = _rt(settings, store, redis)
    await rt.start(scan=False)
    a = await rt.create(TableConfig(variant="8p", models=["mock-stream-fast"]), seed=71)
    spectator = Reader(a)
    await until(lambda: a.t.state == "over", 90)
    await spectator.close()
    assert spectator.mirror.violations(a.t.journal) == []
    c = spectator.mirror.counts()
    assert c["shown"] > 5 and c["committed"] > 5
    assert _commits_match_journal(a.t) >= c["committed"]
    # dseq is contiguous and draft events never go back in the merged order
    dseqs = [ev.dseq for ev in a.t.draft_log]
    assert dseqs == list(range(dseqs[0], dseqs[0] + len(dseqs)))
    ats = [ev.at for ev in a.t.draft_log]
    assert ats == sorted(ats)
    assert not a.t.drafts                                   # nothing left open at the end
    # replay is untouched: drafts are not in the journal-derived event list
    t2 = Table(a.id, 71, TableConfig(variant="8p", models=["mock-stream-fast"]))
    from werewolf_harness.app.runtime.runtime import Runtime
    Runtime.replay(t2, list(a.t.journal))
    assert t2.fingerprint() == a.t.fingerprint()
    await rt.shutdown()


async def test_a_spectator_never_sees_the_pack_draft_but_a_wolf_does(settings, store, redis):
    rt = _rt(settings, store, redis)
    await rt.start(scan=False)
    seen_pack = 0
    for seed in (72, 73):
        a = await rt.create(TableConfig(variant="12p", models=["mock-stream-fast"]), seed=seed)   # 12p has the pack channel
        wolf = next(s for s, r in a.t.env._roles.items() if r == "werewolf")
        good = next(s for s, r in a.t.env._roles.items() if r == "villager")
        readers = [Reader(a), Reader(a, wolf), Reader(a, good)]
        await until(lambda a=a: a.t.state == "over", 90)
        for r in readers:
            await r.close()
        spec, w, g = (r.mirror for r in readers)
        assert "pack_chat" not in spec.seen_tasks
        assert "pack_chat" not in g.seen_tasks or good not in a.t.env._alive   # a dead seat may watch
        seen_pack += "pack_chat" in w.seen_tasks
        for r in readers:
            assert r.mirror.violations(a.t.journal) == []
        # the snapshot a polling client gets is filtered the same way
        assert all(d["task"] != "pack_chat" for d in a.t.drafts_view(None))
    assert seen_pack >= 1
    await rt.shutdown()


# ---------------------------------------------------------------- deadline -> retract + cancel
async def test_a_deadline_retracts_the_draft_and_cancels_the_call(settings, store, redis, profiles):
    rt = _rt(settings.model_copy(update={"agent_timeout_s": 0.8}), store, redis)
    await rt.start(scan=False)
    a = await rt.create(TableConfig(variant="8p", models=["mock-slow-speech"]), seed=75)
    reader = Reader(a)
    await until(lambda: a.t.draft_stats["retracted"] >= 1 and a.t.draft_stats["cancelled_turns"] >= 1, 60)
    retracts = [ev for ev in a.t.draft_log if ev.kind == "draft_retract"]
    assert any(ev.data["reason"] == "timeout" for ev in retracts)
    assert a.t.draft_stats["tokens_saved"] > 0
    # the call really stopped: the executor has no turn of this game still running
    await until(lambda: rt.executor.inflight <= 1, 5)
    timed_out = [ev.data for ev in retracts if ev.data["reason"] == "timeout"]
    await until(lambda: len(a.t.journal) > timed_out[0]["seq"], 5)    # the retract precedes the fallback's commit
    rec = a.t.journal[timed_out[0]["seq"]]
    assert rec.forced and rec.actor == timed_out[0]["seat"]            # the clock's fallback was journalled
    await rt.shutdown(drain=False)
    await reader.close()
    assert reader.mirror.violations(a.t.journal) == []


async def test_stopping_a_game_retracts_and_cancels(settings, store, redis, profiles):
    rt = _rt(settings, store, redis)
    await rt.start(scan=False)
    a = await rt.create(TableConfig(variant="8p", models=["mock-slow-speech"]), seed=76)
    reader = Reader(a)
    await until(lambda: any(d.text for d in a.t.drafts.values()), 60)
    await a.stop("shutdown")
    await reader.close()
    assert not a.t.drafts
    assert any(ev.kind == "draft_retract" and ev.data["reason"] == "shutdown" for ev in a.t.draft_log)
    await until(lambda: a.t.draft_stats["cancelled_turns"] >= 1, 5)
    assert reader.mirror.violations(a.t.journal) == []
    await rt.shutdown(drain=False)


# ---------------------------------------------------------------- crash mid-draft
async def test_a_node_crash_mid_draft_is_retracted_then_asked_again_and_committed(settings, store, redis, profiles):
    a_rt = _rt(settings, store, redis, "node-a")
    await a_rt.start(scan=False)
    # created under the old rules: the adopting node must rebuild it, and ask again, under those
    from werewolf_harness.envs.werewolf import BASELINE_ENGINE_VERSION, ENGINE_VERSION
    assert BASELINE_ENGINE_VERSION != ENGINE_VERSION
    game = await a_rt.create(TableConfig(variant="8p", models=["mock-mid-speech"],
                                         engine_version=BASELINE_ENGINE_VERSION), seed=77)
    reader = Reader(game)
    await until(lambda: any(len(d.text) > 10 for d in game.t.drafts.values()), 60)
    reader.halt()
    reader.pump()                                       # the client saw it...
    open_ids = [d.draft_id for d in reader.mirror.drafts.values() if d.state == "open" and d.text]
    assert open_ids
    shown = reader.mirror.drafts[open_ids[0]]
    # ...and then the node dies without a word: no finally, no retract, the lease just expires
    for t in a_rt._tasks:
        t.cancel()
    game._close_all = lambda reason: None
    game.task.cancel()

    b_rt = _rt(settings, store, redis, "node-b")
    await b_rt.start(scan=True)
    await until(lambda: game.id in b_rt.actors, 10)
    b = b_rt.actors[game.id]
    assert b.t.draft_log and b.t.draft_log[0].kind == "draft_reset"
    assert b.t.engine_version == BASELINE_ENGINE_VERSION and b_rt.counters["adopted"] == 1
    reader.switch(b)
    reader.task = asyncio.create_task(reader._run())
    await until(lambda: len(b.t.journal) > shown.seq, 60)
    await until(lambda: shown.state != "open", 5)
    assert shown.state == "retracted" and shown.reason in ("sync", "recovered")
    rec = b.t.journal[shown.seq]
    assert rec.actor == shown.seat                      # the same turn, asked again on node b...
    again = [ev for ev in b.t.draft_log if ev.kind == "draft_commit" and ev.data["n"] == shown.seq + 1]
    assert not rec.forced and again and again[0].data["text"] == rec.action["args"]["content"]   # ...and committed
    await until(lambda: b.t.state == "over", 90)
    await reader.close()
    assert reader.mirror.violations(b.t.journal) == []
    await b_rt.shutdown()
    await a_rt.executor.close()


async def test_a_worker_that_dies_mid_draft_is_superseded_by_the_rerun(settings, store, redis, profiles):
    ex = StreamExecutor(redis, settings)
    rt = _rt(settings.model_copy(update={"agent_timeout_s": 30.0}), store, redis, executor=ex)
    await rt.start(scan=False)
    w1 = Worker(settings, store, redis, Hub(REDIS_URL, sim_sleep=False), consumer="w1", claim_idle_ms=600)
    t1 = asyncio.create_task(w1.run())
    a = await rt.create(TableConfig(variant="8p", models=["mock-mid-speech"]), seed=78)
    reader = Reader(a)
    await until(lambda: any(len(d.text) > 10 for d in a.t.drafts.values()), 60)
    victim = next(d for d in a.t.drafts.values() if d.text)
    await w1.stop()                                       # dies mid-speech; its job stays unacknowledged
    t1.cancel()
    w2 = Worker(settings, store, redis, Hub(REDIS_URL, sim_sleep=False), consumer="w2", claim_idle_ms=600)
    t2 = asyncio.create_task(w2.run())
    await until(lambda: len(a.t.journal) > victim.seq, 60)
    rec = a.t.journal[victim.seq]
    assert rec.actor == victim.seat and not rec.forced          # the re-run's answer, not the clock's
    closed = [ev for ev in a.t.draft_log if ev.data.get("draft_id") == victim.id and ev.kind != "draft"]
    assert [ev.kind for ev in closed] == ["draft_retract"] and closed[0].data["reason"] in ("rerun", "stale")
    await until(lambda: a.t.state == "over", 120)
    await reader.close()
    assert reader.mirror.violations(a.t.journal) == []
    await w2.stop()
    t2.cancel()
    await rt.shutdown()


async def test_a_cancel_reaches_the_worker_running_the_turn(settings, store, redis, profiles):
    ex = StreamExecutor(redis, settings)
    rt = _rt(settings.model_copy(update={"agent_timeout_s": 0.8}), store, redis, executor=ex)
    await rt.start(scan=False)
    w = Worker(settings, store, redis, Hub(REDIS_URL, sim_sleep=False), consumer="w1")
    wt = asyncio.create_task(w.run())
    a = await rt.create(TableConfig(variant="8p", models=["mock-slow-speech"]), seed=79)
    await until(lambda: w.cancelled >= 1, 60)
    await until(lambda: a.t.draft_stats["cancelled_turns"] >= 1, 10)
    assert a.t.draft_stats["tokens_saved"] > 0
    await w.stop()
    wt.cancel()
    await rt.shutdown(drain=False)


# ---------------------------------------------------------------- reconnect
async def test_a_reconnecting_reader_resumes_from_its_offset(settings, store, redis, profiles):
    rt = _rt(settings, store, redis)
    await rt.start(scan=False)
    a = await rt.create(TableConfig(variant="8p", models=["mock-slow-speech"]), seed=80)
    r1 = Reader(a)
    await until(lambda: any(len(d.text) > 30 for d in a.t.drafts.values()), 60)
    r1.halt()
    r1.pump()
    d = next(x for x in r1.mirror.drafts.values() if x.state == "open" and x.text)
    have = len(d.text)
    await until(lambda: len(a.t.drafts.get(d.draft_id, d).text) > have + 5 or d.draft_id not in a.t.drafts, 30)
    # same epoch, log still holds what it missed: replayed exactly, from its dseq
    cur = dict(r1.cur)
    msgs = [b for op, b in outbox(a.t, None, cur) if op == "draft"]
    assert msgs and all(m["dseq"] > r1.cur["dlast"] for m in msgs)
    # lost its place (another node, trimmed log): synced from the offset it says it has
    cur = {"next": r1.cur["next"], "epoch": None, "dlast": -1, "known": {d.draft_id: have}}
    msgs = [b for op, b in outbox(a.t, None, cur) if op == "draft"]
    assert msgs[0]["kind"] == "draft_sync"
    mine = [m for m in msgs if m.get("draft_id") == d.draft_id]
    assert len(mine) == 1
    if mine[0]["kind"] == "draft":
        assert mine[0]["offset"] == have and a.t.drafts[d.draft_id].text == d.text + mine[0]["delta"]
    for m in msgs:
        r1.mirror.apply(m)
    assert r1.mirror.drafts[d.draft_id].text.startswith(d.text[:have])
    await rt.shutdown(drain=False)


# ---------------------------------------------------------------- the invariant, many games, faults on
async def test_every_shown_draft_is_committed_or_retracted_over_many_games(settings, store, redis, monkeypatch):
    _faulty(monkeypatch, fail=0.08, stall=0.05)
    rt = _rt(settings.model_copy(update={"agent_timeout_s": 1.0}), store, redis)
    await rt.start(scan=False)
    games, readers = [], []
    for i in range(10):
        g = await rt.create(TableConfig(variant="8p", models=["faulty"]), seed=900 + i)
        wolf = next(s for s, r in g.t.env._roles.items() if r == "werewolf")
        games.append(g)
        readers += [Reader(g), Reader(g, wolf)]
    await until(lambda: all(g.t.state == "over" for g in games), 240)
    totals = {"shown": 0, "committed": 0, "retracted": 0}
    reasons: dict[str, int] = {}
    for r in readers:
        await r.close()
        assert r.mirror.violations(r.a.t.journal) == [], r.a.id
        for k, v in r.mirror.counts().items():
            totals[k] = totals.get(k, 0) + v
        for d in r.mirror.drafts.values():
            if d.reason:
                reasons[d.reason] = reasons.get(d.reason, 0) + 1
    for g in games:
        _commits_match_journal(g.t)
        assert not g.t.drafts
    assert totals["committed"] > 50 and totals["retracted"] > 3, (totals, reasons)
    assert any(k.startswith("guard") or k in ("retry", "reset") for k in reasons), reasons
    await rt.shutdown()
