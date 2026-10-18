"""Overload protection without a database: the load signal, admission decisions,
per-game fair queuing, the Streams per-game cap, the sim slow mode, the lobby UI."""
from __future__ import annotations

import asyncio
import pathlib
import time

import pytest

from werewolf_harness.app.llm import sim
from werewolf_harness.app.runtime import agents
from werewolf_harness.app.runtime.admission import ADMIT, QUEUE, REJECT, Admission, Ticket, decide
from werewolf_harness.app.runtime.agents import AgentJob, AgentResult, LocalExecutor
from werewolf_harness.app.runtime.load import (Hysteresis, LatencyWindow, LoadMonitor, LoadSample, Thresholds,
                                               pressure)

WEB = pathlib.Path(__file__).resolve().parent.parent / "web"


# ------------------------------------------------------------------ hysteresis
def test_hysteresis_enters_at_once_and_leaves_only_after_a_calm_hold():
    h = Hysteresis(enter=1.0, exit=0.6, hold_s=3.0)
    assert not h.update(0.99, now=0.0)
    assert h.update(1.0, now=1.0)                 # on immediately
    assert h.update(0.5, now=2.0)                 # calm, but not for long enough
    assert h.update(0.5, now=4.9)
    assert not h.update(0.5, now=5.0)             # 3 s at or under exit: off
    assert h.flips == 2


def test_hysteresis_does_not_flap_on_a_noisy_signal():
    h = Hysteresis(enter=1.0, exit=0.6, hold_s=3.0)
    h.update(1.2, now=0.0)
    # pressure bouncing between "just under exit" and "between the thresholds" every second
    for i in range(1, 30):
        h.update(0.5 if i % 2 else 0.8, now=float(i))
    assert h.on and h.flips == 1                  # never calm for 3 s in a row
    # in the band between the thresholds it never turns on either
    h2 = Hysteresis(enter=1.0, exit=0.6, hold_s=3.0)
    for i in range(30):
        h2.update(0.6 + 0.39 * (i % 2), now=float(i))
    assert not h2.on and h2.flips == 0


def test_exit_must_be_below_enter():
    with pytest.raises(AssertionError):
        Hysteresis(enter=1.0, exit=1.0, hold_s=1)


# ------------------------------------------------------------------ the signal
def test_pressure_parts():
    th = Thresholds(wait_s=2.0, demand_per_slot=1.0, latency_ms=20_000)
    calm = LoadSample(depth={"online": 0}, inflight=4, capacity=16, games=4)
    p, parts = pressure(calm, th)
    assert p < 0.5 and set(parts) == {"demand", "games", "wait", "latency"}
    # a backlog: 16 waiting jobs of 6 s turns on 16 slots is 6 s of expected wait
    s = LoadSample(depth={"online": 16}, inflight=16, capacity=16, turn_p50_ms=6000, games=10)
    assert s.expected_wait_s == pytest.approx(6.0)
    p, parts = pressure(s, th)
    assert parts["wait"] == pytest.approx(3.0) and parts["demand"] == pytest.approx(2.0) and p == pytest.approx(3.0)
    # the oldest job's observed age counts as well
    assert pressure(LoadSample(oldest_wait_s=5, capacity=16), th)[1]["wait"] == pytest.approx(2.5)
    # a provider so slow every turn is over budget, even with no queue
    assert pressure(LoadSample(capacity=16, latency_p50_ms=30_000), th)[0] == pytest.approx(1.5)


def test_turn_time_estimate_moves_before_any_slow_turn_has_finished():
    # finished turns still say 0.6 s, but the running ones are already 2 s old on average
    s = LoadSample(turn_p50_ms=600, inflight_age_s=2.0)
    assert s.turn_s == pytest.approx(4.0)
    assert LoadSample(turn_p50_ms=600, inflight_age_s=0.1).turn_s == pytest.approx(0.6)


def test_games_part_counts_every_running_game_at_the_average_weight():
    m = LoadMonitor(Thresholds(per_game_tau_s=1e-9))     # no smoothing: per_game = this sample
    m.observe(LoadSample(inflight=12, capacity=16, games=6), now=1.0)   # 2 jobs per game
    assert m.per_game == pytest.approx(2.0)
    # ten fresh games that ask little yet: today's demand is low, the projection is not
    m.observe(LoadSample(inflight=4, capacity=16, games=10), now=2.0)
    assert m.parts["games"] == pytest.approx(10 * 0.4 / 16)
    m2 = LoadMonitor(Thresholds(per_game_tau_s=1000))
    m2.observe(LoadSample(inflight=12, capacity=16, games=6), now=1.0)
    m2.observe(LoadSample(inflight=4, capacity=16, games=10), now=2.0)
    assert m2.parts["games"] > 1.0                       # the slow average still says ~2 per game


def test_monitor_smooths_demand_and_reports_a_level():
    m = LoadMonitor(Thresholds(demand_tau_s=3.0, hold_s=1.0))
    m.observe(LoadSample(inflight=0, capacity=10), now=1.0)
    assert m.level == "ok"
    m.observe(LoadSample(inflight=10, depth={"online": 10}, capacity=10), now=1.5)   # a vote: 2x demand
    assert 0 < m.demand < 1.0 and not m.overloaded        # one burst does not trip it
    for t in range(2, 12):
        m.observe(LoadSample(inflight=10, depth={"online": 10}, capacity=10), now=float(t))
    assert m.overloaded and m.level == "overloaded"
    v = m.view()
    assert v["overloaded"] and v["queued_jobs"] == 10 and v["capacity"] == 10 and "thresholds" in v


def test_latency_window_follows_a_slowdown_within_a_few_answers():
    w = LatencyWindow(recent=16)
    for i in range(500):                          # a long fast history
        w.record(300, now=i / 100)
    for i in range(10):                           # ten slow answers come back
        w.record(3000, now=5 + i / 10)
    p50, _p95, n = w.quantiles(now=6.0)
    assert p50 == 3000 and n == 16
    assert w.quantiles(now=1000.0) == (None, None, 0)    # nothing recent


# ------------------------------------------------------------------ admission
def test_admission_decisions():
    assert decide(overloaded=False, queued=0, max_queue=5, active=3) == ADMIT
    assert decide(overloaded=True, queued=0, max_queue=5, active=3) == QUEUE
    assert decide(overloaded=True, queued=5, max_queue=5, active=3) == REJECT
    # FIFO: while anyone is in line, a newcomer goes behind them even if the node is calm now
    assert decide(overloaded=False, queued=1, max_queue=5, active=3) == QUEUE
    # hard cap on active games
    assert decide(overloaded=False, queued=0, max_queue=5, active=10, max_active=10) == QUEUE
    # slow start in the busy band: one start per probe_s
    assert decide(overloaded=False, queued=0, max_queue=5, active=3, busy=True, since_last_start_s=1,
                  probe_s=5) == QUEUE
    assert decide(overloaded=False, queued=0, max_queue=5, active=3, busy=True, since_last_start_s=6,
                  probe_s=5) == ADMIT
    assert decide(overloaded=False, queued=0, max_queue=5, active=3, busy=False, since_last_start_s=0.1,
                  probe_s=5) == ADMIT


def test_admission_queue_is_fifo_with_positions_and_eta():
    a = Admission(max_queue=3, mean_game_s=100, probe_s=0)
    for g in ("g1", "g2", "g3"):
        assert a.decide(overloaded=True, active=10) == QUEUE
        a.push(Ticket(g, table=None))
    assert a.decide(overloaded=True, active=10) == REJECT
    assert [a.position(g) for g in ("g1", "g2", "g3")] == [1, 2, 3]
    assert a.info("g2", active=10) == {"position": 2, "eta_s": 20, "waited_s": pytest.approx(0, abs=1),
                                       "length": 3}
    assert a.retry_after_s(active=10) == 40              # the tail of the line: (3+1) * 100 / 10
    assert not a.can_start_next(overloaded=True, active=10)
    assert a.can_start_next(overloaded=False, active=10)
    assert a.pop().game_id == "g1" and a.position("g2") == 1
    assert a.counts == {ADMIT: 0, QUEUE: 3, REJECT: 1}
    a.game_ended(200)
    assert a.mean_game_s == pytest.approx(120)


def test_retry_after_is_clamped():
    a = Admission(max_queue=1, mean_game_s=10_000)
    assert a.retry_after_s(active=1) == 600
    assert Admission(max_queue=1, mean_game_s=1).retry_after_s(active=100) == 5


def test_probe_spaces_out_starts_while_busy():
    a = Admission(max_queue=10, probe_s=60)
    assert a.decide(overloaded=False, active=1, busy=True) == ADMIT
    assert a.decide(overloaded=False, active=1, busy=True) == QUEUE       # within probe_s of the last start
    a.push(Ticket("g", table=None))
    assert not a.can_start_next(overloaded=False, active=1, busy=True)
    assert a.can_start_next(overloaded=False, active=1, busy=False)       # calm again: no spacing


# ------------------------------------------------------------------ fair queuing
class _Spec:
    fallback = {"name": "pass"}

    def check(self, _a):
        return None


def _job(game: str, version: int = 5, lane: str = "online", seat: int = 1) -> AgentJob:
    return AgentJob(game, seat, version, "vote", {}, _Spec(), None, make_player=lambda on_step: None, lane=lane)


@pytest.fixture
def fake_turn(monkeypatch):
    """Turns that just sleep a little: the pool, not the model, is under test."""
    def run(job):
        time.sleep(0.02)
        return AgentResult(job.game_id, job.seat, job.version, {"name": "x"}, False)
    monkeypatch.setattr(agents, "run_agent_turn", run)


async def _drain(ex: LocalExecutor, n: int) -> list[str]:
    got: list[str] = []
    done = asyncio.Event()

    def deliver(r):
        got.append(r.game_id)
        if len(got) == n:
            done.set()
    return got, done, deliver


async def test_a_vote_of_8_does_not_starve_another_games_turn(fake_turn):
    ex = LocalExecutor(1, fair=True)
    got, done, deliver = await _drain(ex, 9)
    for seat in range(1, 9):
        ex.submit(_job("A", seat=seat), deliver)          # one game asks 8 seats at once
    ex.submit(_job("B"), deliver)                         # another game's single turn
    await asyncio.wait_for(done.wait(), 5)
    # A's first job had the thread already; B goes next, not after all eight of A's
    assert ex.started_order[:2] == ["A", "B"]
    await ex.close()


async def test_fifo_mode_is_the_old_order(fake_turn):
    ex = LocalExecutor(1, fair=False)
    got, done, deliver = await _drain(ex, 9)
    for seat in range(1, 9):
        ex.submit(_job("A", seat=seat), deliver)
    ex.submit(_job("B"), deliver)
    await asyncio.wait_for(done.wait(), 5)
    assert ex.started_order == ["A"] * 8 + ["B"]
    await ex.close()


async def test_games_in_flight_go_before_a_new_games_first_turn(fake_turn):
    ex = LocalExecutor(1, fair=True)
    got, done, deliver = await _drain(ex, 4)
    ex.submit(_job("busy", version=3), deliver)           # holds the thread
    ex.submit(_job("new", version=0), deliver)            # a new game's first turn, asked first...
    ex.submit(_job("old", version=7), deliver)            # ...but a running game's turn goes before it
    ex.submit(_job("old2", version=9), deliver)
    await asyncio.wait_for(done.wait(), 5)
    assert ex.started_order == ["busy", "old", "old2", "new"]
    await ex.close()


async def test_a_new_games_first_turn_ages_into_the_normal_order(fake_turn):
    ex = LocalExecutor(1, fair=True, new_grace_s=0.0)
    got, done, deliver = await _drain(ex, 3)
    ex.submit(_job("busy", version=3), deliver)
    ex.submit(_job("new", version=0), deliver)
    ex.submit(_job("old", version=7), deliver)
    await asyncio.wait_for(done.wait(), 5)
    assert ex.started_order == ["busy", "new", "old"]
    await ex.close()


async def test_lanes_still_come_first(fake_turn):
    ex = LocalExecutor(1, fair=True)
    got, done, deliver = await _drain(ex, 3)
    ex.submit(_job("x", lane="offline"), deliver)
    ex.submit(_job("y", lane="offline"), deliver)
    ex.submit(_job("h", lane="human"), deliver)
    await asyncio.wait_for(done.wait(), 5)
    assert ex.started_order == ["x", "h", "y"]
    await ex.close()


async def test_per_game_cap(fake_turn):
    ex = LocalExecutor(8, fair=True, per_game_cap=2)
    got, done, deliver = await _drain(ex, 8)
    for seat in range(1, 9):
        ex.submit(_job("A", seat=seat), deliver)
    assert ex.inflight == 2 and ex.load_sample().queued == 6     # 8 threads free, 2 used
    await asyncio.wait_for(done.wait(), 5)
    await ex.close()


async def test_load_sample_of_the_local_executor(fake_turn):
    ex = LocalExecutor(2, fair=True)
    got, done, deliver = await _drain(ex, 5)
    for i in range(5):
        ex.submit(_job(f"g{i}", lane="human" if i == 4 else "online"), deliver)
    s = ex.load_sample()
    assert s.capacity == 2 and s.inflight == 2 and s.depth == {"online": 2, "human": 1}
    assert s.oldest_wait_s >= 0
    await asyncio.wait_for(done.wait(), 5)
    s = ex.load_sample()
    assert s.queued == 0 and s.inflight == 0 and s.turn_p50_ms is not None and s.turn_p50_ms >= 15
    await ex.close()


# ------------------------------------------------------------------ cancel (docs/adr/0009) x fair queue
async def test_cancel_drops_a_queued_job_and_cuts_a_running_one(monkeypatch):
    def run(job):                                         # a turn that runs until it is cancelled
        cut = job.cancel.wait(5)
        return AgentResult(job.game_id, job.seat, job.version, None, True,
                           meta={"cancelled": job.cancel.reason} if cut else {})
    monkeypatch.setattr(agents, "run_agent_turn", run)
    ex = LocalExecutor(1, fair=True)
    got: list = []
    done = asyncio.Event()

    def deliver(r):
        got.append(r)
        if len(got) == 2:
            done.set()
    a1, a2, b = _job("A", seat=1), _job("A", seat=2), _job("B")
    for j in (a1, a2, b):
        ex.submit(j, deliver)
    assert ex.inflight == 1 and ex.load_sample().queued == 2
    ex.cancel(a2, "turn_gone")                            # still waiting: out of the queue, never runs
    assert ex.dropped == 1 and ex.load_sample().queued == 1 and a2.cancel.is_set()
    t0 = time.monotonic()
    ex.cancel(a1, "timeout")                              # on a thread: its call is cut, B gets the thread
    while ex.started_order != ["A", "B"]:
        await asyncio.sleep(0.01)
    ex.cancel(b, "shutdown")
    await asyncio.wait_for(done.wait(), 3)
    assert time.monotonic() - t0 < 1
    assert [(r.game_id, r.seat, r.meta.get("cancelled")) for r in got] == [("A", 1, "timeout"), ("B", 1, "shutdown")]
    assert ex.started_order == ["A", "B"] and not ex.waiting and not ex.running
    await ex.close()


async def test_streams_executor_cancel_keeps_a_held_job_from_going_out():
    import json as _json

    from werewolf_harness.app.config import Settings
    from werewolf_harness.app.llm.queue import CANCEL_CHANNEL, StreamExecutor
    r = _FakeRedis()
    ex = StreamExecutor(r, Settings(), per_game_asks=1)
    j1, j2 = _job("A", seat=1), _job("A", seat=2)
    ex.submit(j1, lambda res: None)
    ex.submit(j2, lambda res: None)                       # held on the node behind the cap
    ex.cancel(j2, "turn_gone")
    ex.cancel(j1, "timeout")                              # in the stream: told to the workers
    await asyncio.sleep(0)
    ex._settle("A", 1, 5)                                 # j1's (cancelled) answer frees the slot...
    await asyncio.sleep(0)
    assert [f["seat"] for _k, f in r.added] == [1]        # ...and j2 does not go out
    assert [(c, _json.loads(m)["job_id"]) for c, m in r.published] == [(CANCEL_CHANNEL, j1.job_id)]


# ------------------------------------------------------------------ streams per-game cap
class _FakeRedis:
    def __init__(self):
        self.added: list[tuple[str, dict]] = []
        self.published: list[tuple[str, str]] = []

    async def xadd(self, key, fields, **kw):
        self.added.append((key, fields))

    async def publish(self, channel, msg):
        self.published.append((channel, msg))


async def test_streams_executor_caps_jobs_per_game_in_the_stream():
    from werewolf_harness.app.config import Settings
    from werewolf_harness.app.llm.queue import StreamExecutor
    r = _FakeRedis()
    ex = StreamExecutor(r, Settings(), per_game_asks=3)
    for seat in range(1, 9):
        ex.submit(_job("A", seat=seat), lambda res: None)
    ex.submit(_job("B"), lambda res: None)
    await asyncio.sleep(0)
    sent = [f["game_id"] for _k, f in r.added]
    assert sent.count("A") == 3 and sent.count("B") == 1           # B is not behind A's other five
    ex._settle("A", 1, 5)                                          # an answer comes back: one more goes out
    ex._settle("A", 2, 5)
    await asyncio.sleep(0)
    assert [f["game_id"] for _k, f in r.added].count("A") == 5
    # a lost answer does not hold its slot for ever
    ex.s = ex.s.model_copy(update={"agent_timeout_s": -10})
    ex._release("A")
    await asyncio.sleep(0)
    assert [f["game_id"] for _k, f in r.added].count("A") == 8 and not ex.held


# ------------------------------------------------------------------ sim slow mode
def test_sim_latency_is_injectable():
    ctl = sim.LatencyControl()
    c = sim.SimClient(1, median_ms=300, sigma=0.0, sleep=False, control=ctl)
    msgs = [{"role": "user", "content": "hi"}]
    assert c.chat(msgs).latency_ms == 300
    ctl.set(median_ms=3000)                       # the provider slows down 10x
    assert c.chat(msgs).latency_ms == 3000
    ctl.set(scale=2.0)
    assert c.chat(msgs).latency_ms == 600         # back to its own profile, scaled
    assert sim.LatencyControl(scale=1.0).apply(0) == 0


# ------------------------------------------------------------------ lobby UI
def test_lobby_shows_the_queue_position_and_a_load_badge():
    html = (WEB / "index.html").read_text()
    js = (WEB / "app.js").read_text()
    assert 'id="load-badge"' in html and 'id="queue-banner"' in html
    assert "当前排队中：第 ${q.position} 位，预计 ${q.eta_s} 秒" in js
    assert '"/api/load"' in js and "/queue`" in js
    assert "r.status === 202" in js and "r.status === 503" in js
