"""Is the model layer overloaded? One number, with hysteresis, for admission control.

When the provider slows down, the turns already in the system take longer, the
worker pool fills, and jobs start to wait. If new games keep arriving, every game
-- old and new -- waits behind every other one and they all stall together. The
cure is to stop *adding* work while the existing work is late (admission.py);
this module decides when that is.

    sample (executor + gateway)            pressure                          state
    ---------------------------            --------                          -----
    queued jobs per lane     ─┐
    in flight / capacity      │    max( demand / demand_per_slot,          ─▶  ok / busy / overloaded
    running games             │         games * per_game / capacity / demand_per_slot,
    oldest queued job age    ─┼─▶       max(oldest_wait, expected_wait) / wait_s,   (enter >= 1.0,
    recent turn time          │         call_latency_p50 / latency_ms )              exit <= 0.7 for hold_s)
    recent call latency      ─┘
        demand        = EWMA over ~3 s of (in flight + queued) / capacity
        per_game      = EWMA over ~20 s of (in flight + queued) / running games
        expected_wait = queued * turn_s / capacity            (Little's law)
        turn_s        = max(recent turn time p50, 2 * mean age of running jobs)

Why these inputs and not others:

  * demand: once every slot is busy, starting another game cannot add
    throughput -- the slots are the throughput -- it can only split it over more
    games, so every running game gets slower. Demand at 1.0 (jobs wanting a slot
    = slots) is the point past which a new game costs the running ones. It is a
    smoothed value because one vote asks 8 seats at once and then nothing.
    A provider slowdown shows up here the way it hurts: games take 10x longer to
    finish, so at the same arrival rate 10x as many are running (Little's law).
  * games * per_game is the same demand projected forward: what the running
    games will ask for once each of them asks as much as an average game does.
    A game just started asks for one or two seats at night; its 8-seat votes
    come a minute later. Admission that only looked at today's demand would let
    in a whole burst of new games before the first of them showed its real
    weight. Counting every running game at the average weight closes that gap.
  * wait is what a player feels: a turn that has not even started. The oldest
    waiting job's age is the observed wait; `expected_wait` predicts it (a
    backlog of Q jobs on C slots that each take S seconds clears in Q*S/C), and
    reacts at once to a slowdown because S is estimated from the running jobs'
    age, not only from turns that have finished.
  * per-call model latency catches a provider so slow that every turn is over
    budget even with no queue at all.
  * queue depth per lane and utilisation are reported (metrics, /api/load);
    they are the parts demand is made of.

Why hysteresis: pressure near 1.0 is noisy. Entering is immediate -- protection
is late otherwise -- but leaving needs pressure under a lower bar for `hold_s`
in a row, so admission does not flap between "start everything" and "start
nothing" on each vote.
"""
from __future__ import annotations

import collections
import math
import threading
import time
from dataclasses import dataclass, field


class LatencyWindow:
    """The last `recent` latencies (ms), dropped once older than `window_s`.
    Thread-safe: gateway calls record from worker threads, the monitor reads on
    the event loop.

    A count window, not a time window, on purpose: right after a provider slows
    down, fast answers from before the change keep arriving in bulk while each
    slow one takes 10x as long to come back, so a time window's median stays on
    the old speed for many seconds. The last 16 answers are mostly slow ones
    within a couple of seconds."""

    def __init__(self, recent: int = 16, window_s: float = 60.0):
        self.window_s = window_s
        self._q: collections.deque[tuple[float, float]] = collections.deque(maxlen=recent)
        self._lock = threading.Lock()

    def record(self, ms: float, now: float | None = None) -> None:
        with self._lock:
            self._q.append((now if now is not None else time.monotonic(), float(ms)))

    def quantiles(self, now: float | None = None) -> tuple[float | None, float | None, int]:
        """(p50, p95, n) over what is recent; (None, None, 0) when nothing is."""
        now = now if now is not None else time.monotonic()
        with self._lock:
            while self._q and now - self._q[0][0] > self.window_s:
                self._q.popleft()
            xs = sorted(v for _, v in self._q)
        if not xs:
            return None, None, 0
        return xs[len(xs) // 2], xs[min(len(xs) - 1, int(0.95 * len(xs)))], len(xs)


@dataclass
class LoadSample:
    """What an executor reports about itself (see LocalExecutor / StreamExecutor.load_sample)."""
    depth: dict[str, int] = field(default_factory=dict)    # jobs waiting for a worker, per lane
    oldest_wait_s: float = 0.0                              # age of the oldest waiting job
    inflight: int = 0                                       # jobs running now
    capacity: int = 1                                       # worker slots
    latency_p50_ms: float | None = None                     # recent per-call model latency (gateway)
    latency_p95_ms: float | None = None
    turn_p50_ms: float | None = None                        # recent per-turn service time (executor)
    inflight_age_s: float = 0.0                             # mean age of the jobs running now
    games: int = 0                                          # games running on this node (set by the runtime)

    @property
    def turn_s(self) -> float:
        """Best current guess of one turn's service time. Finished turns say how long
        turns *took*; after a slowdown none of the slow ones has finished yet, but the
        running ones are already old. For jobs in service at a steady rate, the mean
        age is half the service time, so 2x it is an estimate that moves at once."""
        return max((self.turn_p50_ms or 0.0) / 1000, 2 * self.inflight_age_s)

    @property
    def expected_wait_s(self) -> float:
        """How long a job queued now would wait: the backlog's work spread over the slots."""
        return self.queued * self.turn_s / max(1, self.capacity)

    @property
    def queued(self) -> int:
        return sum(self.depth.values())

    @property
    def demand(self) -> float:
        return (self.inflight + self.queued) / max(1, self.capacity)


@dataclass
class Thresholds:
    wait_s: float = 2.0             # observed or expected wait of a job this long = pressure 1
    demand_per_slot: float = 1.0    # smoothed jobs wanting a slot, per slot = pressure 1
    latency_ms: float = 20_000.0    # recent p50 per-call model latency this slow = pressure 1
    enter: float = 1.0              # become overloaded at or above
    exit: float = 0.7               # stop being overloaded at or below ...
    hold_s: float = 3.0             # ... continuously for this long
    demand_tau_s: float = 3.0       # smoothing time constant of demand
    per_game_tau_s: float = 20.0    # ... and of the jobs one game wants on average
    per_game_prior: float = 1.0     # that average before any game has been seen
    per_game_max: float = 12.0      # ... and its ceiling (the largest table has 12 seats)


def pressure(s: LoadSample, th: Thresholds, demand: float | None = None,
             per_game: float | None = None) -> tuple[float, dict[str, float]]:
    """The overload score (1.0 = at the limit) and what contributed to it. `demand`
    is the smoothed (in flight + queued) / capacity, `per_game` the smoothed jobs
    per running game; instant values if not given."""
    if demand is None:
        demand = s.demand
    if per_game is None:
        per_game = (s.inflight + s.queued) / s.games if s.games else th.per_game_prior
    k = th.demand_per_slot * max(1, s.capacity)
    parts = {
        "demand": demand / th.demand_per_slot if th.demand_per_slot > 0 else 0.0,
        "games": s.games * per_game / k if th.demand_per_slot > 0 else 0.0,
        "wait": max(s.oldest_wait_s, s.expected_wait_s) / th.wait_s if th.wait_s > 0 else 0.0,
        "latency": (s.latency_p50_ms or 0.0) / th.latency_ms if th.latency_ms > 0 else 0.0,
    }
    return max(parts.values()), {k: round(v, 3) for k, v in parts.items()}


class Hysteresis:
    """Two thresholds and a hold time: on at `enter`, off only after `hold_s` at or under `exit`."""

    def __init__(self, enter: float, exit: float, hold_s: float):
        assert exit < enter, "exit threshold must be below enter, or the state flaps"
        self.enter, self.exit, self.hold_s = enter, exit, hold_s
        self.on = False
        self.since = time.monotonic()            # when `on` last changed
        self._calm_since: float | None = None    # when pressure last went under `exit` while on
        self.flips = 0

    def update(self, p: float, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        if not self.on:
            if p >= self.enter:
                self._set(True, now)
            return self.on
        if p > self.exit:
            self._calm_since = None
        elif self._calm_since is None:
            self._calm_since = now
        if self._calm_since is not None and now - self._calm_since >= self.hold_s:
            self._set(False, now)
        return self.on

    def _set(self, on: bool, now: float) -> None:
        self.on, self.since, self._calm_since = on, now, None
        self.flips += 1


class LoadMonitor:
    """Turns executor samples into the overload state. Owned by the Runtime, which
    calls `observe` every tick; admission reads `overloaded`."""

    def __init__(self, th: Thresholds | None = None, gateway_latency: LatencyWindow | None = None):
        self.th = th or Thresholds()
        self.h = Hysteresis(self.th.enter, self.th.exit, self.th.hold_s)
        self.gateway_latency = gateway_latency
        self.last: LoadSample = LoadSample()
        self.pressure = 0.0
        self.parts: dict[str, float] = {}
        self.at = 0.0
        self.demand = 0.0                        # EWMA of LoadSample.demand
        self.per_game = self.th.per_game_prior   # EWMA of jobs per running game

    @property
    def overloaded(self) -> bool:
        return self.h.on

    def observe(self, s: LoadSample, now: float | None = None) -> bool:
        if self.gateway_latency is not None:
            # per-call latency from the gateway is the provider's own speed (a turn is
            # several calls; the executor's per-turn time stays in turn_p50_ms)
            p50, p95, n = self.gateway_latency.quantiles(now)
            if n:
                s.latency_p50_ms, s.latency_p95_ms = p50, p95
        self.last = s
        t = now if now is not None else time.monotonic()
        # time-based EWMA: samples come every tick and on every creation request
        a = 1.0 if not self.at else 1 - math.exp(-max(0.0, t - self.at) / max(1e-6, self.th.demand_tau_s))
        self.demand += a * (s.demand - self.demand)
        if s.games:
            b = 1.0 if not self.at else 1 - math.exp(-max(0.0, t - self.at) / max(1e-6, self.th.per_game_tau_s))
            # a game cannot want more slots than it has seats; a stuck queue must not teach otherwise
            self.per_game += b * (min(self.th.per_game_max, (s.inflight + s.queued) / s.games) - self.per_game)
        self.at = t
        self.pressure, self.parts = pressure(s, self.th, self.demand, self.per_game)
        from ..obs import metrics
        metrics.LOAD_PRESSURE.set(self.pressure)
        for lane, n in s.depth.items():
            metrics.QUEUE_DEPTH.labels(lane=lane).set(n)
        on = self.h.update(self.pressure, now)
        metrics.OVERLOADED.set(1 if on else 0)
        return on

    @property
    def level(self) -> str:
        if self.overloaded:
            return "overloaded"
        return "busy" if self.pressure > self.th.exit else "ok"

    def view(self) -> dict:
        s = self.last
        return {"overloaded": self.overloaded, "level": self.level, "pressure": round(self.pressure, 3),
                "parts": self.parts, "demand": round(self.demand, 3), "per_game": round(self.per_game, 2),
                "games": s.games, "depth": dict(s.depth), "queued_jobs": s.queued,
                "oldest_wait_s": round(s.oldest_wait_s, 2), "expected_wait_s": round(s.expected_wait_s, 2),
                "turn_p50_ms": None if s.turn_p50_ms is None else round(s.turn_p50_ms),
                "turn_estimate_s": round(s.turn_s, 2),
                "inflight": s.inflight, "capacity": s.capacity,
                "utilisation": round(s.inflight / max(1, s.capacity), 3),
                "latency_p50_ms": None if s.latency_p50_ms is None else round(s.latency_p50_ms),
                "latency_p95_ms": None if s.latency_p95_ms is None else round(s.latency_p95_ms),
                "state_for_s": round(time.monotonic() - self.h.since, 1), "flips": self.h.flips,
                "thresholds": {"enter": self.th.enter, "exit": self.th.exit, "hold_s": self.th.hold_s,
                               "wait_s": self.th.wait_s, "demand_per_slot": self.th.demand_per_slot,
                               "latency_ms": self.th.latency_ms}}
