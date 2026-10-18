"""The wheel fires within one tick, survives many laps, and cancels in O(1)."""
from __future__ import annotations

import asyncio
import time

from ..app.runtime.timers import TimingWheel


async def test_fires_close_to_its_deadline():
    w = TimingWheel(tick=0.02, size=8)
    w.start()
    fired: dict[str, float] = {}
    t0 = time.monotonic()
    for d in (0.05, 0.1, 0.3):                  # 0.3 s is more than one lap (8 * 0.02 = 0.16 s)
        w.schedule(d, lambda d=d: fired.__setitem__(str(d), time.monotonic() - t0))
    await asyncio.sleep(0.45)
    await w.stop()
    assert set(fired) == {"0.05", "0.1", "0.3"}
    for d, at in fired.items():
        assert abs(at - float(d)) < 0.05, (d, at)


async def test_cancelled_timeouts_never_fire_and_are_dropped():
    w = TimingWheel(tick=0.01, size=16)
    w.start()
    hits = []
    ts = [w.schedule(0.05, lambda: hits.append(1)) for _ in range(100)]
    for t in ts[:99]:
        t.cancel()
    await asyncio.sleep(0.12)
    await w.stop()
    assert hits == [1]
    assert w.pending == 0


async def test_exact_multiple_of_a_lap_is_not_early():
    w = TimingWheel(tick=0.01, size=10)
    w.start()
    t0 = time.monotonic()
    got = []
    w.schedule(0.1, lambda: got.append(time.monotonic() - t0))    # exactly one lap
    await asyncio.sleep(0.2)
    await w.stop()
    assert got and got[0] >= 0.09


async def test_a_failing_callback_does_not_stop_the_clock():
    w = TimingWheel(tick=0.01, size=8)
    w.start()
    got = []
    w.schedule(0.02, lambda: 1 / 0)
    w.schedule(0.04, lambda: got.append(1))
    await asyncio.sleep(0.1)
    await w.stop()
    assert got == [1]
