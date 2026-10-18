from __future__ import annotations

import asyncio
import time

from werewolf_harness.app.runtime.agents import AgentJob, LocalExecutor


class CountingExecutor(LocalExecutor):
    """Records (game, version, seat) of every model turn it was asked to run."""

    def __init__(self, workers: int = 8, delay: float = 0.0):
        super().__init__(workers)
        self.jobs: list[tuple[str, int, int]] = []
        self.delay = delay

    def submit(self, job: AgentJob, deliver):
        self.jobs.append((job.game_id, job.version, job.seat))
        if self.delay:
            loop = asyncio.get_running_loop()
            loop.call_later(self.delay, lambda: LocalExecutor.submit(self, job, deliver))
        else:
            super().submit(job, deliver)


class GatedExecutor(LocalExecutor):
    """Computes results but holds them until `release()` -- a node frozen mid-turn."""

    def __init__(self):
        super().__init__(4)
        self.held: list = []
        self.gate = False

    def submit(self, job, deliver):
        def hold(res):
            if self.gate:
                deliver(res)
            else:
                self.held.append((deliver, res))
        super().submit(job, hold)

    def release(self):
        self.gate = True
        for deliver, res in self.held:
            deliver(res)
        self.held.clear()


async def until(cond, timeout: float = 20.0, every: float = 0.02):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        await asyncio.sleep(every)
    raise AssertionError(f"condition not met within {timeout:.1f}s")
