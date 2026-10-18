"""How much of every prompt a provider could serve from its prefix cache:
old system-message layout vs the cache layout.

Simulated cache (app/llm/sim.py: 128-token blocks, >=1024 tokens, 5 min TTL) over
the real prompts the harness builds. Two byte orders, because providers differ
in where the tool schemas sit in the cached prefix:

  tools_first  tools, then messages (Anthropic documents this order)
  tools_last   messages, then tools

    python -m werewolf_harness.scripts.measure_prompt_cache --seeds 5
"""
from __future__ import annotations

import argparse
import json

from ..app.llm.sim import PrefixCache, SimClient
from ..arena import driver
from ..envs import Werewolf
from ..players import GuardStack, LLMPlayer
from ..players.llm.context import ContextBuilder


class CountingCache(PrefixCache):
    def __init__(self):
        super().__init__()
        self.prompt = self.cached = self.calls = 0

    def lookup_and_fill(self, text, now=None):
        p, c = super().lookup_and_fill(text, now)
        self.prompt += p
        self.cached += c
        self.calls += 1
        return p, c


def run(layout: bool, tools_first: bool, seeds: list[int]) -> dict:
    prompt = cached = calls = 0
    for seed in seeds:
        cache = CountingCache()
        guard = GuardStack(("L1", "L2", "L3"))
        players = {}
        for s in range(1, 13):
            p = LLMPlayer(SimClient(seed * 100 + s, cache=cache, sleep=False, tools_first=tools_first), guard=guard)
            p.context = ContextBuilder(guard, max_steps=p.policy.max_react_steps, cache_layout=layout)
            players[s] = p
        driver.run(Werewolf("12p"), players, seed=seed)
        prompt, cached, calls = prompt + cache.prompt, cached + cache.cached, calls + cache.calls
    return {"calls": calls, "prompt_tokens": prompt, "cached_tokens": cached,
            "hit_rate": round(cached / prompt, 3) if prompt else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    seeds = list(range(1, a.seeds + 1))
    out = {f"{'tools_first' if tf else 'tools_last'}/{'cache_layout' if lay else 'old_layout'}": run(lay, tf, seeds)
           for tf in (True, False) for lay in (False, True)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
