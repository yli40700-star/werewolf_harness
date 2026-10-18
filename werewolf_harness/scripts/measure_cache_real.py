"""Real prefix-cache hit rate on the Anthropic API: old prompt layout vs the cache layout.

The simulated measurement (measure_prompt_cache.py) said 62.9% -> 68.8%; the real
games played through `claude -p` read back ~8%, because the CLI places the
breakpoints. This plays real games with the breakpoints placed by us
(players/providers/anthropic_api.py) and reads the hit rate from the API's usage:

  old     ContextBuilder(cache_layout=False); breakpoints at the end of the
          system message and the end of the user message
  layout  cache_layout=True, plus a breakpoint after the shared part

hit rate = cache_read / (input + cache_write + cache_read), summed over calls.

    python -m werewolf_harness.scripts.measure_cache_real --seeds 2001-2003 --out runs/cache_real
"""
from __future__ import annotations

import argparse
import json
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor

from ..arena import driver
from ..envs import Werewolf
from ..players import GuardStack, LLMPlayer
from ..players.llm.context import ContextBuilder
from ..players.providers.anthropic_api import AnthropicClient
from ..server.http import read_env

ARMS = {"old": dict(cache_layout=False, shared_breakpoint=False),
        "layout": dict(cache_layout=True, shared_breakpoint=True)}


class Logged(AnthropicClient):
    def __init__(self, *a, rows: list, lock: threading.Lock, seat: int, **kw):
        super().__init__(*a, **kw)
        self.rows, self.lock, self.seat = rows, lock, seat

    def chat(self, *a, **kw):
        r = super().chat(*a, **kw)
        with self.lock:
            self.rows.append({"seat": self.seat, **r.raw["usage"], "latency_ms": r.latency_ms})
        return r


def play(arm: str, seed: int, model: str, key: str, variant: str, out: pathlib.Path) -> dict:
    cfg = ARMS[arm]
    rows: list[dict] = []
    lock = threading.Lock()
    guard = GuardStack(("L1", "L2", "L3"))
    env = Werewolf(variant)
    players = {}
    for s in range(1, int(variant.rstrip("p")) + 1):
        p = LLMPlayer(Logged(model, key, shared_breakpoint=cfg["shared_breakpoint"], rows=rows, lock=lock, seat=s),
                      guard=guard)
        p.context = ContextBuilder(guard, max_steps=p.policy.max_react_steps, cache_layout=cfg["cache_layout"])
        players[s] = p
    ep = driver.run(env, players, seed=seed)
    (out / f"{arm}_{seed}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return {"arm": arm, "seed": seed, "calls": len(rows), "winner": getattr(ep, "winner", None), **totals(rows)}


def totals(rows: list[dict]) -> dict:
    prompt = sum(r["prompt_tokens"] for r in rows)
    read = sum(r["cache_read_input_tokens"] for r in rows)
    write = sum(r["cache_creation_input_tokens"] for r in rows)
    return {"prompt_tokens": prompt, "cache_read": read, "cache_write": write,
            "hit_rate": round(read / prompt, 4) if prompt else 0.0,
            "write_rate": round(write / prompt, 4) if prompt else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="2001-2003")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--variant", default="8p")
    ap.add_argument("--arms", default="old,layout")
    ap.add_argument("--out", default="runs/cache_real")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    key = read_env().get("ANTHROPIC_API_KEY", "")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(arm, s) for arm in a.arms.split(",") for s in seeds]
    with ThreadPoolExecutor(len(jobs)) as ex:
        games = list(ex.map(lambda j: play(j[0], j[1], a.model, key, a.variant, out), jobs))
    summary = {"model": a.model, "variant": a.variant, "games": games, "arms": {}}
    for arm in a.arms.split(","):
        rows = [json.loads(line) for s in seeds for line in (out / f"{arm}_{s}.jsonl").read_text().splitlines() if line]
        summary["arms"][arm] = {"games": len(seeds), "calls": len(rows), **totals(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary["arms"], indent=2))


if __name__ == "__main__":
    main()
