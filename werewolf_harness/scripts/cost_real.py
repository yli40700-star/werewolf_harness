"""Real model cost per game from the recorded API usage (--record cassettes).

Each recorded reply carries the API's usage: input = uncached input + cache writes +
cache reads, plus output. Priced per million tokens with the official multipliers for the
5-minute cache: writes 1.25x input, reads 0.1x input. Arms are compared seed by seed.

    python -m werewolf_harness.scripts.cost_real strong=runs/costreal/strong routed=runs/costreal/routed --ref strong
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0)}   # USD per 1M tokens, 2026-10-15
WRITE, READ = 1.25, 0.10


def game_cost(cassette: str) -> dict:
    usd = 0.0
    tok = {"input": 0, "write": 0, "read": 0, "output": 0}
    for line in open(cassette, encoding="utf-8"):
        resp = json.loads(line).get("resp")
        if not isinstance(resp, dict) or resp.get("model") not in PRICES:
            continue
        raw = resp.get("raw") or {}
        u = raw.get("usage") or {}
        pi, po = PRICES[resp["model"]]
        cli_usd = raw.get("total_cost_usd")         # claude -p prices its own call at list price (1 h cache writes)
        read, write = int(u.get("cache_read_input_tokens") or 0), int(u.get("cache_creation_input_tokens") or 0)
        plain = int(u.get("prompt_tokens") or 0) - read - write
        out = int(u.get("completion_tokens") or 0)
        usd += cli_usd if cli_usd is not None else (plain * pi + write * pi * WRITE + read * pi * READ + out * po) / 1e6
        for k, v in (("input", plain + read + write), ("write", write), ("read", read), ("output", out)):
            tok[k] += v
    return {"usd": usd, **tok}


def legal_rate(log: str) -> tuple[int, int]:
    d = json.load(open(log, encoding="utf-8"))
    h = d.get("health") or {}
    return h.get("turns", 0) - h.get("forced", 0), h.get("turns", 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="name=dir")
    ap.add_argument("--ref", required=True)
    a = ap.parse_args()
    games: dict[str, dict[int, dict]] = {}
    for spec in a.arms:
        name, d = spec.split("=", 1)
        games[name] = {}
        for log in sorted(glob.glob(f"{d}/g*.json")):
            seed = int(re.search(r"g(\d+)\.json$", log).group(1))
            cas = log.replace(".json", ".cassette.jsonl")
            if os.path.exists(cas):
                games[name][seed] = {**game_cost(cas), "legal": legal_rate(log)}
    ref = games[a.ref]
    for name, gs in games.items():
        seeds = sorted(set(gs) & set(ref))
        usd = [gs[s]["usd"] for s in seeds]
        ins = sum(gs[s]["input"] for s in seeds)
        reads = sum(gs[s]["read"] for s in seeds)
        ok, n = (sum(gs[s]["legal"][i] for s in seeds) for i in (0, 1))
        line = (f"{name}: {len(seeds)} games, ${sum(usd) / max(1, len(seeds)):.3f}/game, "
                f"cache reads {reads / max(1, ins):.1%} of input, legal {ok}/{n} = {ok / max(1, n):.1%}")
        if name != a.ref and seeds:
            r = [gs[s]["usd"] / ref[s]["usd"] for s in seeds]
            bs = sorted(sum(random.choice(r) for _ in r) / len(r) for _ in range(2000))
            line += f", cost vs {a.ref} {sum(r) / len(r):.3f}x [{bs[50]:.3f}, {bs[1950]:.3f}]"
        print(line)


if __name__ == "__main__":
    main()
