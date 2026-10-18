"""Scripted seats: zero-cost players for tests, replay and offline runs.

They implement the same ``act`` a model or a human does. ``RandomPlayer`` picks
a legal terminal at random; ``clumsy`` seats first submit garbage so the retry
path is exercised.
"""
from __future__ import annotations

import random
from typing import Any

from ..arena import ActionSpec, Param, ToolSpec


def random_args(tool: ToolSpec, rng: random.Random, view: dict | None = None) -> dict:
    """A legal argument set for one ToolSpec, chosen at random."""
    args: dict[str, Any] = {}
    for k, p in tool.params.items():
        if not p.required and rng.random() < 0.3:
            args[k] = None
            continue
        args[k] = _random_value(p, rng, view, tool.name)
    if tool.name == "pack_plan" and view is not None:
        wolves = [view["you"]] + list(view["private"].get("fellow_wolves", []))
        tactics = list(tool.params["roles"].choices or []) or ["shenshui"]
        args["roles"] = {w: rng.choice(_tactic_pool()) for w in wolves}
    return args


def _tactic_pool() -> list[str]:
    from .strategies.playbook import strategy_ids
    return strategy_ids()


def _random_value(p: Param, rng: random.Random, view: dict | None, tool: str) -> Any:
    if p.choices is not None:
        if not p.choices:
            return None
        return rng.choice(list(p.choices))
    if p.type == "int":
        return 1
    if p.type == "float":
        lo, hi = p.range or (0.0, 1.0)
        return round(lo + rng.random() * (hi - lo), 1)
    if p.type == "str":
        who = view["you"] if view else "?"
        return f"player {who} says something about {tool}"
    if p.type == "list[str]":
        return []
    if p.type == "dict[int,str]":
        return {}
    return None


class RandomPlayer:
    def __init__(self, seed: int, clumsy: bool = False):
        self._rng = random.Random(seed)
        self._clumsy = clumsy

    def act(self, view: dict, task: str, spec: ActionSpec, feedback: str | None) -> Any:
        if self._clumsy and feedback is None:
            return {"name": "???", "args": {}}
        if spec.kind == "tool":
            tool = self._rng.choice(spec.tools)
            return {"name": tool.name, "args": random_args(tool, self._rng, view)}
        if spec.kind == "choice":
            return self._rng.choice(spec.options)
        if spec.kind == "text":
            return f"{view.get('you')} says something about {task}"
        return {k: round(lo + self._rng.random() * (hi - lo), 1) for k, (lo, hi) in spec.fields.items()}
