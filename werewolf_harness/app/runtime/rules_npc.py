"""What a model seat does when no model can answer: a rule, not a shrug.

The environment's own fallback is the quietest legal action (e.g. vote for the
lowest-numbered living seat). When the model layer is down for a whole turn, the
seat still has its belief table, so a vote goes to the seat it suspects most.
Only used for turns the model could not produce (transport failure), never to
overrule an answer the model gave.
"""
from __future__ import annotations

from typing import Any

from ...arena import ActionSpec


def rule_action(task: str, spec: ActionSpec, belief: dict | None) -> Any:
    fb = spec.fallback
    if task == "vote" and belief:
        tool = spec.tool("vote")
        choices = list(tool.params["target_id"].choices or []) if tool else []
        scored = [(float((belief.get(str(c)) or {}).get("suspicion", 0.5)), -c, c) for c in choices]
        if scored:
            target = max(scored)[2]
            act = {"name": "vote", "args": {"target_id": target, "thought": "rule seat: model unavailable; most suspected"}}
            if spec.check(act) is None:
                return act
    return fb
