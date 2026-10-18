"""Test helpers: build a table with known roles and drive it with a script."""
from __future__ import annotations

from typing import Callable

from ..arena import ActionSpec, Terminal, Turn
from ..envs import Werewolf

SEATS12 = {1: "werewolf", 2: "werewolf", 3: "werewolf", 4: "werewolf", 5: "seer", 6: "witch",
           7: "hunter", 8: "guard", 9: "villager", 10: "villager", 11: "villager", 12: "villager"}


def table(roles: dict[int, str] | None = None, variant: str = "12p", playbook: str = "none", seed: int = 1,
          engine_version: int | None = None, **variant_overrides):
    """A fresh environment whose deal is `roles` (tests may fix the deal; players never can)."""
    from ..envs.werewolf import VARIANTS, Variant
    v = VARIANTS[variant]
    if variant_overrides:
        v = Variant(**{**v.__dict__, **variant_overrides})
    env = Werewolf(v, playbook_mode=playbook, engine_version=engine_version)
    env.reset(seed)
    if roles is not None:
        assert sorted(roles) == list(range(1, v.num_players + 1))
        env._roles = dict(roles)
        env._begin_night()
    return env


def default_action(view: dict, task: str, spec: ActionSpec) -> dict:
    """The quietest legal move: skip, pass, hold, speak nothing, vote the first seat."""
    names = [t.name for t in spec.tools]
    for quiet in ("night_skip", "campaign_pass", "hunter_hold", "badge_tear"):
        if quiet in names:
            return {"name": quiet, "args": {"reason": "test"} if quiet != "night_skip" else {}}
    t = spec.tools[0]
    args = {}
    for k, p in t.params.items():
        if p.type == "dict[int,str]":                 # a whitelist here is per value, not the whole map
            wolves = [view["you"]] + list(view["private"].get("fellow_wolves", []))
            args[k] = {w: "shenshui" for w in wolves}
        elif p.choices is not None:
            args[k] = p.choices[0] if p.choices else None
        elif p.type == "str":
            args[k] = "..."
    return {"name": t.name, "args": args}


class Script:
    """Answer (actor, task) from a table of actions; anything else takes the default."""

    def __init__(self, moves: dict[tuple[int, str], dict | Callable] | None = None):
        self.moves = dict(moves or {})
        self.log: list[tuple[int, str, dict]] = []

    def act(self, env: Werewolf, turn: Turn) -> dict:
        view = env.observe(turn.actor)
        spec = env.action_space(turn.actor, turn.task)
        move = self.moves.get((turn.actor, turn.task))
        action = move(view, spec) if callable(move) else (move or default_action(view, turn.task, spec))
        reason = spec.check(action)
        assert reason is None, f"script gave an illegal action for {turn}: {reason}"
        self.log.append((turn.actor, turn.task, action))
        return action


def play_until(env: Werewolf, turn, stop: Callable[[Werewolf, Turn], bool], script: Script | None = None, limit: int = 500):
    """Drive the table from `turn` until `stop(env, turn)` is true (or the game ends). Returns the turn stopped at."""
    script = script or Script()
    for _ in range(limit):
        if isinstance(turn, Terminal) or stop(env, turn):
            return turn
        turn = env.step(turn.actor, turn.task, script.act(env, turn))
    raise AssertionError("play_until did not stop")


def play_out(env: Werewolf, turn, script: Script | None = None, limit: int = 2000) -> Terminal:
    script = script or Script()
    for _ in range(limit):
        if isinstance(turn, Terminal):
            return turn
        turn = env.step(turn.actor, turn.task, script.act(env, turn))
    raise AssertionError("game did not end")


def first_turn(env: Werewolf):
    return env._next()


def tool_names(env: Werewolf, actor: int, task: str) -> list[str]:
    return [t.name for t in env.action_space(actor, task).tools]


def choices(env: Werewolf, actor: int, task: str, tool: str, param: str = "target_id"):
    spec = env.action_space(actor, task)
    t = spec.tool(tool)
    return list(t.params[param].choices) if t else None
