"""What the actor ships to a model worker: a seat's view and its action spec as JSON.

The worker no longer rebuilds the game; it acts on what the job carries. So the trip
through JSON must lose nothing the player reads: the same prompts, the same tool
schemas, the same check() verdicts -- including the wolf pack's staffing rule, which
used to be a Python callable and is now data (AssignRule).
"""
from __future__ import annotations

import json
import random

import pytest

from ..arena import ActionSpec, AssignRule, Param, Terminal, ToolSpec, Turn
from ..arena.wire import decode, encode
from ..envs import VARIANTS, Werewolf
from ..players import GuardStack, LLMPlayer, MockClient, RandomPlayer
from ..players.providers.base import LLMClient
from ..players.strategies.playbook import strategy_ids
from .util import SEATS12, first_turn, play_until, table


def _trip(x):
    return decode(json.loads(json.dumps(encode(x), ensure_ascii=False)))


def _spec_trip(spec: ActionSpec) -> ActionSpec:
    return ActionSpec.from_dict(json.loads(json.dumps(spec.to_dict(), ensure_ascii=False)))


def _same(a, b) -> bool:
    """Equal, and of the same types all the way down (1 == 1.0 and {1: x} != {"1": x})."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a) == list(b) and all(type(k) is type(j) for k, j in zip(a, b)) \
            and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _states(variant: str, seed: int, playbook: str = "static"):
    """Every (env, turn) of one scripted-random game."""
    env = Werewolf(variant, playbook_mode=playbook)
    turn = env.reset(seed)
    seats: dict = {}
    while isinstance(turn, Turn):
        yield env, turn
        spec = env.action_space(turn.actor, turn.task)
        p = seats.setdefault(turn.actor, RandomPlayer(seed * 100 + turn.actor))
        a = p.act(env.observe(turn.actor), turn.task, spec, None)
        turn = env.step(turn.actor, turn.task, a if spec.check(a) is None else spec.fallback)


def test_wire_keeps_int_keys_tuples_and_order():
    x = {"votes": {3: 1, 1: 2}, "pair": (1, "a"), "nested": [{2: (3, 4)}], "s": "x", "f": 1.5, "b": True,
         "n": None, "__map__": "a key that looks like the tag"}
    assert _same(_trip(x), x)
    with pytest.raises(TypeError):
        encode({"a": {1, 2}})                              # not JSON: refused, not stringified


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_every_view_and_spec_of_a_game_survives_the_trip(variant):
    n = VARIANTS[variant].num_players
    turns = 0
    for env, turn in _states(variant, seed=11):
        turns += 1
        for s in range(1, n + 1):
            v = env.observe(s)
            assert _same(_trip(v), v), (turn, s)
        spec = env.action_space(turn.actor, turn.task)
        back = _spec_trip(spec)
        assert back == spec and _same(back.fallback, spec.fallback)
        assert back.describe() == spec.describe()
        assert [t.openai_schema() for t in back.tools] == [t.openai_schema() for t in spec.tools]
        assert back.check(spec.fallback) == spec.check(spec.fallback)
    assert turns > 20


def _old_pack_rule(wolves):
    """The rule as it was written before (a closure in envs/werewolf.py), kept as the reference."""
    def validate(args: dict) -> str | None:
        roles = args.get("roles") or {}
        unknown = sorted(set(roles.values()) - set(strategy_ids()))
        if unknown:
            return f"unknown tactic(s): {unknown}"
        off = sorted(set(roles) - set(wolves))
        if off:
            return f"player(s) {off} are not living wolves; the pack is {wolves}"
        missing = sorted(set(wolves) - set(roles))
        if missing:
            return f"every living wolf needs a tactic; {missing} have none"
        return None
    return validate


def _pack_plan_spec():
    env = table(SEATS12, playbook="static")
    turn = play_until(env, first_turn(env), lambda e, t: t.task == "pack_plan")
    assert isinstance(turn, Turn) and turn.task == "pack_plan"
    return env, turn, env.action_space(turn.actor, turn.task)


def test_the_pack_plan_rule_is_data_and_refuses_the_same_inputs_after_the_trip():
    env, turn, spec = _pack_plan_spec()
    plan = spec.tool("pack_plan")
    assert plan.validate is None and len(plan.rules) == 1
    wolves = env._living_wolves()
    old = ToolSpec(plan.name, plan.description, plan.params, validate=_old_pack_rule(wolves))
    back = _spec_trip(spec).tool("pack_plan")
    assert back == plan
    ids = strategy_ids()
    p = plan.params["plan"].choices[0]
    good = {w: ids[0] for w in wolves}
    cases = [good, {**good, wolves[0]: "no-such-tactic"}, {**good, 5: ids[1]}, {w: ids[0] for w in wolves[1:]},
             {}, {str(w): ids[2] for w in wolves}, {**{str(w): ids[0] for w in wolves}, "9": ids[0]},
             {w: ids[0] for w in wolves[:2]} | {9: "bogus"}]
    rng = random.Random(5)
    for _ in range(300):                                   # random staffings over seats 1..12, known and unknown ids
        keys = rng.sample(range(1, 13), rng.randint(0, 6))
        cases.append({k: rng.choice(ids + ["zzz"]) for k in keys})
    refused = 0
    for roles in cases:
        args = {"plan": p, "roles": roles}
        want = old.check(args)
        assert plan.check(args) == want, roles
        assert back.check(args) == want, roles
        assert spec.check({"name": "pack_plan", "args": args}) == want
        refused += want is not None
    assert plan.check({"plan": p, "roles": good}) is None
    assert "unknown tactic" in back.check({"plan": p, "roles": {**good, wolves[0]: "no-such-tactic"}}) \
        or "not one of" in back.check({"plan": p, "roles": {**good, wolves[0]: "no-such-tactic"}})
    assert "are not living wolves" in back.check({"plan": p, "roles": {**good, 5: ids[1]}})
    assert "needs a tactic" in back.check({"plan": p, "roles": {w: ids[0] for w in wolves[1:]}})
    assert refused > 100


def test_assign_rule_wording_and_order():
    r = AssignRule("m", keys=(2, 7), values=("a", "b"), unknown="u {bad}", extra="e {bad} {keys}", missing="m {bad}")
    assert r.check({"m": {2: "a", 7: "b"}}) is None
    assert r.check({"m": {2: "c", 9: "a"}}) == "u ['c']"           # unknown value first
    assert r.check({"m": {2: "a", 9: "a"}}) == "e [9] [2, 7]"
    assert r.check({"m": {2: "a"}}) == "m [7]"
    assert AssignRule.from_dict(json.loads(json.dumps(r.to_dict()))) == r


def test_a_callable_rule_cannot_be_shipped():
    t = ToolSpec("x", "d", {"a": Param("int")}, validate=lambda args: None)
    with pytest.raises(ValueError):
        ActionSpec("tool", tools=(t,)).to_dict()


class _Seen(LLMClient):
    """Delegates to a mock seat and keeps every request it was sent."""

    def __init__(self, inner: LLMClient):
        self.inner, self.name, self.tool_mode = inner, inner.name, inner.tool_mode
        self.seen: list[str] = []

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0):
        self.seen.append(json.dumps([messages, tools, temperature, max_tokens], sort_keys=True, ensure_ascii=False,
                                    default=str))
        return self.inner.chat(messages, tools, temperature=temperature, max_tokens=max_tokens, timeout=timeout)


@pytest.mark.parametrize("variant,seed", [("12p", 3), ("8p+sheriff", 4)])
def test_a_seat_acting_on_the_shipped_copy_sends_the_same_prompts_and_answers_the_same(variant, seed):
    """Two copies of every seat: one acts on the view/spec the engine computed, the other on
    the JSON copy a worker would receive. Every request to the model and every answer match."""
    env = Werewolf(variant, playbook_mode="static")
    turn = env.reset(seed)
    n = VARIANTS[variant].num_players

    def seat(s):
        c = _Seen(MockClient(seed * 100 + s))
        return LLMPlayer(c, guard=GuardStack(("L1", "L2", "L3"))), c
    here = {s: seat(s) for s in range(1, n + 1)}
    there = {s: seat(s) for s in range(1, n + 1)}
    turns = 0
    while isinstance(turn, Turn):
        view, spec = env.observe(turn.actor), env.action_space(turn.actor, turn.task)
        a = here[turn.actor][0].act(view, turn.task, spec, None)
        b = there[turn.actor][0].act(_trip(view), turn.task, _spec_trip(spec), None)
        assert _same(a, b), turn
        assert here[turn.actor][1].seen == there[turn.actor][1].seen, turn
        assert _spec_trip(spec).check(a) == spec.check(a)
        turn = env.step(turn.actor, turn.task, a if spec.check(a) is None else spec.fallback)
        turns += 1
    assert isinstance(turn, Terminal) and turns > 20
    assert sum(len(c.seen) for _p, c in here.values()) > turns
