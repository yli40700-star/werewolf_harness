"""The framework layer: driver, retry path, replay, and the no-game-words rule."""
from __future__ import annotations

import pathlib
import re

import pytest

from ..arena import ActionSpec, Param, Record, Terminal, ToolSpec, Turn, replay, run
from ..envs import Werewolf
from ..players import RandomPlayer


def _players(seed, n=12, clumsy=()):
    return {s: RandomPlayer(seed * 100 + s, clumsy=s in clumsy) for s in range(1, n + 1)}


@pytest.mark.parametrize("seed", range(1, 8))
def test_world_is_a_function_of_seed_and_journal(seed):
    ep = run(Werewolf("12p"), _players(seed), seed)
    end = replay(Werewolf("12p"), seed, ep.journal)
    assert isinstance(end, Terminal) and end.outcome == ep.outcome
    mid = replay(Werewolf("12p"), seed, ep.journal, upto=5)
    assert isinstance(mid, Turn)


def test_replay_detects_a_diverging_journal():
    ep = run(Werewolf("12p"), _players(3), 3)
    bad = list(ep.journal)
    bad[2] = Record(actor=99, task=bad[2].task, action=bad[2].action)
    with pytest.raises(AssertionError, match="diverged"):
        replay(Werewolf("12p"), 3, bad)


def test_illegal_answers_are_refused_with_a_reason_then_retried():
    ep = run(Werewolf("12p"), _players(5, clumsy=(3, 7)), 5)
    assert all(not r.forced for r in ep.journal)          # the clumsy seats recovered on the second ask
    assert len(ep.journal) > 20


def test_fallback_after_retries_is_marked_forced():
    class Stubborn:
        def act(self, view, task, spec, feedback):
            return {"name": "nonsense", "args": {}}
    players = _players(2)
    players[9] = Stubborn()
    ep = run(Werewolf("12p"), players, 2)
    forced = [r for r in ep.journal if r.actor == 9]
    assert forced and all(r.forced for r in forced)
    assert all(Werewolf("12p") is not None for _ in forced)


def test_forced_actions_are_legal():
    class Stubborn:
        def act(self, view, task, spec, feedback):
            return "garbage"
    players = _players(4)
    for s in (1, 5, 6, 7, 8):
        players[s] = Stubborn()
    ep = run(Werewolf("12p"), players, 4)           # would raise inside step() if a fallback were illegal
    end = replay(Werewolf("12p"), 4, ep.journal)
    assert isinstance(end, Terminal) and end.outcome == ep.outcome


def test_spec_describe_and_check_share_one_object():
    t = ToolSpec("vote", "Cast. ENDS YOUR TURN.", {"target_id": Param("int", "seat", choices=(2, 3))})
    spec = ActionSpec("tool", tools=(t,), fallback={"name": "vote", "args": {"target_id": 2}})
    assert "[2, 3]" in spec.describe()
    assert spec.check({"name": "vote", "args": {"target_id": 3}}) is None
    assert "not one of [2, 3]" in spec.check({"name": "vote", "args": {"target_id": 4}})
    assert "not available now" in spec.check({"name": "shoot", "args": {}})
    assert spec.check(spec.fallback) is None


def test_framework_layer_contains_no_game_words():
    here = pathlib.Path(__file__).resolve().parent.parent / "arena"
    words = re.compile(r"wolf|vote|witch|seer|refund|bid|sheriff|badge", re.I)
    for f in here.glob("*.py"):
        assert not words.search(f.read_text()), f"{f.name} mentions a game"


def test_every_variant_plays_out_with_scripted_seats():
    from ..envs import VARIANTS
    for name, v in VARIANTS.items():
        ep = run(Werewolf(name), _players(11, n=v.num_players), 11)
        assert ep.outcome["winner"] in ("village", "werewolf")
