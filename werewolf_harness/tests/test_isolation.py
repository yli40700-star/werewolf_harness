"""G1: what a seat can see is exactly what observe() puts in its view."""
from __future__ import annotations

import json

from ..arena import Terminal, Turn
from ..envs import Werewolf
from .util import SEATS12, Script, first_turn, play_out, play_until, table

W1, SEER, WITCH, GUARD, V1 = 1, 5, 6, 8, 9

# The phases in which the game is still being played. The guarantee below is the guarantee
# for exactly these. `review` is the one phase where the environment opens the deal on
# purpose, and it exists only after the result has already been frozen.
LIVE_PHASES = ("night", "pack_chat", "pack_plan", "campaign", "campaign_vote",
               "day_speech", "day_vote", "interrupt")

VIEW_KEYS_LIVE = {"you", "role", "team", "alive", "round", "phase", "num_players", "rules",
                  "public", "speech_position", "speakers_total", "private", "night"}


def _assert_sealed(env):
    """No seat's view carries the deal, or any key that could smuggle it in."""
    assert env._phase in LIVE_PHASES, env._phase
    for seat in SEATS12:
        v = env.observe(seat)
        blob = json.dumps(v, default=str)
        assert "ground_truth" not in blob and '"roles"' not in blob
        assert "review" not in v and "winner" not in blob
        assert json.dumps(env._roles) not in blob
        assert set(v) <= VIEW_KEYS_LIVE, sorted(set(v) - VIEW_KEYS_LIVE)
        for other, role in SEATS12.items():           # the only role in a view is the seat's own
            if other != seat:
                assert f'"{other}": "{role}"' not in blob


def _mid_game():
    env = table(SEATS12, playbook="static")
    moves = {(SEER, "night"): {"name": "night_check", "args": {"target_id": W1}},
             (GUARD, "night"): {"name": "night_protect", "args": {"target_id": V1}},
             **{(w, "night"): (lambda v, s: {"name": "night_kill", "args": {"target_id": 10}}) for w in (1, 2, 3, 4)}}
    turn = play_until(env, first_turn(env), lambda e, t: t.task == "speak", Script(moves))
    return env


def test_villager_view_has_no_private_keys_of_other_roles():
    env = _mid_game()
    priv = env.observe(V1)["private"]
    for key in ("fellow_wolves", "pack_chat", "pack_roles", "checks", "antidote_available", "seer_log"):
        assert key not in priv


def test_no_view_contains_the_deal():
    _assert_sealed(_mid_game())


def test_no_view_contains_the_deal_at_any_point_while_the_game_is_live():
    """Not one sampled moment: every turn of a whole game, up to the review."""
    env = table(SEATS12, playbook="static")
    turn = first_turn(env)
    script = Script()
    checked = 0
    while isinstance(turn, Turn) and turn.task != "review":
        _assert_sealed(env)
        checked += 1
        turn = env.step(turn.actor, turn.task, script.act(env, turn))
    assert checked > 30, checked


def test_the_deal_opens_in_review_and_never_one_turn_earlier():
    """The single door, and it is a door in time.

    Every live turn is sealed (the test above walks all of them); one step past the last
    of them the phase is `review`, the result is already frozen, and every seat -- alive
    or dead -- holds the whole deal.
    """
    env = table(SEATS12, playbook="static")
    turn = play_until(env, first_turn(env), lambda e, t: t.task == "review")
    assert isinstance(turn, Turn) and turn.task == "review", "the game never reached the review"
    assert env._phase == "review" and env.outcome() is not None
    for seat in SEATS12:
        v = env.observe(seat)
        assert v["review"]["roles"] == dict(SEATS12)
        assert v["review"]["winner"] == env.outcome()["winner"]
        assert v["review"]["pack_chat"] == env.pack_chat_log()
        assert v["review"]["seer_checks"] == env._seer_checks
    # a seat that died still gets the handout, which is the point of a 复盘
    dead = [s for s in SEATS12 if s not in env._alive]
    assert dead and all("review" in env.observe(s) for s in dead)


def test_the_review_handout_closes_again_when_the_review_is_over():
    env = table(SEATS12, playbook="static")
    end = play_out(env, first_turn(env))
    assert isinstance(end, Terminal) and env._phase == "over"
    for seat in SEATS12:
        assert "review" not in env.observe(seat)


def test_the_review_handout_is_a_copy_not_a_handle():
    env = table(SEATS12, playbook="static")
    play_until(env, first_turn(env), lambda e, t: t.task == "review")
    r = env.observe(V1)["review"]
    r["roles"][1] = "villager"
    r["deaths"].clear()
    r["pack_chat"].clear()
    assert env.observe(V1)["review"]["roles"][1] == SEATS12[1]
    assert env.observe(V1)["review"]["deaths"]
    assert env._roles[1] == SEATS12[1]


def test_only_the_seer_sees_checks_and_only_wolves_see_the_pack():
    env = _mid_game()
    assert env.observe(SEER)["private"]["checks"] == [{"round": 1, "target": W1, "is_wolf": True}]
    assert env.observe(W1)["private"]["fellow_wolves"] == [2, 3, 4]
    for seat in SEATS12:
        v = env.observe(seat)
        if seat != SEER:
            assert "checks" not in v["private"]
        if SEATS12[seat] != "werewolf":
            assert "fellow_wolves" not in v["private"] and "pack_chat" not in v["private"]


def test_pack_chat_never_reaches_the_public_transcript():
    env = _mid_game()
    assert env.pack_chat_log()                              # the wolves did talk
    for seat in SEATS12:
        pub = env.observe(seat)["public"]
        assert all(s["kind"] in ("speech", "campaign", "last_words") for s in pub["speeches"])


def test_night_deaths_never_say_how():
    env = _mid_game()
    dead = env.observe(V1)["public"]["dead"]
    assert dead and all(d["cause"] in ("night", "exiled", "shot", "detonated") for d in dead)
    assert all(d["cause"] != "werewolf" and d["cause"] != "witch" for d in dead)


def test_view_is_a_copy_not_a_handle():
    env = _mid_game()
    v = env.observe(V1)
    v["public"]["alive"].clear()
    v["private"]["my_actions"].append({"kind": "forged"})
    assert env.observe(V1)["public"]["alive"]
    assert not any(a.get("kind") == "forged" for a in env.observe(V1)["private"]["my_actions"])


def test_guard_only_learns_his_own_charge_through_his_own_record():
    env = _mid_game()
    mine = env.observe(GUARD)["private"]["my_actions"]
    assert any(a["kind"] == "night_protect" and a["target"] == V1 for a in mine)
    for seat in SEATS12:
        if seat != GUARD:
            assert not any(a.get("kind") == "night_protect" for a in env.observe(seat)["private"]["my_actions"])
