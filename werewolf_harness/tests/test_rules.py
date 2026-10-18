"""Rules consistency: every sentence in rules_text() that states engine behaviour is played out here."""
from __future__ import annotations

import pytest

from ..arena import Terminal, Turn
from ..envs.werewolf import MAX_ROUNDS, SHERIFF_VOTE_WEIGHT
from .util import SEATS12, Script, choices, first_turn, play_out, play_until, table, tool_names

W1, W2, W3, W4, SEER, WITCH, HUNTER, GUARD = 1, 2, 3, 4, 5, 6, 7, 8
V1, V2, V3, V4 = 9, 10, 11, 12


def _night(moves: dict, roles=None, **kw):
    env = table(roles or SEATS12, **kw)
    turn = first_turn(env)
    script = Script(moves)
    # play the first night through to the first day (or the first interrupt)
    turn = play_until(env, turn, lambda e, t: t.task not in ("night", "pack_chat", "pack_plan"), script)
    return env, turn, script


def kill(target):
    return lambda view, spec: {"name": "night_kill", "args": {"target_id": target}}


# ---------------------------------------------------------------- night order

def test_night_order_is_seer_guard_wolves_witch():
    env = table(SEATS12)
    order = []
    turn = first_turn(env)
    while isinstance(turn, Turn) and turn.task == "night":
        order.append(turn.actor)
        turn = env.step(turn.actor, "night", Script().act(env, turn))
    assert order == [SEER, GUARD, W1, W2, W3, W4, WITCH]


# ---------------------------------------------------------------- guard rules

def test_guard_may_protect_himself():
    env = table(SEATS12)
    play_until(env, first_turn(env), lambda e, t: t.actor == GUARD)
    assert GUARD in choices(env, GUARD, "night", "night_protect")
    assert "The guard may protect himself." in env.rules_text()


def test_guard_may_not_protect_same_seat_twice_running():
    moves = {(GUARD, "night"): {"name": "night_protect", "args": {"target_id": V1}}}
    env, turn, script = _night(moves)
    # go to the second night's guard turn
    turn = play_until(env, turn, lambda e, t: t.task == "night" and t.actor == GUARD and e._round == 2, script)
    assert V1 not in choices(env, GUARD, "night", "night_protect")
    assert "may not protect the same seat twice running" in env.rules_text()


def test_guard_stops_the_knife():
    moves = {(GUARD, "night"): {"name": "night_protect", "args": {"target_id": V1}},
             **{(w, "night"): kill(V1) for w in (W1, W2, W3, W4)}}
    env, _, _ = _night(moves)
    assert V1 in env._alive
    assert env._wasted_knives == [{"round": 1, "target": V1}]


def test_guard_does_not_stop_the_poison():
    moves = {(GUARD, "night"): {"name": "night_protect", "args": {"target_id": V1}},
             **{(w, "night"): kill(V2) for w in (W1, W2, W3, W4)},
             (WITCH, "night"): {"name": "night_poison", "args": {"target_id": V1}}}
    env, _, _ = _night(moves)
    assert V1 not in env._alive and V2 not in env._alive
    assert "does not stop the witch's poison" in env.rules_text()


def test_guard_and_witch_on_same_seat_survives_by_default():
    moves = {(GUARD, "night"): {"name": "night_protect", "args": {"target_id": V1}},
             **{(w, "night"): kill(V1) for w in (W1, W2, W3, W4)},
             (WITCH, "night"): {"name": "night_save", "args": {"target_id": V1}}}
    env, _, _ = _night(moves)
    assert V1 in env._alive
    assert "do not cancel each other out" in env.rules_text()


def test_guard_and_witch_overlap_can_be_switched_to_kill():
    moves = {(GUARD, "night"): {"name": "night_protect", "args": {"target_id": V1}},
             **{(w, "night"): kill(V1) for w in (W1, W2, W3, W4)},
             (WITCH, "night"): {"name": "night_save", "args": {"target_id": V1}}}
    env, _, _ = _night(moves, guard_witch_overlap_kills=True)
    assert V1 not in env._alive
    assert "cancel each other out" in env.rules_text() and "dies" in env.rules_text()


# ---------------------------------------------------------------- witch rules

def test_witch_may_not_save_herself():
    moves = {(w, "night"): kill(WITCH) for w in (W1, W2, W3, W4)}
    env = table(SEATS12)
    turn = play_until(env, first_turn(env), lambda e, t: t.actor == WITCH, Script(moves))
    assert "night_save" not in tool_names(env, WITCH, "night")
    assert env.observe(WITCH)["night"]["victims_tonight"] == [WITCH]
    assert "may not use her antidote on herself" in env.rules_text()


def test_witch_sees_victim_and_nobody_else_does():
    moves = {(w, "night"): kill(V1) for w in (W1, W2, W3, W4)}
    env = table(SEATS12)
    play_until(env, first_turn(env), lambda e, t: t.actor == WITCH, Script(moves))
    assert env.observe(WITCH)["night"]["victims_tonight"] == [V1]
    for seat in (SEER, GUARD, V1, W1):
        assert "night" not in env.observe(seat)


def test_witch_potions_are_single_use_and_poison_never_targets_the_victim():
    moves = {**{(w, "night"): kill(V1) for w in (W1, W2, W3, W4)},
             (WITCH, "night"): {"name": "night_save", "args": {"target_id": V1}}}
    env, turn, script = _night(moves)
    assert V1 in env._alive
    turn = play_until(env, turn, lambda e, t: t.actor == WITCH and t.task == "night", script)
    assert "night_save" not in tool_names(env, WITCH, "night")
    assert WITCH not in choices(env, WITCH, "night", "night_poison")


def test_pack_majority_with_first_named_breaking_ties():
    moves = {(W1, "night"): kill(V1), (W2, "night"): kill(V2), (W3, "night"): kill(V2), (W4, "night"): kill(V1)}
    env, _, _ = _night(moves)
    assert V1 not in env._alive and V2 in env._alive          # 2-2, W1 named first


def test_a_skipping_wolf_does_not_cancel_the_pack():
    moves = {(W1, "night"): kill(V1), (W2, "night"): kill(V1), (W3, "night"): kill(V1)}   # W4 skips by default
    env, _, _ = _night(moves)
    assert V1 not in env._alive


# ---------------------------------------------------------------- hunter

@pytest.mark.parametrize("how", ["werewolf", "witch", "vote"])
def test_hunter_shoots_whatever_killed_him(how):
    if how == "werewolf":
        moves = {(w, "night"): kill(HUNTER) for w in (W1, W2, W3, W4)}
    elif how == "witch":
        moves = {**{(w, "night"): kill(V1) for w in (W1, W2, W3, W4)},
                 (WITCH, "night"): {"name": "night_poison", "args": {"target_id": HUNTER}}}
    else:
        moves = {(s, "vote"): {"name": "vote", "args": {"target_id": HUNTER}} for s in SEATS12 if s != HUNTER}
    env = table(SEATS12)
    turn = play_until(env, first_turn(env), lambda e, t: t.task == "hunter_shoot", Script(moves))
    assert turn == Turn(HUNTER, "hunter_shoot")
    assert env._death_cause(HUNTER) == how
    assert "Nothing silences the shot" in env.rules_text()


def test_hunter_shot_is_public_and_the_victim_owes_no_last_words():
    moves = {**{(w, "night"): kill(HUNTER) for w in (W1, W2, W3, W4)},
             (HUNTER, "hunter_shoot"): {"name": "hunter_shoot", "args": {"target_id": W1, "reason": "test"}}}
    env = table(SEATS12)
    turn = play_until(env, first_turn(env), lambda e, t: t.task not in ("night", "hunter_shoot"), Script(moves))
    assert W1 not in env._alive
    assert env.observe(V1)["public"]["shots"] == [{"round": 1, "by": HUNTER, "target": W1}]
    assert W1 not in env._pending_last_words
    assert turn == Turn(HUNTER, "last_words")           # night-1 death: the hunter gets his


# ---------------------------------------------------------------- sheriff

def _to_day(moves=None):
    env, turn, script = _night(moves or {})
    return env, turn, script


def test_election_on_day_one_and_candidates_do_not_vote():
    env, turn, script = _to_day()
    assert turn.task == "campaign"
    script.moves[(V1, "campaign")] = {"name": "campaign_run", "args": {"content": "me"}}
    script.moves[(V2, "campaign")] = {"name": "campaign_run", "args": {"content": "me too"}}
    turn = play_until(env, turn, lambda e, t: t.task == "campaign_vote", script)
    assert env._candidates == [V1, V2]
    assert V1 not in env._electorate() and V2 not in env._electorate()
    assert choices(env, turn.actor, "campaign_vote", "campaign_vote") == [V1, V2]


def test_single_candidate_takes_the_badge_without_a_ballot():
    env, turn, script = _to_day()
    script.moves[(V1, "campaign")] = {"name": "campaign_run", "args": {"content": "me"}}
    turn = play_until(env, turn, lambda e, t: t.task == "speak", script)
    assert env._sheriff == V1
    assert turn.actor == V1                              # the sheriff speaks first


def test_sheriff_ballot_counts_one_and_a_half():
    env, turn, script = _to_day()
    script.moves[(V1, "campaign")] = {"name": "campaign_run", "args": {"content": "me"}}
    alive = sorted(env._alive)
    # 5 vote W1 (incl. sheriff = 5.5), 5 vote W2 (5.0), rest vote elsewhere
    rest = [s for s in alive if s not in (V1, W1, W2)]          # 9 seats
    a, b = rest[:4], rest[4:9]                                  # sheriff+4 -> W1 = 5.5 ; 5 -> W2 = 5.0
    for s in [V1] + a:
        script.moves[(s, "vote")] = {"name": "vote", "args": {"target_id": W1}}
    for s in b:
        script.moves[(s, "vote")] = {"name": "vote", "args": {"target_id": W2}}
    script.moves[(W1, "vote")] = {"name": "vote", "args": {"target_id": W3}}
    script.moves[(W2, "vote")] = {"name": "vote", "args": {"target_id": W3}}
    play_until(env, turn, lambda e, t: e._round == 2 or isinstance(t, Terminal), script)
    assert env._exiles[1] == W1
    assert SHERIFF_VOTE_WEIGHT == 1.5


def test_badge_passes_on_any_death_including_a_shot():
    env, turn, script = _to_day()
    script.moves[(V1, "campaign")] = {"name": "campaign_run", "args": {"content": "me"}}
    # night 2: wolves kill the hunter, hunter shoots the sheriff
    script.moves.update({(w, "night"): kill(HUNTER) for w in (W1, W2, W3, W4)})
    script.moves[(HUNTER, "hunter_shoot")] = {"name": "hunter_shoot", "args": {"target_id": V1, "reason": "test"}}
    turn = play_until(env, turn, lambda e, t: t.task == "badge", script)
    assert turn == Turn(V1, "badge")
    assert env._death_cause(V1) == "hunter"


def _tied_day(engine_version):
    """Day one, everyone's ballot scripted so that W1 and V2 tie 5-5 (their own ballots go elsewhere)."""
    env, turn, script = _night({}, engine_version=engine_version)
    alive = sorted(env._alive)
    half = len(alive) // 2
    for s in alive[:half]:
        script.moves[(s, "vote")] = {"name": "vote", "args": {"target_id": W1 if s != W1 else W2}}
    for s in alive[half:]:
        script.moves[(s, "vote")] = {"name": "vote", "args": {"target_id": V2 if s != V2 else V3}}
    return env, turn, script


def test_tie_exiles_nobody_under_engine_v1():
    env, turn, script = _tied_day(engine_version=1)
    play_until(env, turn, lambda e, t: e._round == 2, script)
    assert env._exiles[1] is None
    assert len(env._alive) == 12
    assert "a tie exiles nobody" in env.rules_text()
    assert "tie_votes" not in env.observe(V1)["public"]          # v1 views do not grow the v2 keys


def test_tie_is_revoted_once_between_the_tied_under_engine_v2():
    env, turn, script = _tied_day(engine_version=2)
    turn = play_until(env, turn, lambda e, t: bool(e._revote), script)
    assert env._revote == [W1, V2] and turn.task == "vote"
    assert W1 not in env._day_voters and V2 not in env._day_voters and len(env._day_voters) == 10
    assert choices(env, turn.actor, "vote", "vote") == [W1, V2]
    first = env.observe(V1)["public"]["tie_votes"][1]
    assert first[W2] == W1 and first[V1] == V2 and len(first) == 12                  # the first ballot stays on the record
    # the same split again -> a second tie, which exiles nobody
    play_until(env, turn, lambda e, t: e._round == 2, script)
    assert env._exiles[1] is None and len(env._alive) == 12 and env._revote == []
    assert "held once more between the tied players" in env.rules_text()


def test_a_revote_can_exile_one_of_the_tied():
    env, turn, script = _tied_day(engine_version=2)
    turn = play_until(env, turn, lambda e, t: bool(e._revote), script)
    for s in env._day_voters:
        script.moves[(s, "vote")] = {"name": "vote", "args": {"target_id": W1}}
    play_until(env, turn, lambda e, t: e._round == 2 or t.task == "last_words", script)
    assert env._exiles[1] == W1 and W1 not in env._alive


def test_vote_has_no_abstention_and_excludes_self():
    env, turn, script = _to_day()
    turn = play_until(env, turn, lambda e, t: t.task == "vote", script)
    spec = env.action_space(turn.actor, "vote")
    p = spec.tool("vote").params["target_id"]
    assert p.required and turn.actor not in p.choices
    assert "no way to abstain" in env.rules_text() or "everyone votes" in env.rules_text()


# ---------------------------------------------------------------- last words

def test_exile_gets_last_words_and_night_death_only_on_night_one():
    moves = {(w, "night"): kill(V1) for w in (W1, W2, W3, W4)}
    env, turn, script = _night(moves)
    assert turn == Turn(V1, "last_words")
    for s in SEATS12:
        script.moves[(s, "vote")] = {"name": "vote", "args": {"target_id": V2 if s != V2 else V3}}
    turn = play_until(env, turn, lambda e, t: t.task == "last_words" and t.actor == V2, script)
    assert turn.actor == V2
    # night 2 death: no last words
    script.moves.update({(w, "night"): kill(V3) for w in (W1, W2, W3, W4)})
    turn = play_until(env, turn, lambda e, t: e._round == 2 and t.task not in ("night", "last_words"), script)
    assert V3 not in env._alive and V3 not in env._last_words_given and V3 not in env._pending_last_words


# ---------------------------------------------------------------- self-detonate

def test_self_detonate_ends_the_day_with_no_vote_and_burns_the_badge():
    env, turn, script = _to_day()
    script.moves[(W1, "campaign")] = {"name": "campaign_run", "args": {"content": "me"}}
    turn = play_until(env, turn, lambda e, t: t.task == "speak" and t.actor == W1, script)
    assert env._sheriff == W1
    turn = env.step(W1, "speak", {"name": "self_detonate", "args": {"reason": "test"}})
    assert env._round == 2 and W1 not in env._alive
    assert env._exiles[1] is None and env._sheriff is None
    assert W1 not in env._pending_last_words
    assert env.observe(V1)["public"]["dead"][-1]["cause"] == "detonated"


def test_only_wolves_are_offered_self_detonate():
    env, turn, script = _to_day()
    turn = play_until(env, turn, lambda e, t: t.task == "speak", script)
    assert "self_detonate" not in tool_names(env, V1, "speak")
    assert "self_detonate" in tool_names(env, W1, "speak")


# ---------------------------------------------------------------- win conditions

def test_village_wins_when_all_wolves_are_dead():
    env, turn, script = _to_day()
    for w in (W1, W2, W3, W4):
        for s in SEATS12:
            script.moves[(s, "vote")] = {"name": "vote", "args": {"target_id": w if s != w else V1}}
        turn = play_until(env, turn, lambda e, t: e._round > w or isinstance(t, Terminal), script)
    assert isinstance(turn, Terminal) and turn.outcome["winner"] == "village"


def test_wolves_win_on_parity():
    env = table(SEATS12)
    env._alive = {W1, W2, V1, V2, V3}
    env._begin_night()
    turn = env._next()
    moves = {(w, "night"): kill(V1) for w in (W1, W2)}
    turn = play_out(env, turn, Script(moves))
    assert turn.outcome["winner"] == "werewolf"


def test_round_cap_hands_the_game_to_the_wolves():
    env = table(SEATS12)
    env._round = MAX_ROUNDS
    env._begin_night()
    turn = play_out(env, env._next(), Script())
    assert turn.outcome["winner"] == "werewolf" and turn.outcome["round"] == MAX_ROUNDS + 1
    assert f"round {MAX_ROUNDS} unresolved" in env.rules_text()


# ---------------------------------------------------------------- seer

def test_seer_cannot_recheck_and_learns_alignment():
    moves = {(SEER, "night"): {"name": "night_check", "args": {"target_id": W1}}}
    env, turn, script = _night(moves)
    assert env.observe(SEER)["private"]["checks"] == [{"round": 1, "target": W1, "is_wolf": True}]
    turn = play_until(env, turn, lambda e, t: t.actor == SEER and t.task == "night", script)
    assert W1 not in choices(env, SEER, "night", "night_check")
    assert SEER not in choices(env, SEER, "night", "night_check")


# ---------------------------------------------------------------- pack planning

def test_pack_phases_run_once_in_static_mode_and_require_every_wolf():
    env = table(SEATS12, playbook="static")
    turn = first_turn(env)
    assert turn == Turn(W1, "pack_chat")
    turn = play_until(env, turn, lambda e, t: t.task == "pack_plan")
    spec = env.action_space(W1, "pack_plan")
    bad = {"name": "pack_plan", "args": {"plan": "standard", "roles": {1: "hantiao"}, "vote_target": None}}
    assert "every living wolf needs a tactic" in spec.check(bad)
    good = {"name": "pack_plan", "args": {"plan": "standard", "roles": {1: "hantiao", 2: "chongfeng", 3: "shenshui", 4: "shenshui"}, "vote_target": V1}}
    assert spec.check(good) is None
    turn = env.step(W1, "pack_plan", good)
    assert turn.task == "night"
    for w in (W2, W3, W4):
        assert env.observe(w)["private"]["pack_vote_target"] == V1
    turn = play_until(env, turn, lambda e, t: e._round == 2 and t.task not in ("last_words", "hunter_shoot", "badge"))
    assert turn.task == "night"                          # no second chat in static mode


def test_playbook_none_has_no_pack_phases():
    env = table(SEATS12, playbook="none")
    assert first_turn(env).task == "night"


# ---------------------------------------------------------------- pack chat: talk until done

SAY = {"name": "pack_say", "args": {"content": "我有话说。", "tactic": None}}
PASS = {"name": "pack_pass", "args": {}}


def _chat(env, turn, *actions):
    """Feed the night channel one action per wolf turn, in the order the engine asks."""
    for a in actions:
        assert isinstance(turn, Turn) and turn.task == "pack_chat", turn
        turn = env.step(turn.actor, turn.task, a)
    return turn


def test_pack_chat_offers_both_say_and_pass():
    env = table(SEATS12, playbook="static")
    assert tool_names(env, W1, "pack_chat") == ["pack_say", "pack_pass"]
    assert env.action_space(W1, "pack_chat").fallback["name"] == "pack_pass"


def test_one_pass_ends_nothing_the_floor_just_moves_on():
    env = table(SEATS12, playbook="static")
    turn = _chat(env, first_turn(env), PASS)
    assert turn == Turn(W2, "pack_chat")                 # still talking, next wolf's turn
    assert env.pack_chat_log() == []                     # a pass says nothing


def test_a_whole_round_of_passes_ends_the_discussion():
    env = table(SEATS12, playbook="static")
    turn = _chat(env, first_turn(env), PASS, PASS, PASS, PASS)
    assert turn.task == "pack_plan"                      # chat over, on to the plan
    assert env.pack_chat_log() == []
    assert "one whole round goes by with every living wolf passing" in env.rules_text()


def test_a_wolf_who_passed_speaks_again_after_a_packmate_talks():
    env = table(SEATS12, playbook="static")
    # round 1: W1 passes, W2 speaks, W3 and W4 pass -- the round was not silent
    turn = _chat(env, first_turn(env), PASS, SAY, PASS, PASS)
    assert turn == Turn(W1, "pack_chat")                 # round 2, and W1 has the floor again
    turn = _chat(env, turn, SAY)
    assert [e["player_id"] for e in env.pack_chat_log()] == [W2, W1]
    assert "speaks again in a later round" in env.rules_text()


def test_the_round_cap_ends_a_pack_that_never_stops_talking():
    env = table(SEATS12, playbook="static", chat_passes=2)
    turn = first_turn(env)
    said = 0
    while isinstance(turn, Turn) and turn.task == "pack_chat":
        turn = env.step(turn.actor, turn.task, SAY)
        said += 1
    assert said == 2 * 4                                 # 2 rounds x 4 living wolves, then it stops
    assert turn.task == "pack_plan"


def test_the_chat_rule_sentence_is_rendered_from_the_variant():
    assert "after 2 rounds" in table(SEATS12, playbook="static", chat_passes=2).rules_text()
    assert "after 5 rounds" in table(SEATS12, playbook="static", chat_passes=5).rules_text()
    assert "after 3 rounds" in table(SEATS12, playbook="static").rules_text()     # the default
    assert "private night channel" not in table(SEATS12, playbook="none").rules_text()


# ---------------------------------------------------------------- 复盘 (post-game review)
#
# Cost note: one review round is one model call per SEAT, alive and dead alike
# (12p = 12 calls a round), which is why the cap is a rule and not a convention.

RSAY = {"name": "review_say", "args": {"content": "我当时判断错了。"}}
RPASS = {"name": "review_pass", "args": {}}


def _tiny(review_rounds: int = 2):
    """A three-seat table that reaches its result on night one: the wolf knifes a
    villager and the pack is instantly at parity. Small on purpose -- a review is
    every seat, so a 12-seat review is 12 turns a round."""
    from ..envs.werewolf import Variant, Werewolf
    env = Werewolf(Variant("tiny", {"werewolf": 1, "villager": 2}, review_rounds=review_rounds),
                   playbook_mode="none")
    env.reset(1)
    env._roles = {1: "werewolf", 2: "villager", 3: "villager"}
    env._begin_night()
    turn = env.step(1, "night", {"name": "night_kill", "args": {"target_id": 2}})
    return env, turn


def test_the_review_opens_only_after_the_result_is_fixed():
    env, turn = _tiny()
    assert env._phase == "review" and turn == Turn(1, "review")
    assert env._winner == "werewolf"
    assert env.outcome() == {"winner": "werewolf", "round": 1, "roles": dict(env._roles)}


def test_a_review_turn_cannot_change_the_outcome():
    env, turn = _tiny(review_rounds=3)
    frozen = env.outcome()
    before = (sorted(env._alive), len(env._deaths), env._winner)
    seen = 0
    while isinstance(turn, Turn):
        assert turn.task == "review"
        turn = env.step(turn.actor, turn.task, RSAY)
        seen += 1
        # nothing a review turn can call touches life, death or the result
        assert env.outcome() == frozen
        assert (sorted(env._alive), len(env._deaths), env._winner) == before
    assert seen == 3 * 3                                   # 3 rounds x 3 seats
    assert turn.outcome == frozen
    # and the result cannot be re-decided, whatever the board now looks like
    env._alive = {2, 3}
    assert env._check_end() is True and env.outcome() == frozen


def test_the_review_is_offered_to_the_dead():
    env, turn = _tiny()
    assert 2 not in env._alive                             # seat 2 was knifed on night one
    seats = []
    while isinstance(turn, Turn):
        seats.append(turn.actor)
        turn = env.step(turn.actor, turn.task, RPASS if turn.actor == 2 else RSAY)
    assert seats[:3] == [1, 2, 3]                          # everyone, in seat order
    assert tool_names(env, 2, "review") == ["review_say", "review_pass"]


def test_a_whole_round_of_passes_ends_the_review():
    env, turn = _tiny(review_rounds=5)
    for _ in range(3):
        assert isinstance(turn, Turn) and turn.task == "review"
        turn = env.step(turn.actor, turn.task, RPASS)
    assert isinstance(turn, Terminal)
    assert env.review_log() == []                           # a pass says nothing
    assert "one whole round goes by with every player passing" in env.rules_text()


def test_a_seat_that_passed_speaks_again_after_somebody_talks():
    env, turn = _tiny(review_rounds=3)
    for move in (RPASS, RSAY, RPASS):                        # the round was not silent
        turn = env.step(turn.actor, turn.task, move)
    assert turn == Turn(1, "review")                        # round 2, seat 1 has the floor again
    assert [e["player_id"] for e in env.review_log()] == [2]


def test_the_round_cap_ends_a_table_that_never_stops_talking():
    env, turn = _tiny(review_rounds=2)
    said = 0
    while isinstance(turn, Turn):
        turn = env.step(turn.actor, turn.task, RSAY)
        said += 1
    assert said == 2 * 3                                    # 2 rounds x 3 seats, then it stops
    assert isinstance(turn, Terminal)


def test_review_off_means_no_review_turn_at_all():
    env, turn = _tiny(review_rounds=0)
    assert isinstance(turn, Terminal) and turn.outcome["winner"] == "werewolf"
    assert env._phase == "over" and env.review_log() == []
    assert env._review_speaker() is None


def test_the_review_rule_sentence_is_rendered_from_the_variant():
    assert "in any case after 2 rounds" in _tiny()[0].rules_text()
    assert "in any case after 5 rounds" in _tiny(5)[0].rules_text()
    off = _tiny(0)[0].rules_text()
    assert "复盘" not in off and "review" not in off.lower()
    assert "复盘" in table(SEATS12).rules_text()             # on by default


def test_a_full_game_still_ends_on_the_same_terminal():
    """The review sits between the win condition and the Terminal, and changes neither."""
    env = table(SEATS12, playbook="static")
    end = play_out(env, first_turn(env))
    assert isinstance(end, Terminal) and end.outcome["winner"] in ("village", "werewolf")
    assert end.outcome == env.outcome()
    assert env.review_log(), "the review ran"


def test_only_the_death_funnel_may_change_who_is_alive():
    """The whole "sum, not product" argument rests on one door.

    Six things can kill a player and three things follow from a death; writing
    the consequences once instead of six times is only sound while every death
    goes through `_kill`. Nothing in the language enforces that, so it is
    enforced here: the source is parsed and every assignment to `self._alive`
    must sit inside `_kill`.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parent.parent / "envs" / "werewolf.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    offenders = []
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for fn in (n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
            for node in ast.walk(fn):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                    targets = [node.target]
                for t in targets:
                    if (isinstance(t, ast.Attribute) and t.attr == "_alive"
                            and isinstance(t.value, ast.Name) and t.value.id == "self"
                            and fn.name not in ("_kill", "reset")):
                        offenders.append(f"{fn.name}:{node.lineno}")
    assert not offenders, f"这些地方绕过了死亡漏斗直接改 _alive：{offenders}"


def test_the_living_set_cannot_be_edited_in_place():
    """`self._alive.discard(x)` must fail loudly rather than skip the funnel."""
    from ..envs import Werewolf
    env = Werewolf("8p")
    env.reset(3)
    assert isinstance(env._alive, frozenset)
    with pytest.raises(AttributeError):
        env._alive.discard(1)                        # the natural typo is now a hard error
    before = env._alive
    env._kill(1, "vote")
    assert 1 in before and 1 not in env._alive       # the funnel rebinds, it does not mutate
