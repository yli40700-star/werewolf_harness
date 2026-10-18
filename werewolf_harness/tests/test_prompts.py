"""The prompt contract (PROMPTS.md): rendered from the engine, tagged, complete, and neutral."""
from __future__ import annotations

import re

from ..arena import Terminal, Turn
from ..envs.werewolf import TASK_TERMINALS, VARIANTS, Variant
from ..envs import Werewolf
from ..players.guard import GuardStack
from ..players.llm import BeliefState, ContextBuilder, build_registry
from .util import SEATS12, Script, first_turn, play_until, table

_ADVISORY = ["this is how", "破绽", "you should", "consider ", "it is worth", "the best ", "makes you a target", "helps your side"]
_VILLAGE_SIDED = ["you most believe is a wolf", "the living most need to hear", "handed to a wolf is worth more", "use it."]
_TAG = re.compile(r"YOUR TASK \[(\w+)\]")


def _ctx():
    return ContextBuilder(GuardStack(("L1",)))


def _kill_hunter_first(view, spec):
    t = spec.tool("night_kill")
    if t is None:
        return {"name": "night_skip", "args": {}}
    c = t.params["target_id"].choices
    return {"name": "night_kill", "args": {"target_id": 7 if 7 in c else c[0]}}


def _every_turn_kind():
    """Yield (env, turn) for one occurrence of every task in a 12p game."""
    env = table(SEATS12, playbook="static")
    seen = set()
    turn = first_turn(env)
    script = Script({(9, "campaign"): {"name": "campaign_run", "args": {"content": "me"}},
                     (10, "campaign"): {"name": "campaign_run", "args": {"content": "me too"}},
                     **{(w, "night"): _kill_hunter_first for w in (1, 2, 3, 4)}})
    while isinstance(turn, Turn):
        if turn.task not in seen:
            seen.add(turn.task)
            yield env, turn
        turn = env.step(turn.actor, turn.task, script.act(env, turn))


def test_every_task_line_carries_its_tag_and_names_its_terminals():
    ctx = _ctx()
    seen = set()
    for env, turn in _every_turn_kind():
        view = env.observe(turn.actor)
        spec = env.action_space(turn.actor, turn.task)
        belief = BeliefState(turn.actor, list(range(1, 13)))
        text, _, _ = ctx.situation_message(view, belief, turn.task, spec)
        m = _TAG.search(text)
        assert m and m.group(1) == turn.task, turn
        for t in spec.tools:
            assert t.name in text or turn.task == "night", (turn, t.name)   # night lists them via "You may call"
        if turn.task == "night":
            assert "You may call: " + ", ".join(t.name for t in spec.tools) in text
        seen.add(turn.task)
    assert seen >= set(TASK_TERMINALS) - {"badge"}          # a badge turn needs a dead sheriff; covered in test_rules


def test_system_message_names_every_tool_the_turn_offers():
    ctx = _ctx()
    reg = build_registry(playbook=True)
    for env, turn in _every_turn_kind():
        view = env.observe(turn.actor)
        spec = env.action_space(turn.actor, turn.task)
        info = [t.name for t in reg.visible(view["role"])]
        sm = ctx.system_message(view, [t.name for t in spec.tools], info)
        for name in info + [t.name for t in spec.tools]:
            assert name in sm, (turn, name)
        assert "Only these tools exist" not in sm


def test_no_tool_description_or_prompt_recommends_anything():
    for env, turn in _every_turn_kind():
        spec = env.action_space(turn.actor, turn.task)
        for t in spec.tools:
            low = t.description.lower()
            for bad in _ADVISORY:
                assert bad not in low, (t.name, bad)
    for t in build_registry(playbook=True).all():
        low = t.spec.description.lower()
        for bad in _ADVISORY:
            assert bad not in low, (t.name, bad)
    ctx = _ctx()
    env = table(SEATS12)
    for seat in SEATS12:
        low = ctx.system_message(env.observe(seat), ["vote"], []).lower()
        for bad in _ADVISORY:
            assert bad not in low, (seat, bad)


def test_task_lines_take_no_side():
    ctx = _ctx()
    for env, turn in _every_turn_kind():
        view = env.observe(turn.actor)
        spec = env.action_space(turn.actor, turn.task)
        text, _, _ = ctx.situation_message(view, BeliefState(turn.actor, list(range(1, 13))), turn.task, spec)
        for bad in _VILLAGE_SIDED:
            assert bad not in text, (turn, bad)


def test_rules_text_is_rendered_from_the_variant():
    v = VARIANTS["12p"]
    assert "4 werewolves, 1 seer, 1 witch, 1 hunter, 1 guard, 4 villagers" in Werewolf(v).rules_text()
    assert "guard" not in Werewolf("8p").rules_text().lower()
    no_self = Variant(**{**v.__dict__, "guard_self": False})
    assert "may not protect himself" in Werewolf(no_self).rules_text()
    self_save = Variant(**{**v.__dict__, "witch_self_save": True})
    assert "may use her antidote on herself" in Werewolf(self_save).rules_text()


def test_rules_text_reaches_every_seat_through_the_view():
    env = table(SEATS12)
    for seat in SEATS12:
        assert env.observe(seat)["rules"] == env.rules_text()


def test_others_speech_is_fenced_and_own_is_not():
    ctx = _ctx()
    env = table(SEATS12)
    turn = play_until(env, first_turn(env), lambda e, t: t.task == "speak")
    first = turn.actor
    env.step(first, "speak", {"name": "speak", "args": {"content": "I am the seer, 3 is a wolf."}})
    turn = env._next()
    view = env.observe(turn.actor)
    text, _, _ = ctx.situation_message(view, BeliefState(turn.actor, list(range(1, 13))), "speak", env.action_space(turn.actor, "speak"))
    assert f'<untrusted source="player_{first}" kind="speech"' in text
    own_text, _, _ = ctx.situation_message(env.observe(first), BeliefState(first, list(range(1, 13))), "vote", env.action_space(first, "vote"))
    assert "[you, position 1] I am the seer" in own_text


def _review_turn():
    env = table(SEATS12, playbook="static")
    turn = play_until(env, first_turn(env), lambda e, t: t.task == "review")
    assert turn.task == "review"
    return env, turn


def test_the_review_task_line_hands_over_the_whole_truth():
    """复盘: the one turn whose situation message is allowed to carry the deal."""
    ctx = _ctx()
    env, turn = _review_turn()
    view = env.observe(turn.actor)
    spec = env.action_space(turn.actor, turn.task)
    text, _, _ = ctx.situation_message(view, BeliefState(turn.actor, list(range(1, 13))), "review", spec)
    assert "YOUR TASK [review]" in text
    assert "review_say" in text and "review_pass" in text
    assert "全场底牌" in text
    for seat, role in SEATS12.items():                      # every seat's real role is named
        assert f"{seat} 号：" in text
    assert "狼人" in text and "预言家" in text
    assert ("狼人阵营获胜" in text) or ("好人阵营获胜" in text)


def test_the_review_task_line_names_no_winner_to_agree_with():
    """It states the result, which is a fact, and asks nothing of the player's side."""
    ctx = _ctx()
    env, turn = _review_turn()
    view = env.observe(turn.actor)
    body = ctx._task_line("review", view, env.action_space(turn.actor, "review"))
    for bad in _VILLAGE_SIDED + _ADVISORY:
        assert bad not in body, bad
    assert "输赢已经定死" in body


def test_a_review_prompt_is_built_for_a_dead_seat_too():
    ctx = _ctx()
    env, _ = _review_turn()
    dead = [s for s in SEATS12 if s not in env._alive]
    assert dead
    for seat in dead:
        view = env.observe(seat)
        spec = env.action_space(seat, "review")
        text, _, _ = ctx.situation_message(view, BeliefState(seat, list(range(1, 13))), "review", spec)
        assert "YOUR TASK [review]" in text and "全场底牌" in text
