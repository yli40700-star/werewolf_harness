"""The LLM seat: three gates, guard blocks, retries, fallbacks, gateway exhaustion."""
from __future__ import annotations

import json

import pytest

from ..arena import Terminal, replay, run
from ..envs import Werewolf
from ..players import GatewayExhausted, GuardStack, LLMPlayer, MockClient, RecoveryPolicy, TransportBudget
from ..players.providers.base import LLMClient, LLMResponse, ProviderError, ToolCall
from .util import SEATS12, first_turn, play_until, table


class Scripted(LLMClient):
    """Returns the given replies in order; each reply is a (name, args) tool call, a text, or an exception."""
    name = "scripted"
    tool_mode = "native"

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0):
        self.seen.append(list(messages))
        r = self.replies.pop(0) if self.replies else ("vote", {"target_id": 2})
        if isinstance(r, Exception):
            raise r
        if isinstance(r, str):
            return LLMResponse(text=r)
        name, args = r
        return LLMResponse(tool_calls=[ToolCall(name=name, arguments=args, raw_arguments=json.dumps(args))])


def _speak_turn():
    env = table(SEATS12)
    turn = play_until(env, first_turn(env), lambda e, t: t.task == "speak")
    return env, turn


def _vote_turn():
    env = table(SEATS12)
    turn = play_until(env, first_turn(env), lambda e, t: t.task == "vote")
    return env, turn


def test_full_game_with_mock_seats_and_all_guards():
    budget = TransportBudget()
    players = {s: LLMPlayer(MockClient(700 + s), guard=GuardStack(("L1", "L2", "L3")), transport_budget=budget) for s in range(1, 13)}
    ep = run(Werewolf("12p"), players, 7)
    assert ep.outcome["winner"] in ("village", "werewolf")
    assert not any(r.forced for r in ep.journal)
    assert not any(t.fallback_used for p in players.values() for t in p.history)
    end = replay(Werewolf("12p"), 7, ep.journal)
    assert isinstance(end, Terminal) and end.outcome == ep.outcome


def test_information_tool_then_terminal():
    env, turn = _speak_turn()
    client = Scripted([("query_deaths", {}), ("speak", {"content": "nobody died yet"})])
    p = LLMPlayer(client)
    action = p.act(env.observe(turn.actor), "speak", env.action_space(turn.actor, "speak"), None)
    # `thought` is an optional param on every terminal, so normalise fills it in
    assert action == {"name": "speak", "args": {"content": "nobody died yet", "thought": None}}
    steps = p.last_turn.steps
    assert [s.action for s in steps] == ["query_deaths", "speak"]
    assert "nobody has died yet" in steps[0].observation


def test_unknown_tool_is_refused_with_the_whitelist_attached():
    env, turn = _speak_turn()
    client = Scripted([("cast_spell", {}), ("speak", {"content": "ok"})])
    p = LLMPlayer(client)
    p.act(env.observe(turn.actor), "speak", env.action_space(turn.actor, "speak"), None)
    refusal = client.seen[1][-1]["content"]
    assert "unknown tool 'cast_spell'" in refusal and "speak" in refusal
    assert p.last_turn.recovery["retries"] == 1


def test_illegal_target_is_refused_by_the_environments_own_list():
    env, turn = _vote_turn()
    me = turn.actor
    client = Scripted([("vote", {"target_id": me}), ("vote", {"target_id": [s for s in range(1, 13) if s != me][0]})])
    p = LLMPlayer(client)
    action = p.act(env.observe(me), "vote", env.action_space(me, "vote"), None)
    assert action["args"]["target_id"] != me
    assert "is not one of" in client.seen[1][-1]["content"]
    assert p.last_turn.steps[0].block_reason == "semantics"


def test_l3_blocks_a_ballot_that_contradicts_the_seats_own_speech():
    env, turn = _speak_turn()
    me = turn.actor
    env.step(me, "speak", {"name": "speak", "args": {"content": "I will vote for player 9."}})
    turn = play_until(env, env._next(), lambda e, t: t.task == "vote" and t.actor == me)
    client = Scripted([("vote", {"target_id": 10}), ("vote", {"target_id": 9})])
    p = LLMPlayer(client, guard=GuardStack(("L3",)))
    action = p.act(env.observe(me), "vote", env.action_space(me, "vote"), None)
    assert action["args"]["target_id"] == 9
    assert p.last_turn.guard_blocks[0]["check"] == "say_do_consistency"


def test_malformed_replies_are_bounded_then_fall_back_to_the_environment_default():
    env, turn = _vote_turn()
    client = Scripted(["I'm thinking...", "still thinking", "hmm", "no idea", "..."])
    p = LLMPlayer(client, policy=RecoveryPolicy(max_retries=2))
    spec = env.action_space(turn.actor, "vote")
    action = p.act(env.observe(turn.actor), "vote", spec, None)
    assert action == spec.fallback
    assert p.last_turn.fallback_used == "default_action"
    assert p.last_turn.recovery["forced_terminal"]


def test_transport_failure_spends_the_shared_budget():
    env, turn = _vote_turn()
    budget = TransportBudget(limit=1)
    err = ProviderError("gateway timed out after 30s", status=None)
    client = Scripted([err, err, err, err, err, err])
    policy = RecoveryPolicy(max_transport_retries=0)
    policy.sleep = lambda s: None
    p = LLMPlayer(client, policy=policy, transport_budget=budget)
    spec = env.action_space(turn.actor, "vote")
    action = p.act(env.observe(turn.actor), "vote", spec, None)
    assert action == spec.fallback and p.last_turn.fallback_used == "transport_failure"
    assert budget.failures == 1
    with pytest.raises(GatewayExhausted):
        p.act(env.observe(turn.actor), "vote", spec, None)


def test_terminal_refusal_is_not_retried():
    env, turn = _vote_turn()
    # Twice, because a refusal that ends a game is confirmed before it is
    # believed -- see confirm_terminal in RecoveryPolicy.
    client = Scripted([ProviderError("403: Insufficient account balance", status=403),
                       ProviderError("403: Insufficient account balance", status=403)])
    policy = RecoveryPolicy(confirm_after_s=0)
    policy.sleep = lambda s: None
    p = LLMPlayer(client, policy=policy)
    with pytest.raises(GatewayExhausted):
        p.act(env.observe(turn.actor), "vote", env.action_space(turn.actor, "vote"), None)
    # Asked exactly twice: once, then once more to confirm before ending the game.
    assert len(client.seen) == 2


def test_driver_feedback_is_shown_to_the_model():
    env, turn = _vote_turn()
    client = Scripted([("vote", {"target_id": 2 if turn.actor != 2 else 3})])
    p = LLMPlayer(client)
    p.act(env.observe(turn.actor), "vote", env.action_space(turn.actor, "vote"), "'99' is not one of [...]")
    assert "[guard] Your previous answer was refused" in client.seen[0][-1]["content"]


def test_beliefs_persist_across_turns_inside_the_player():
    env, turn = _speak_turn()
    me = turn.actor
    other = 2 if me != 2 else 3
    client = Scripted([("update_belief", {"player_id": other, "suspicion": 0.9, "reason": "loud"}), ("speak", {"content": "hm"}),
                       "note: nothing yet", ("read_belief", {"player_id": other}), ("vote", {"target_id": other})])
    p = LLMPlayer(client)
    p.act(env.observe(me), "speak", env.action_space(me, "speak"), None)
    assert p.belief.get(other).suspicion == 0.9
    env.step(me, "speak", {"name": "speak", "args": {"content": "hm"}})
    turn = play_until(env, env._next(), lambda e, t: t.task == "vote" and t.actor == me)
    p.act(env.observe(me), "vote", env.action_space(me, "vote"), None)
    assert "suspicion=0.90" in p.last_turn.steps[0].observation


def test_the_note_to_the_next_turn_is_written_read_back_and_never_reaches_the_table():
    env, turn = _speak_turn()
    me = turn.actor
    other = 2 if me != 2 else 3
    long_note = "watch player %d: " % other + "x" * 500
    client = Scripted([("speak", {"content": "hm"}), long_note, ("vote", {"target_id": other}), "next note"])
    p = LLMPlayer(client)
    p.act(env.observe(me), "speak", env.action_space(me, "speak"), None)
    # written at the end of the turn: one extra text-only call on the turn's own transcript
    assert p.note == long_note[:400] and len(p.note) == 400
    assert client.seen[1][-1]["content"].startswith("[harness] Before this turn ends")
    assert "tools" not in client.seen[1][-1]
    assert p.last_turn.note_before == "" and p.last_turn.note_after == p.note
    assert p.last_turn.to_dict()["note_after"] == p.note
    # nothing of it is in the environment: no seat's view carries it
    env.step(me, "speak", {"name": "speak", "args": {"content": "hm"}})
    for seat in range(1, 13):
        assert "watch player" not in json.dumps(env.observe(seat), ensure_ascii=False, default=str)
    # read back at the start of the next turn, under the beliefs
    turn = play_until(env, env._next(), lambda e, t: t.task == "vote" and t.actor == me)
    p.act(env.observe(me), "vote", env.action_space(me, "vote"), None)
    situation = client.seen[2][1]["content"]
    assert "--- your note from last turn ---\n" + long_note[:400] in situation
    assert situation.index("--- your recorded beliefs ---") < situation.index("--- your note from last turn ---")
    assert p.last_turn.note_before == long_note[:400] and p.note == "next note"


def test_a_failed_or_empty_note_call_keeps_the_previous_note():
    env, turn = _vote_turn()
    me = turn.actor
    other = 2 if me != 2 else 3
    policy = RecoveryPolicy(max_transport_retries=0)
    policy.sleep = lambda s: None
    client = Scripted([("vote", {"target_id": other}), ProviderError("gateway timed out after 30s", status=None)])
    p = LLMPlayer(client, policy=policy)
    p.note = "kept from before"
    spec = env.action_space(me, "vote")
    action = p.act(env.observe(me), "vote", spec, None)
    assert action["name"] == "vote" and not p.last_turn.fallback_used     # the note call failing is not the turn failing
    assert p.note == "kept from before"
    # an empty text reply (here: a stray tool call with no text) also leaves it alone
    client.replies = [("vote", {"target_id": other}), ("read_belief", {"player_id": other})]
    p.act(env.observe(me), "vote", spec, None)
    assert p.note == "kept from before"
    # a fresh seat with no note yet is told so
    fresh = Scripted([("vote", {"target_id": other}), "n"])
    q = LLMPlayer(fresh)
    q.act(env.observe(me), "vote", spec, None)
    assert "--- your note from last turn ---\n(none)" in fresh.seen[0][1]["content"]


def test_no_note_is_asked_for_after_a_transport_failure():
    env, turn = _vote_turn()
    err = ProviderError("gateway timed out after 30s", status=None)
    client = Scripted([err, "should never be asked"])
    policy = RecoveryPolicy(max_transport_retries=0)
    policy.sleep = lambda s: None
    p = LLMPlayer(client, policy=policy)
    p.act(env.observe(turn.actor), "vote", env.action_space(turn.actor, "vote"), None)
    assert p.last_turn.fallback_used == "transport_failure"
    assert len(client.seen) == 1 and p.note == ""


def test_wolf_playbook_tools_are_invisible_to_villagers():
    from ..players.llm import build_registry
    reg = build_registry(playbook=True)
    assert "read_board" in [t.name for t in reg.visible("werewolf")]
    assert "read_board" not in [t.name for t in reg.visible("villager")]
    assert "read_board" not in [t.name for t in build_registry(playbook=False).all()]


def test_empty_gateway_completion_is_a_retryable_transport_failure(monkeypatch):
    from ..players.providers.openai_compat import OpenAICompatClient
    cl = OpenAICompatClient("m", api_key="k", base_url="http://x/v1")
    empty = {"choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "tool_calls"}],
             "usage": {"prompt_tokens": 289, "completion_tokens": 1}}
    monkeypatch.setattr(cl, "_post", lambda path, payload, timeout: empty)
    with pytest.raises(ProviderError) as exc:
        cl.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}])
    assert exc.value.status == 502 and "empty completion" in str(exc.value)


def test_a_call_that_never_returns_hits_the_hard_deadline():
    import time
    from ..players.recovery import call_model, RecoveryStats

    class Hangs(LLMClient):
        name = "hangs"
        def chat(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0):
            time.sleep(30)
            return LLMResponse(text="late")
    policy = RecoveryPolicy(timeout_s=0.1, max_transport_retries=0)
    stats = RecoveryStats()
    t0 = time.time()
    assert call_model(Hangs(), [], [], policy=policy, stats=stats, temperature=0, max_tokens=1) is None
    assert time.time() - t0 < 10 and stats.timeouts      # deadline = 1.5 * timeout_s + 5s


def test_every_model_profile_is_measured_and_used():
    """The profiles table is how a measured fact about the relay stays fixed.

    It exists because two wrong conclusions were drawn from defaults: a 75s
    timeout read claude-opus-5 as a dead channel, and native tool calling read
    claude-sonnet-5 as a broken model. Both were the relay.
    """
    from ..players.providers import PROFILES, profile
    for name, p in PROFILES.items():
        assert p["tool_mode"] in ("native", "json_prompt"), name
        assert 30.0 <= p["timeout_s"] <= 600.0, name
        assert p["note"], name
    assert all(PROFILES[m]["tool_mode"] == "json_prompt" for m in PROFILES if m.startswith("claude"))
    assert profile("something-unlisted")["tool_mode"] == "native"


def test_the_service_builds_seats_from_the_profile():
    from ..server.service import Game
    g = Game(variant="8p", model="claude-opus-5", guard="L1", playbook="none", human_seats=[3],
             turn_seconds=None, seed=1, max_tokens=1000, env_keys={"LLM_KEY_CLAUDE_GROUP": "k", "LLM_BASE_URL": "http://x/v1"})
    seat = next(p for s, p in g.players.items() if s != 3)
    assert seat.client.tool_mode == "json_prompt"
    assert seat.policy.timeout_s == 120.0
    assert g.players[3].is_human


def test_the_thought_shown_is_the_thought_not_its_envelope():
    """A reader should see the reasoning, not the JSON block it arrived in."""
    from ..players.llm.loop import _thought_of

    class R:
        def __init__(self, text): self.text = text

    fenced = '```json\n{"thought": "先摸清局势", "action": "read_board", "args": {}}\n```'
    assert _thought_of(R(fenced)) == "先摸清局势"
    assert _thought_of(R('{"thought": "裸的也行"}')) == "裸的也行"
    assert _thought_of(R("就是一句普通的话")) == "就是一句普通的话"
    assert _thought_of(R("")) == ""
    assert _thought_of(R("```json\n{坏掉的 json\n```")).startswith("{坏掉")


def test_reasoning_from_the_json_envelope_lands_on_the_action():
    """A claude seat is driven by JSON, where the thought sits above `args`.

    Without moving it in, the record of a JSON-protocol seat has no reasoning at
    all while a native-tool-calling seat has one on every move.
    """
    from ..envs import Werewolf
    from ..players import GuardStack, LLMPlayer
    from ..players.providers.base import LLMResponse, ToolCall

    env = Werewolf("12p")
    turn = env.reset(3)
    while turn.task != "speak":
        spec = env.action_space(turn.actor, turn.task)
        from .util import default_action
        turn = env.step(turn.actor, turn.task, default_action(env.observe(turn.actor), turn.task, spec))
    spec = env.action_space(turn.actor, "speak")

    class Envelope:
        name = "json-ish"

        def chat(self, messages, tools=None, **kw):
            return LLMResponse(
                text='{"thought": "先立住身份", "action": "speak", "args": {"content": "我是预言家"}}',
                tool_calls=[ToolCall("speak", {"content": "我是预言家"})])

    p = LLMPlayer(Envelope(), guard=GuardStack())
    action = p.act(env.observe(turn.actor), "speak", spec, None)
    assert action["args"]["content"] == "我是预言家"
    assert action["args"]["thought"] == "先立住身份"     # carried down from the envelope
    assert p.last_turn.steps[-1].thought == "先立住身份"


def test_an_info_tool_is_not_given_an_argument_it_does_not_take():
    """Only a spec that declares `thought` gets one; info tools would go illegal."""
    from ..players.llm import build_registry
    for t in build_registry(True).visible("villager"):
        assert "thought" not in t.spec.params


class Mumbler(LLMClient):
    """Never produces a usable answer: every turn ends on the seat's own fallback."""
    name = "mumbler"
    tool_mode = "native"

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0):
        return LLMResponse(text="hmm...")


@pytest.mark.parametrize("parallel", [1, 4])
def test_a_seat_reported_fallback_is_journalled_as_forced(parallel):
    """The seat returns a *legal* action (its fallback), so the driver's own check passes;
    the journal and the health line must still count it as forced, on both paths."""
    from ..players import RandomPlayer
    players = {s: RandomPlayer(900 + s) for s in range(1, 9)}
    players[3] = LLMPlayer(Mumbler(), policy=RecoveryPolicy(max_retries=1), async_note=parallel > 1)
    ep = run(Werewolf("8p"), players, 9, parallel=parallel)
    players[3].flush()
    fell_back = sum(1 for t in players[3].history if t.fallback_used)
    seat3 = [r for r in ep.journal if r.actor == 3]
    assert fell_back > 0 and fell_back == len(seat3) == sum(r.forced for r in seat3)
    assert not any(r.forced for r in ep.journal if r.actor != 3)
    assert ep.health.forced == fell_back
