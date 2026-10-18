"""Idempotent steps, the health rollup, and local token accounting."""
from __future__ import annotations

import json

import pytest

from ..arena import Health, StepLedger, Terminal, Turn, replay, run
from ..envs import Werewolf
from ..players import BudgetExhausted, RandomPlayer, TokenBudget, estimate_messages, estimate_tokens
from .util import SEATS12, first_turn, play_until, table


def _players(seed, n=12, clumsy=()):
    return {s: RandomPlayer(seed * 100 + s, clumsy=s in clumsy) for s in range(1, n + 1)}


# ---------------------------------------------------------------- idempotency

def test_the_same_submission_twice_advances_the_world_once():
    env = table(SEATS12)
    ledger = StepLedger(game="g1")
    turn = first_turn(env)
    action = env.action_space(turn.actor, turn.task).fallback

    rec1, next1 = ledger.apply(env, turn, action, seq=0)
    alive_after_first = sorted(env._alive)
    rec2, next2 = ledger.apply(env, turn, action, seq=0)          # the retry

    assert rec2 is rec1 and next2 == next1                        # same outcome handed back
    assert sorted(env._alive) == alive_after_first                # world untouched
    assert ledger.seq == 1 and ledger.replays == 1


def test_a_different_turn_is_not_a_replay():
    env = table(SEATS12)
    ledger = StepLedger(game="g1")
    turn = first_turn(env)
    _, turn2 = ledger.apply(env, turn, env.action_space(turn.actor, turn.task).fallback, seq=0)
    assert isinstance(turn2, Turn) and turn2 != turn
    ledger.apply(env, turn2, env.action_space(turn2.actor, turn2.task).fallback, seq=1)
    assert ledger.seq == 2 and ledger.replays == 0


def test_the_key_is_game_position_actor_and_task():
    ledger = StepLedger(game="g7")
    assert ledger.key(Turn("3", "vote"), seq=4) == "g7:4:3:vote"


def test_a_replayed_journal_is_identical_to_a_deduplicated_one():
    ep = run(Werewolf("12p"), _players(6), 6)
    end = replay(Werewolf("12p"), 6, ep.journal)
    assert isinstance(end, Terminal) and end.outcome == ep.outcome


# ---------------------------------------------------------------- health

def test_every_episode_carries_its_health():
    ep = run(Werewolf("12p"), _players(4), 4)
    assert ep.health.turns == len(ep.journal)
    assert ep.health.forced == 0 and ep.health.usable
    assert ep.health.to_dict()["forced_rate"] == 0.0


def test_a_run_the_harness_had_to_fill_in_is_not_usable():
    class Stubborn:
        def act(self, view, task, spec, feedback):
            return "garbage"
    players = _players(2)
    for s in range(1, 8):
        players[s] = Stubborn()
    ep = run(Werewolf("12p"), players, 2)
    assert ep.health.forced > 0
    assert not ep.health.usable                       # over 5% of turns were the harness, not the model


def test_health_rate_boundary():
    assert Health(turns=100, forced=5).usable
    assert not Health(turns=100, forced=6).usable
    assert Health(turns=0).forced_rate == 0.0


# ---------------------------------------------------------------- token budget

def test_chinese_costs_more_per_character_than_latin():
    cn = estimate_tokens("这是一段中文提示词" * 10)
    en = estimate_tokens("this is a line of english prompt " * 10)
    assert cn > 0 and en > 0
    assert cn / 90 > en / 330                          # per character, CJK is dearer


def test_tool_schemas_are_counted_because_they_are_resent_every_step():
    msgs = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "vote", "description": "x" * 400, "parameters": {}}}]
    assert estimate_messages(msgs, tools) > estimate_messages(msgs) + 50


def test_the_cap_stops_the_run_and_says_what_it_spent():
    b = TokenBudget(max_tokens=1000)
    b.spend(prompt=600, completion=100)
    assert b.total_tokens == 700
    with pytest.raises(BudgetExhausted, match="超过上限"):
        b.spend(prompt=400, completion=0)


def test_a_dollar_cap_uses_the_prices_it_was_given():
    b = TokenBudget(max_tokens=10**9, max_usd=1.0, price_per_1m=(3.0, 15.0))
    b.spend(prompt=200_000, completion=10_000)         # 0.60 + 0.15
    assert round(b.usd, 2) == 0.75
    with pytest.raises(BudgetExhausted, match=r"\$"):
        b.spend(prompt=200_000, completion=0)


def test_gateway_drift_is_recorded_not_trusted():
    b = TokenBudget()
    b.spend(prompt=100, completion=20, gateway_prompt=4000, gateway_completion=20)
    assert b.prompt_tokens == 100                      # ours is what counts
    assert b.drift() == 40.0                           # theirs is what we log
    assert TokenBudget().drift() is None


def test_the_llm_seat_meters_every_call():
    from ..players import GuardStack, LLMPlayer, MockClient
    env = table(SEATS12)
    turn = play_until(env, first_turn(env), lambda e, t: t.task == "speak")
    budget = TokenBudget(max_tokens=10**9)
    p = LLMPlayer(MockClient(3), guard=GuardStack(), token_budget=budget)
    p.act(env.observe(turn.actor), "speak", env.action_space(turn.actor, "speak"), None)
    assert budget.calls >= 1 and budget.prompt_tokens > 0


def test_an_exhausted_budget_stops_the_seat_mid_turn():
    from ..players import GuardStack, LLMPlayer, MockClient
    env = table(SEATS12)
    turn = play_until(env, first_turn(env), lambda e, t: t.task == "speak")
    p = LLMPlayer(MockClient(3), guard=GuardStack(), token_budget=TokenBudget(max_tokens=1))
    with pytest.raises(BudgetExhausted):
        p.act(env.observe(turn.actor), "speak", env.action_space(turn.actor, "speak"), None)


# ---------------------------------------------------------------- watching

def _service_game(**kw):
    from ..server.service import Game
    base = dict(variant="8p", model="script", guard="无", playbook="none", human_seats=[3],
                turn_seconds=None, seed=5, max_tokens=10**9, env_keys={})
    return Game(**{**base, **kw})


def test_a_dead_seat_sees_no_more_than_it_saw_alive():
    """Death is not a promotion: roles stay hidden until the game is over."""
    g = _service_game(human_seats=[])
    g.env.reset(g.seed)
    assert "revealed" not in g.seat_state(1)
    g.state = "over"
    revealed = g.seat_state(1)["revealed"]
    assert set(revealed) == set(range(1, 9)) and "werewolf" in revealed.values()


def test_the_full_log_names_the_model_in_each_seat():
    g = _service_game(model=["script", "mock"], human_seats=[3])
    g.env.reset(g.seed)
    turn = g.env._next()
    g.ledger.apply(g.env, turn, g.env.action_space(turn.actor, turn.task).fallback)
    g.journal.append(__import__("werewolf_harness.arena", fromlist=["Record"]).Record(
        turn.actor, turn.task, {"name": "x"}, False))
    log = g.full_log(seat=3)
    assert log["revealed"] is None                      # still running
    assert "public" in log and "news" in log
    if log["journal"]:
        assert log["journal"][0]["model"] in ("script", "mock", "真人")


def test_a_timed_out_human_turn_is_recorded_as_forced():
    """The health number must not read a clock running out as a decision."""
    from ..players import HumanPlayer
    from .util import SEATS12, first_turn, play_until, table
    env = table(SEATS12)
    turn = play_until(env, first_turn(env), lambda e, t: t.task == "speak")
    p = HumanPlayer(turn.actor, turn_seconds=0.05)
    spec = env.action_space(turn.actor, "speak")
    action = p.act(env.observe(turn.actor), "speak", spec, None)
    assert action == spec.fallback
    assert p.last_forced is True and p.snapshot()["forced"] == 1
    assert p.history[-1]["fallback_used"] == "timeout"


def test_a_seat_that_is_out_may_look_at_every_role():
    """Once you cannot act, hiding the table from you protects nobody."""
    g = _service_game(human_seats=[])
    g.env.reset(g.seed)
    g.env._kill(1, "vote")                           # through the funnel, like every other death
    assert "revealed" not in g.seat_state(2)          # still playing: the table stays closed
    assert set(g.seat_state(1)["revealed"]) == set(range(1, 9))


def test_test_mode_opens_the_table_from_the_first_turn():
    g = _service_game(human_seats=[], xray=True)
    g.env.reset(g.seed)
    assert g.summary()["xray"] is True
    st = g.seat_state(2)                              # alive, mid-game, sees everything
    assert set(st["revealed"]) == set(range(1, 9)) and st["traces"] == {}


def test_the_page_can_always_say_who_is_on_the_clock():
    g = _service_game(human_seats=[])
    assert g.seat_state(1)["acting"] is None
    g.acting = {"actor": 4, "task": "speak", "since": __import__("time").time(),
                "human": False, "model": "script", "steps": ["read_board"]}
    now = g.seat_state(1)["acting"]
    assert now["actor"] == 4 and now["task"] == "speak" and now["steps"] == ["read_board"]
    assert now["seconds"] >= 0


def test_a_step_reaches_the_watcher_without_changing_the_turn():
    """The trace is a read: whether anyone watches must not alter the play."""
    from ..players import GuardStack, LLMPlayer, MockClient
    env = table(SEATS12)
    turn = play_until(env, first_turn(env), lambda e, t: t.task == "speak")
    spec = env.action_space(turn.actor, "speak")
    seen = []
    quiet = LLMPlayer(MockClient(3), guard=GuardStack())
    loud = LLMPlayer(MockClient(3), guard=GuardStack(), on_step=lambda s, st: seen.append((s, st.action)))
    view = env.observe(turn.actor)
    assert loud.act(view, "speak", spec, None) == quiet.act(view, "speak", spec, None)
    assert seen and all(s == turn.actor for s, _ in seen)


def test_a_broken_watcher_cannot_break_a_turn():
    from ..players import GuardStack, LLMPlayer, MockClient
    env = table(SEATS12)
    turn = play_until(env, first_turn(env), lambda e, t: t.task == "speak")

    def explode(seat, step):
        raise RuntimeError("the page fell over")
    p = LLMPlayer(MockClient(3), guard=GuardStack(), on_step=explode)
    assert p.act(env.observe(turn.actor), "speak", env.action_space(turn.actor, "speak"), None)


# ---------------------------------------------------------------- names, not ids

def test_a_choice_a_person_reads_carries_a_name():
    """`hantiao` is what the rules check. It is not what anyone calls the move."""
    from ..players.human import Pending
    from ..players.strategies.playbook import strategy_labels
    env = Werewolf("12p")
    turn = env.reset(8)
    while turn.task != "pack_plan":
        spec = env.action_space(turn.actor, turn.task)
        turn = env.step(turn.actor, turn.task, __import__("werewolf_harness.tests.util",
                        fromlist=["default_action"]).default_action(env.observe(turn.actor), turn.task, spec))
    spec = env.action_space(turn.actor, turn.task)
    tool = Pending(0, turn.task, env.observe(turn.actor), spec, None).to_dict()["tools"][0]
    for key in ("plan", "roles"):
        labels = tool["params"][key]["labels"]
        assert labels and set(labels) >= set(tool["params"][key]["choices"])
        assert all(any("一" <= ch <= "鿿" for ch in v) for v in labels.values())
    assert tool["params"]["roles"]["labels"]["hantiao"].startswith("悍跳")
    assert set(strategy_labels()) == set(tool["params"]["roles"]["choices"])


def test_a_whitelist_on_a_map_is_about_its_values():
    """One tactic per wolf is legal when every tactic is -- not when the map is."""
    from ..arena import Param, ToolSpec
    t = ToolSpec("plan", "", {"roles": Param("dict[int,str]", "", choices=("a", "b"))})
    assert t.check({"roles": {1: "a", 2: "b"}}) is None
    assert "is not one of" in t.check({"roles": {1: "a", 2: "zzz"}})
    assert t.openai_schema()["function"]["parameters"]["properties"]["roles"][
        "additionalProperties"]["enum"] == ["a", "b"]


# ---------------------------------------------------------------- retired models

def test_a_retired_model_is_not_on_the_menu():
    from ..players.providers import PROFILES, offered, profile, retired
    assert "gpt-5.6-terra" not in offered() and "gemini-3.8-flash" not in offered()
    assert "claude-opus-5" in offered() and "claude-sonnet-5" in offered()
    assert retired("gpt-5.6-terra") and retired("claude-opus-5") is None
    # the row stays, so the measurement and the right timeout survive the ban
    assert PROFILES["gpt-5.6-terra"]["timeout_s"] == 180.0
    assert profile("gpt-5.6-sol")["tool_mode"] == "native"


def test_asking_for_a_retired_model_says_so_instead_of_seating_it():
    from ..server.service import GameService
    svc = GameService({})
    with pytest.raises(ValueError, match="已经下架"):
        svc.create(variant="8p", model=["gemini-3.8-flash", "gpt-5.6-sol"], guard="无",
                   playbook="none", human_seats=[1], turn_seconds=None, seed=1, max_tokens=10**9)
    assert not svc.games                                  # nothing half-built was left behind


# ---------------------------------------------------------------- schema dialects

def test_googles_schema_gets_string_enums_and_openai_keeps_its_own():
    """Google refuses `enum` on anything but a string -- and a seat number is an int."""
    from ..players.providers.openai_compat import OpenAICompatClient, _as_string_enums
    tools = [{"type": "function", "function": {"name": "vote", "parameters": {"type": "object",
              "properties": {"target_id": {"type": "integer", "enum": [2, 3]},
                             "note": {"type": "string", "enum": ["a"]},
                             "roles": {"type": "object",
                                       "additionalProperties": {"type": "string", "enum": ["x"]}}}}}}]
    out = _as_string_enums(tools)[0]["function"]["parameters"]["properties"]
    assert out["target_id"] == {"type": "string", "enum": ["2", "3"]}
    assert out["note"] == {"type": "string", "enum": ["a"]}          # already fine, untouched
    assert out["roles"]["additionalProperties"] == {"type": "string", "enum": ["x"]}
    assert tools[0]["function"]["parameters"]["properties"]["target_id"]["enum"] == [2, 3]  # not mutated

    assert OpenAICompatClient("gemini-3.8-flash", api_key="k").schema_dialect == "google"
    assert OpenAICompatClient("gpt-5.6-terra", api_key="k").schema_dialect == "openai"
    assert OpenAICompatClient("claude-opus-5", api_key="k").schema_dialect == "openai"


def test_a_seat_number_that_comes_back_as_text_is_still_a_legal_vote():
    """The wire spells it "5"; the rules still mean seat 5."""
    env = table(SEATS12)
    turn = play_until(env, first_turn(env), lambda e, t: t.task == "vote")
    spec = env.action_space(turn.actor, "vote")
    tool = spec.tool("vote")
    target = tool.params["target_id"].choices[0]
    assert tool.check({"target_id": str(target)}) is None
    assert tool.normalise({"target_id": str(target)})["target_id"] == target


# ---------------------------------------------------------------- chinese providers

def test_a_model_with_no_key_is_not_offered():
    """A seat that cannot be filled is worse than a seat that is not offered."""
    from ..players.providers import offered
    assert "deepseek-v4-pro" not in offered({})
    assert "deepseek-v4-pro" in offered({"LLM_KEY_DEEPSEEK": "sk-x"})
    assert "glm-5.3" not in offered({"LLM_KEY_DEEPSEEK": "sk-x"})


def test_each_vendor_quirk_rides_on_the_profile_not_on_a_branch():
    from ..players.providers import CANDIDATES, PROVIDERS, profile
    glm = profile("glm-5.3")
    assert glm["extra_body"]["thinking"] == {"type": "enabled"}  # disabled is a hard 400 on 5.3
    assert glm["extra_body"]["reasoning_effort"] == "low"        # default max is slow and dear
    # Kimi allows exactly one temperature, and which one depends on thinking.
    assert profile("kimi-k2.6")["extra_body"]["thinking"] == {"type": "disabled"}
    assert profile("kimi-k2.6")["temperature"] == 0.6
    assert profile("deepseek-v4-flash")["extra_body"]["thinking"] == {"type": "disabled"}
    for c in CANDIDATES:
        assert c.provider in PROVIDERS
        assert "timeout_s" in profile(c.model) and "note" in profile(c.model)


def test_no_chinese_row_claims_a_latency_nobody_measured():
    """A number here means somebody called the thing. Documentation says nothing."""
    from ..players.providers import CANDIDATES
    assert CANDIDATES
    for c in CANDIDATES:
        if c.measured:
            assert c.verified, f"{c.model} quotes a number without having been probed"
        if not c.verified:
            assert not c.measured, f"{c.model} is unprobed but claims {c.measured!r}"
    assert any(c.verified for c in CANDIDATES)          # deepseek, probed 2026-09-03


def test_a_chinese_model_goes_direct_and_says_which_key_it_wants():
    from ..server.service import _client
    from ..players.providers.openai_compat import OpenAICompatClient
    cl = _client("deepseek-v4-flash", 1, {"LLM_KEY_DEEPSEEK": "sk-x", "LLM_BASE_URL": "https://relay/v1"})
    assert isinstance(cl, OpenAICompatClient)
    assert cl.base_url == "https://api.deepseek.com"          # its own endpoint, not the relay
    cl2 = _client("deepseek-v4-flash", 1, {"LLM_KEY_DEEPSEEK": "k",
                                           "LLM_KEY_DEEPSEEK_BASE_URL": "https://mirror/v1"})
    assert cl2.base_url == "https://mirror/v1"                # overridable
    try:
        _client("glm-5.3", 1, {})
    except ValueError as e:
        assert "LLM_KEY_ZHIPU" in str(e) and "open.bigmodel.cn" in str(e)
    else:
        raise AssertionError("a missing key must say which key and where to get it")


def test_a_game_in_review_is_neither_running_a_hand_nor_finished():
    """复盘: the result is fixed and the deal is open, but the game has not terminated."""
    g = _service_game(human_seats=[])
    g.env.reset(g.seed)
    turn = g.env._next()
    while isinstance(turn, Turn) and turn.task != "review":
        turn = g.env.step(turn.actor, turn.task, g.env.action_space(turn.actor, turn.task).fallback)
    assert isinstance(turn, Turn) and turn.task == "review"
    s = g.summary()
    assert s["phase"] == "review" and s["reviewing"] is True
    assert s["state"] != "over" and s["outcome"] is None
    # the environment has already opened the deal, so the page may show it
    assert set(g.seat_state(1)["revealed"]) == set(range(1, 9))
    g.env.step(turn.actor, turn.task, {"name": "review_say", "args": {"content": "复盘一下"}})
    assert g.seat_state(1)["review"] == g.env.review_log()
    assert g.full_log()["review"] == g.env.review_log()


def test_the_log_a_living_player_reads_back_holds_no_night_and_no_reasoning():
    """The record is a read like any other, so it obeys the same gate as a view."""
    g = _service_game(human_seats=[3])
    g.env.reset(g.seed)
    from ..arena import Record
    g.journal.append(Record(1, "night", {"name": "night_kill",
                                         "args": {"target_id": 5, "thought": "先切神职"}}, False))
    g.journal.append(Record(2, "speak", {"name": "speak",
                                         "args": {"content": "我是好人", "thought": "先藏一手"}}, False))

    mine = g.full_log(seat=3)                       # alive, mid-game
    tasks = [r["task"] for r in mine["journal"]]
    assert "night" not in tasks                     # the night never happened, as far as seat 3 knows
    said = next(r for r in mine["journal"] if r["task"] == "speak")
    assert said["action"]["args"]["content"] == "我是好人"
    assert "thought" not in said["action"]["args"]   # what he said, never why
    assert mine["traces"] == {} and mine["revealed"] is None

    g.state = "over"                                # the game ends and everything opens
    done = g.full_log(seat=3)
    assert "night" in [r["task"] for r in done["journal"]]
    assert done["journal"][1]["action"]["args"]["thought"] == "先藏一手"
    assert done["revealed"]


def test_a_seat_still_reads_back_its_own_night_move():
    g = _service_game(human_seats=[3])
    g.env.reset(g.seed)
    from ..arena import Record
    g.journal.append(Record(3, "night", {"name": "night_check", "args": {"target_id": 5}}, False))
    assert [r["task"] for r in g.full_log(seat=3)["journal"]] == ["night"]
    assert g.full_log(seat=4)["journal"] == []


# ---------------------------------------------------------------- the batch report

def _fake_game(seed, winner, roles, per):
    return {"seed": seed, "state": "over", "error": None, "winner": winner, "rounds": 4,
            "wall_s": 10.0, "turns": 100, "forced": sum(m["forced"] for m in per.values()),
            "forced_rate": 0.0, "usable": True, "transport_failures": 0, "tokens": 1000,
            "roles": roles, "review_remarks": 3, "pack_chat": 2, "by_model": per}


def test_a_batch_scores_each_side_separately():
    """Winning as a wolf and winning as a villager are not the same skill."""
    from ..experiment import summarise
    roles = {"1": "werewolf", "2": "villager"}
    per_a = {"m": {"seats": [1], "turns": 10, "forced": 0, "terminals": 10, "thoughts": 8,
                   "latency_median_s": 2.0}}
    per_b = {"m": {"seats": [2], "turns": 10, "forced": 1, "terminals": 10, "thoughts": 10,
                   "latency_median_s": 4.0}}
    s = summarise([_fake_game(1, "werewolf", roles, per_a), _fake_game(2, "village", roles, per_b)])
    m = s["by_model"]["m"]
    assert m["wolf_games"] == 1 and m["wolf_wins"] == 1 and m["wolf_winrate"] == 1.0
    assert m["village_games"] == 1 and m["village_wins"] == 1 and m["village_winrate"] == 1.0
    assert m["forced_rate"] == 0.05                      # 1 of 20 turns
    assert m["thought_rate"] == 0.9                      # 18 of 20 terminals
    assert m["latency_median_s"] == 3.0
    assert s["village_wins"] == 1 and s["wolf_wins"] == 1 and s["finished"] == 2


def test_a_batch_says_which_games_never_finished():
    from ..experiment import summarise
    bad = _fake_game(7, None, {}, {})
    bad["state"], bad["error"] = "aborted", "熔断"
    s = summarise([bad])
    assert s["aborted"] == [7] and s["finished"] == 0
    assert s["median_rounds"] is None                    # nothing to average


def test_the_default_budget_covers_the_game_it_is_the_default_for():
    """A cap below what a normal game costs is not a safety net, it is a bug.

    Seed 101 (12p, six models, static playbook) measured 2 367 608 tokens over
    153 turns. A dynamic pack adds a discussion every night on top of that.
    """
    from ..server.http import DEFAULT_MAX_TOKENS
    MEASURED_12P = 2_367_608
    assert DEFAULT_MAX_TOKENS > MEASURED_12P * 2


# ---------------------------------------------------------------- resuming

def test_a_cut_short_game_is_picked_up_where_it_stopped_without_paying_twice():
    """`world == f(seed, journal)` is what makes this possible at all."""
    from ..server.service import GameService
    svc = GameService({})
    g = svc.create(variant="8p", model=["script"], guard="无", playbook="none",
                   human_seats=[], turn_seconds=None, seed=7, max_tokens=10**9)
    g._thread.join()
    whole = [(r.actor, r.task) for r in g.journal]
    assert g.state == "over" and len(whole) > 20

    g.state, g.error = "aborted", "预算用尽"        # cut it short at turn 20
    g.journal = g.journal[:20]
    r = svc.resume(g.id)
    r._thread.join()

    assert r.resumed_from == g.id and r.id != g.id
    assert [(x.actor, x.task) for x in r.journal] == whole      # same game, all of it
    assert r.outcome == g.outcome                               # and the same ending
    assert r.tokens.max_tokens > g.tokens.max_tokens            # with room to finish
    assert svc.games[g.id].state == "aborted"                   # the old row is left alone


def test_resume_refuses_what_it_cannot_honestly_continue():
    from ..server.service import GameService
    svc = GameService({})
    g = svc.create(variant="8p", model=["script"], guard="无", playbook="none",
                   human_seats=[], turn_seconds=None, seed=7, max_tokens=10**9)
    g._thread.join()
    with pytest.raises(ValueError, match="只有中断的局"):
        svc.resume(g.id)                                        # it finished; nothing to resume
    with pytest.raises(ValueError, match="没有这局"):
        svc.resume("nope")


def test_a_journal_that_does_not_fit_is_refused_rather_than_half_applied():
    """A replay that silently diverges would rebuild a different world."""
    from ..arena import Record
    from ..server.service import GameService
    svc = GameService({})
    g = svc.create(variant="8p", model=["script"], guard="无", playbook="none",
                   human_seats=[], turn_seconds=None, seed=7, max_tokens=10**9)
    g._thread.join()
    g.state = "aborted"
    g.journal = [Record(99, "speak", {"name": "speak", "args": {"content": "x"}}, False)]
    with pytest.raises(ValueError, match="日志对不上"):
        svc.resume(g.id)


def test_a_game_resumes_from_a_saved_log_because_a_restart_loses_memory():
    """The fix for what killed the game is a restart, which drops the game."""
    from ..server.service import GameService
    live = GameService({})
    g = live.create(variant="8p", model=["script"], guard="无", playbook="none",
                    human_seats=[], turn_seconds=None, seed=7, max_tokens=10**9)
    g._thread.join()
    whole = [(r.actor, r.task) for r in g.journal]
    g.state = "aborted"
    g.journal = g.journal[:20]
    saved = g.full_log(seat=None)                      # what /log writes to disk

    fresh = GameService({})                            # a brand new process would look like this
    r = fresh.resume_saved(saved)
    r._thread.join()
    assert [(x.actor, x.task) for x in r.journal] == whole
    assert r.outcome == g.outcome and r.resumed_from == g.id

    with pytest.raises(ValueError, match="没有回合"):
        fresh.resume_saved({"game": {"seed": 1}, "journal": []})


def test_a_game_writes_its_journal_as_it_goes_so_a_dead_one_is_resumable(tmp_path, monkeypatch):
    """The ways a game dies are the ways nobody is left to save it."""
    from ..server import service
    monkeypatch.setattr(service, "SAVE_DIR", tmp_path)
    monkeypatch.setattr(service.Game, "SAVE_EVERY", 5)
    svc = service.GameService({})
    g = svc.create(variant="8p", model=["script"], guard="无", playbook="none",
                   human_seats=[], turn_seconds=None, seed=7, max_tokens=10**9)
    g._thread.join()

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1 and g.saved_to == str(files[0])
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert len(saved["journal"]) == len(g.journal)

    # and what was written is enough to carry on from
    r = service.GameService({}).resume_saved({**saved, "journal": saved["journal"][:10]})
    r._thread.join()
    assert r.outcome == g.outcome


def test_a_save_that_fails_does_not_take_the_game_with_it():
    from ..server import service
    svc = service.GameService({})
    g = svc.create(variant="8p", model=["script"], guard="无", playbook="none",
                   human_seats=[], turn_seconds=None, seed=7, max_tokens=10**9)
    g._thread.join()
    g.full_log = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full"))
    g._save()                                        # must not raise
    assert g.state == "over"


# ---------------------------------------------------------------- money, not weather

def test_an_empty_wallet_is_never_retried():
    """401 and 402 mean the same thing everywhere: sending it again cannot help."""
    from ..players.recovery import is_terminal_refusal
    from ..players.providers.base import ProviderError
    for status in (401, 402):
        assert is_terminal_refusal(ProviderError("nope", status=status))
    assert is_terminal_refusal(ProviderError('{"message":"Insufficient Balance"}', status=400))
    assert not is_terminal_refusal(ProviderError("gateway timeout", status=504))
    assert not is_terminal_refusal(ProviderError("connection dropped", status=None))


def test_a_dead_account_says_which_one_and_where_to_pay():
    """The vendor's JSON is not an answer to "what do I do now"."""
    from ..server.service import _in_plain_words
    said = _in_plain_words('HTTP 402 from https://api.deepseek.com/chat/completions: '
                           '{"error":{"message":"Insufficient Balance"}}')
    assert "DeepSeek" in said and "余额" in said and "platform.deepseek.com" in said
    key = _in_plain_words('HTTP 401 from https://api.moonshot.cn/v1/chat/completions: '
                          '{"error":{"message":"Invalid Authentication"}}')
    assert "Kimi" in key and "key" in key and "platform.kimi.com" in key
    plain = _in_plain_words("模型连续答不上来：11 个回合被兜底")
    assert plain == "模型连续答不上来：11 个回合被兜底"      # untouched when it is not about money


def test_a_refusal_that_would_end_the_game_is_confirmed_before_it_is_believed():
    """A gateway reports an empty wallet that is not empty; a game costs too much
    to throw away on one such answer."""
    from ..players.recovery import GatewayExhausted, RecoveryPolicy, RecoveryStats, call_model
    from ..players.providers.base import LLMResponse, ProviderError

    calls = {"n": 0}

    class Flaky:
        def chat(self, *a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ProviderError('{"message":"insufficient balance"}', status=403)
            return LLMResponse(text="ok")

    pol = RecoveryPolicy(confirm_after_s=0)
    pol.sleep = lambda *_: None
    got = call_model(Flaky(), [], None, policy=pol, stats=RecoveryStats(),
                     temperature=0.7, max_tokens=10)
    assert got is not None and calls["n"] == 2          # asked again, and it was fine

    class Dead:
        def chat(self, *a, **kw):
            raise ProviderError('{"message":"Insufficient Balance"}', status=402)

    pol2 = RecoveryPolicy(confirm_after_s=0)
    pol2.sleep = lambda *_: None
    with pytest.raises(GatewayExhausted):
        call_model(Dead(), [], None, policy=pol2, stats=RecoveryStats(),
                   temperature=0.7, max_tokens=10)


def test_403_alone_is_not_proof_of_an_empty_wallet():
    """Relays use it for "not right now" too; only the body settles it."""
    from ..players.recovery import is_terminal_refusal
    from ..players.providers.base import ProviderError
    assert not is_terminal_refusal(ProviderError("forbidden", status=403))
    assert is_terminal_refusal(ProviderError('{"message":"insufficient balance"}', status=403))
    assert is_terminal_refusal(ProviderError("nope", status=402))


def test_the_reason_survives_far_enough_to_be_classified():
    """Kimi answers 429 and says why on the 240th character.

    Truncating before classifying turned "no money" into "refused the request",
    which sends the reader looking for a rate limit that was never there.
    """
    from ..players.recovery import GatewayExhausted, RecoveryPolicy, RecoveryStats, call_model
    from ..players.providers.base import ProviderError
    from ..server.service import _in_plain_words

    body = ('HTTP 429 from https://api.moonshot.cn/v1/chat/completions: '
            '{"error":{"message":"Your account org-1e7e2c58581346b2807f6b1bb6e6739a '
            r'\u003cak-fceo7rnguhyi11d56xji\u003e is suspended due to insufficient balance",'
            '"type":"exceeded_current_quota_error"}}')
    # The contract is not "at character 200" -- it is that cutting the message
    # short throws away the only word that says what went wrong.
    assert "余额" not in _in_plain_words(body[:180])
    assert "余额" in _in_plain_words(body)

    class Broke:
        def chat(self, *a, **kw):
            raise ProviderError(body, status=429)

    pol = RecoveryPolicy(confirm_after_s=0)
    pol.sleep = lambda *_: None
    with pytest.raises(GatewayExhausted) as caught:
        call_model(Broke(), [], None, policy=pol, stats=RecoveryStats(),
                   temperature=0.7, max_tokens=10)
    said = _in_plain_words(str(caught.value))
    assert "Kimi" in said and "余额" in said and "platform.kimi.com/console/pay" in said


def test_a_relay_model_needs_the_relay_key():
    from werewolf_harness.players.providers import offered
    assert "claude-opus-5" not in offered({})
    assert "claude-opus-5" in offered({"LLM_KEY_CLAUDE_GROUP": "sk-x"})
    assert "claude-opus-5" in offered({"LLM_API_KEY": "sk-x"})
