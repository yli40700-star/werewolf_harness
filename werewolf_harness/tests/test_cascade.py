"""Model cascade: the policy's decisions, escalation inside a turn, logs, recordings, paired stats."""
from __future__ import annotations

import json
import os

import pytest

from ..players import GuardStack, LLMPlayer, RecoveryPolicy
from ..players.llm.cascade import DEFAULT_DECISIVE, FAIL_REASONS, CascadePolicy, cites_nothing
from ..players.providers.base import LLMClient, LLMResponse, ProviderError, ToolCall
from ..players.providers.mock import FlakyMockClient, MockClient, cascade_mock
from ..scripts import paired_cascade as pc
from .util import SEATS12, first_turn, play_until, table


class Tier(LLMClient):
    """Replies in order; a note call (no tools) gets plain text and consumes nothing."""
    tool_mode = "native"

    def __init__(self, name, replies):
        self.name = name
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0):
        if tools is None:
            return LLMResponse(text=f"{self.name} note", prompt_tokens=10, completion_tokens=2)
        self.seen.append([dict(m) for m in messages])
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        if isinstance(r, str):
            return LLMResponse(text=r, prompt_tokens=100, completion_tokens=5)
        name, args = r
        return LLMResponse(tool_calls=[ToolCall(name=name, arguments=args, raw_arguments=json.dumps(args))],
                           prompt_tokens=100, completion_tokens=20)


def _turn(task):
    env = table(SEATS12)
    turn = play_until(env, first_turn(env), lambda e, t: t.task == task)
    return env, turn


def _player(cheap, strong, mode="on-fail", **kw):
    policy = kw.pop("policy", None) or RecoveryPolicy()
    policy.sleep = lambda _s: None
    return LLMPlayer(cheap, strong_client=strong, cascade=CascadePolicy.from_mode(mode, **kw), policy=policy)


def _act(p, env, turn):
    return p.act(env.observe(turn.actor), turn.task, env.action_space(turn.actor, turn.task), None)


def _other(me):
    return next(s for s in range(1, 13) if s != me)


# ------------------------------------------------------------ the policy

def test_modes_and_reasons():
    assert CascadePolicy.from_mode("off") is None
    on_fail = CascadePolicy.from_mode("on-fail")
    decisive = CascadePolicy.from_mode("decisive")
    both = CascadePolicy.from_mode("both", low_confidence=True)
    assert (on_fail.mode, decisive.mode, both.mode) == ("on-fail", "decisive", "both")
    for r in FAIL_REASONS:
        assert on_fail.escalates_on(r) and both.escalates_on(r) and not decisive.escalates_on(r)
    assert on_fail.escalates_on("guard:say_do_consistency") and not on_fail.escalates_on("guard_x")
    assert not on_fail.escalates_on("low_confidence") and both.escalates_on("low_confidence")
    with pytest.raises(ValueError):
        CascadePolicy.from_mode("sometimes")
    with pytest.raises(ValueError):
        CascadePolicy.from_mode("on-fail", fail_reasons=["vibes"])
    narrow = CascadePolicy.from_mode("on-fail", fail_reasons=["schema"])
    assert narrow.escalates_on("schema") and not narrow.escalates_on("guard:evidence")


def test_decisive_turns_by_task_or_by_tool():
    p = CascadePolicy.from_mode("decisive")
    assert set(p.decisive) == set(DEFAULT_DECISIVE)
    assert p.start("vote", ["vote"]) == ("strong", "decisive")
    assert p.start("night", ["night_kill"]) == ("strong", "decisive")
    assert p.start("speak", ["speak"]) == ("cheap", None)
    only_check = CascadePolicy.from_mode("both", decisive=["night_check"])
    assert only_check.start("night", ["night_check", "night_skip"]) == ("strong", "decisive")
    assert only_check.start("night", ["night_kill"]) == ("cheap", None)
    assert CascadePolicy.from_mode("on-fail").start("vote", ["vote"]) == ("cheap", None)


def test_policy_round_trips_through_a_log():
    p = CascadePolicy.from_mode("both", decisive=["vote", "badge"], low_confidence=True)
    d = json.loads(json.dumps(p.to_dict()))
    assert CascadePolicy.from_dict(d) == p and CascadePolicy.from_dict(None) is None


def test_low_confidence_is_a_speech_that_names_no_seat():
    assert cites_nothing("大家好，我听听再说。") and not cites_nothing("我怀疑3号") and not cites_nothing("player 4 is odd")
    p = CascadePolicy.from_mode("on-fail", low_confidence=True)
    assert p.low_confidence_reply("speak", {"content": "我没什么好说的"})
    assert not p.low_confidence_reply("vote", {"target_id": 3})
    assert not CascadePolicy.from_mode("on-fail").low_confidence_reply("speak", {"content": "我没什么好说的"})


def test_tiers_must_speak_the_same_tool_protocol():
    a, b = Tier("a", []), Tier("b", [])
    b.tool_mode = "json_prompt"
    with pytest.raises(ValueError):
        LLMPlayer(a, strong_client=b, cascade=CascadePolicy.from_mode("on-fail"))
    with pytest.raises(ValueError):
        LLMPlayer(a, cascade=CascadePolicy.from_mode("on-fail"))


# ------------------------------------------------------------ escalation inside a turn

def test_a_malformed_cheap_reply_is_retried_on_the_strong_model_with_the_same_transcript():
    env, turn = _turn("speak")
    cheap = Tier("cheap", ["嗯……"])
    strong = Tier("strong", [("speak", {"content": "我怀疑3号"})])
    p = _player(cheap, strong)
    action = _act(p, env, turn)
    assert action["name"] == "speak" and len(cheap.seen) == 1 and len(strong.seen) == 1
    sent, got = cheap.seen[0], strong.seen[0]
    assert got[:len(sent)] == sent                         # the cheap model's whole prompt, unchanged
    assert got[len(sent)]["role"] == "assistant" and "no usable tool call" in got[-1]["content"]
    c = p.last_turn.cascade
    assert c["start"] == "cheap" and c["tier"] == "strong"
    assert c["escalations"] == [{"step": 1, "reason": "schema", "from": "cheap", "to": "strong"}]
    assert [x["model"] for x in c["calls"]] == ["cheap", "strong", "cheap"]          # the note is the cheap model's
    assert [x["kind"] for x in c["calls"]] == ["step", "step", "note"]
    assert p.last_turn.model == "strong"
    assert [(s.model, s.escalated) for s in p.last_turn.steps] == [("cheap", "schema"), ("strong", None)]


@pytest.mark.parametrize("reply,reason", [(("cast_spell", {}), "unknown_tool"), ("illegal", "illegal_args")])
def test_unknown_tool_and_illegal_args_escalate(reply, reason):
    env, turn = _turn("vote")
    me = turn.actor
    if reply == "illegal":
        reply = ("vote", {"target_id": me})                # voting for yourself is not on the list
    cheap = Tier("cheap", [reply])
    strong = Tier("strong", [("vote", {"target_id": _other(me)})])
    p = _player(cheap, strong)
    assert _act(p, env, turn)["args"]["target_id"] == _other(me)
    assert [e["reason"] for e in p.last_turn.cascade["escalations"]] == [reason]
    assert strong.seen[0][:len(cheap.seen[0])] == cheap.seen[0]


def test_a_guard_block_escalates_with_the_guards_name():
    env, turn = _turn("speak")
    me = turn.actor
    env.step(me, "speak", {"name": "speak", "args": {"content": "I will vote for player 9."}})
    turn = play_until(env, env._next(), lambda e, t: t.task == "vote" and t.actor == me)
    cheap = Tier("cheap", [("vote", {"target_id": 10})])
    strong = Tier("strong", [("vote", {"target_id": 9})])
    p = LLMPlayer(cheap, strong_client=strong, cascade=CascadePolicy.from_mode("on-fail"), guard=GuardStack(("L3",)))
    assert _act(p, env, turn)["args"]["target_id"] == 9
    assert [e["reason"] for e in p.last_turn.cascade["escalations"]] == ["guard:say_do_consistency"]
    assert "[guard]" in strong.seen[0][-1]["content"] and strong.seen[0][:len(cheap.seen[0])] == cheap.seen[0]


def test_an_unsupported_citation_is_refused_only_under_a_cascade():
    env, turn = _turn("speak")
    quote = ("speak", {"content": f"{_other(turn.actor)}号说过他是预言家"})
    plain = LLMPlayer(Tier("solo", [quote]))
    _act(plain, env, turn)
    assert plain.last_turn.hallucinated_citations and plain.last_turn.cascade is None
    cheap, strong = Tier("cheap", [quote]), Tier("strong", [("speak", {"content": "我怀疑3号"})])
    p = _player(cheap, strong)
    _act(p, env, turn)
    assert [e["reason"] for e in p.last_turn.cascade["escalations"]] == ["unsupported_citation"]
    assert p.last_turn.steps[0].block_reason == "unsupported_citation" and not p.last_turn.hallucinated_citations


def test_out_of_steps_the_forced_terminal_goes_to_the_strong_model():
    env, turn = _turn("vote")
    me = turn.actor
    looks = [("query_votes", {"round": 1}), ("query_deaths", {})] * 2
    cheap = Tier("cheap", looks)
    strong = Tier("strong", [("vote", {"target_id": _other(me)})])
    p = _player(cheap, strong, policy=RecoveryPolicy(max_react_steps=4))
    assert _act(p, env, turn)["args"]["target_id"] == _other(me)
    assert [e["reason"] for e in p.last_turn.cascade["escalations"]] == ["forced"]
    assert "out of steps" in strong.seen[0][-1]["content"]
    assert p.last_turn.steps[-1].action == "<forced>" and p.last_turn.steps[-1].model == "strong"


def test_a_cheap_model_that_does_not_answer_hands_the_same_request_to_the_strong_one():
    env, turn = _turn("vote")
    err = ProviderError("upstream 503", status=503)
    cheap = Tier("cheap", [err] * 3)
    strong = Tier("strong", [("vote", {"target_id": _other(turn.actor)})])
    p = _player(cheap, strong)
    _act(p, env, turn)
    assert p.last_turn.fallback_used is None
    assert [e["reason"] for e in p.last_turn.cascade["escalations"]] == ["transport"]
    assert strong.seen[0] == cheap.seen[0]


def test_low_confidence_asks_again_on_the_transcript_before_the_reply():
    env, turn = _turn("speak")
    cheap = Tier("cheap", [("speak", {"content": "我没什么想说的，过。"})])
    strong = Tier("strong", [("speak", {"content": "我怀疑3号"})])
    p = _player(cheap, strong, low_confidence=True)
    assert _act(p, env, turn)["args"]["content"] == "我怀疑3号"
    assert strong.seen[0] == cheap.seen[0]                 # the vague reply is not shown to the strong model
    assert [e["reason"] for e in p.last_turn.cascade["escalations"]] == ["low_confidence"]
    assert not p.last_turn.steps[0].guard_blocked


def test_decisive_turns_start_on_the_strong_model_and_escalation_is_sticky():
    env, turn = _turn("vote")
    cheap = Tier("cheap", [])
    strong = Tier("strong", ["?", ("vote", {"target_id": _other(turn.actor)})])
    p = _player(cheap, strong, mode="both")
    _act(p, env, turn)
    assert cheap.seen == [] and len(strong.seen) == 2       # a strong-tier failure stays on the strong tier
    c = p.last_turn.cascade
    assert c["start"] == "strong" and [e["reason"] for e in c["escalations"]] == ["decisive"]
    env2, speak = _turn("speak")
    cheap2 = Tier("cheap", [("speak", {"content": "我怀疑3号"})])
    p2 = _player(cheap2, Tier("strong", []), mode="decisive")
    _act(p2, env2, speak)
    assert p2.last_turn.cascade["tier"] == "cheap" and p2.last_turn.cascade["escalations"] == []


def test_decisive_only_does_not_escalate_on_failure():
    env, turn = _turn("speak")
    cheap = Tier("cheap", ["嗯", ("speak", {"content": "我怀疑3号"})])
    p = _player(cheap, Tier("strong", []), mode="decisive")
    _act(p, env, turn)
    assert p.last_turn.cascade["escalations"] == [] and len(cheap.seen) == 2


def test_without_a_cascade_the_turn_record_is_unchanged():
    env, turn = _turn("speak")
    p = LLMPlayer(Tier("solo", ["嗯", ("speak", {"content": "我怀疑3号"})]))
    _act(p, env, turn)
    d = p.last_turn.to_dict()
    assert "cascade" not in d and all("model" not in s and "escalated" not in s for s in d["react_trace"])


# ------------------------------------------------------------ escalation on the streaming path

def _streaming_player(cheap, strong, events, **kw):
    p = _player(cheap, strong, **kw)
    p.on_draft = lambda k, a, t: events.append((k, a, t))
    return p


def _drafts(events, attempt):
    return "".join(t for k, a, t in events if k == "delta" and a == attempt)


@pytest.mark.parametrize("cheap_says,kw,reason", [
    ("我没什么想说的，过。", {"low_confidence": True}, "low_confidence"),   # asked again, not refused
    ("我觉得大家都挺好。", {"low_confidence": True, "mode": "both"}, "low_confidence"),
])
def test_an_escalated_speech_retracts_the_cheap_draft_and_streams_the_strong_one(cheap_says, kw, reason):
    env, turn = _turn("speak")
    events: list[tuple[str, int, str]] = []
    cheap = Tier("cheap", [("speak", {"content": cheap_says})])
    strong = Tier("strong", [("speak", {"content": "我怀疑3号"})])
    p = _streaming_player(cheap, strong, events, **kw)
    assert _act(p, env, turn)["args"]["content"] == "我怀疑3号"
    assert [e["reason"] for e in p.last_turn.cascade["escalations"]] == [reason]
    kept = p.last_turn.draft_attempt
    attempts = sorted({a for k, a, _ in events if k == "delta"})
    abandoned = [a for k, a, _ in events if k == "abandon"]
    assert attempts == [1, 2] and abandoned == [1] and kept == 2
    assert _drafts(events, 1) == cheap_says and _drafts(events, 2) == "我怀疑3号"
    # the retraction reaches watchers before the strong model's first fragment
    assert events.index(("abandon", 1, next(t for k, a, t in events if k == "abandon"))) < \
        next(i for i, (k, a, _) in enumerate(events) if a == 2)


def test_a_refused_cheap_speech_is_retracted_when_the_strong_retry_starts():
    env, turn = _turn("speak")
    events: list[tuple[str, int, str]] = []
    other = _other(turn.actor)
    cheap = Tier("cheap", [("speak", {"content": f"{other}号说过他是预言家，我觉得他在骗人。"})])
    strong = Tier("strong", [("speak", {"content": "我怀疑3号"})])
    p = _streaming_player(cheap, strong, events)
    assert _act(p, env, turn)["args"]["content"] == "我怀疑3号"
    assert p.last_turn.cascade["tier"] == "strong" and p.last_turn.cascade["escalations"]
    assert [a for k, a, _ in events if k == "abandon"] == [1] and p.last_turn.draft_attempt == 2
    assert _drafts(events, 2) == "我怀疑3号"


def test_a_streaming_flaky_seat_still_makes_mistakes_to_escalate_on():
    """Online seats always stream; the cheap mock must misbehave on that path too."""
    from ..players.providers.base import CancelToken
    escalated = kept = 0
    for seed in range(4):
        env, turn = _turn("speak")
        events: list[tuple[str, int, str]] = []
        p = LLMPlayer(FlakyMockClient(seed=seed, bad_rate=0.9), strong_client=MockClient(seed=seed, name="mock-strong"),
                      cascade=CascadePolicy.from_mode("on-fail"), guard=GuardStack(("L1", "L2", "L3")),
                      on_draft=lambda k, a, t, ev=events: ev.append((k, a, t)))
        p.cancel = CancelToken()
        action = p.act(env.observe(turn.actor), turn.task, env.action_space(turn.actor, turn.task), None)
        escalated += bool(p.last_turn.cascade["escalations"])
        attempts = {a for k, a, _ in events if k == "delta"}
        abandoned = [a for k, a, _ in events if k == "abandon"]
        last = p.last_turn.draft_attempt
        assert len(abandoned) == len(set(abandoned))
        for a in attempts:
            assert (a in abandoned) != (a == last)
        if last is not None:
            kept += 1
            assert _drafts(events, last) == action["args"]["content"]
    assert escalated >= 2 and kept >= 1


# ------------------------------------------------------------ the mock tiers

def test_the_flaky_mock_is_deterministic_and_makes_each_kind_of_mistake():
    env, turn = _turn("speak")
    from ..players.llm.tools import build_registry
    spec = env.action_space(turn.actor, "speak")
    tools = [t.spec.openai_schema() for t in build_registry(playbook=False, num_players=12).visible("seer")] + \
            [t.openai_schema() for t in spec.tools]
    seen = set()
    for i in range(80):
        msgs = [{"role": "system", "content": f"You are player {turn.actor}, role: seer"}, {"role": "user", "content": str(i)}]
        a = FlakyMockClient(seed=1, bad_rate=1.0).chat(msgs, tools)
        b = FlakyMockClient(seed=1, bad_rate=1.0).chat(msgs, tools)
        assert (a.text, [(c.name, c.arguments) for c in a.tool_calls]) == (b.text, [(c.name, c.arguments) for c in b.tool_calls])
        if not a.tool_calls:
            seen.add("malformed")
        elif a.tool_calls[0].name == "reveal_all_roles":
            seen.add("unknown")
        elif 99 in a.tool_calls[0].arguments.values() or "bogus" in a.tool_calls[0].arguments:
            seen.add("illegal")
    assert {"malformed", "unknown", "illegal"} <= seen
    assert cascade_mock("mock-strong", 1).name == "mock-strong" and cascade_mock("mock-flaky:0.4", 1).bad_rate == 0.4
    assert cascade_mock("mock", 1) is None and isinstance(cascade_mock("mock-strong", 1), MockClient)


# ------------------------------------------------------------ cli: log fields and recordings

def _play(tmp_path, name, *extra):
    from .. import cli
    out = str(tmp_path / f"{name}.json")
    rc = cli.main(["play", "--variant", "8p", "--seed", "5", "--model", "mock-flaky:0.4", "--strong", "mock-strong",
                   "--out", out, *extra])
    assert rc == 0
    return out, json.load(open(out))


def test_cli_cascade_game_logs_which_model_answered_and_why(tmp_path):
    _out, log = _play(tmp_path, "g", "--cascade", "both", "--low-confidence")
    assert log["cascade"] == "both" and log["strong"] == "mock-strong"
    assert log["settings"]["cascade"]["low_confidence"] is True
    turns = log["turns"]
    assert all("cascade" in t for t in turns)
    reasons = {e["reason"] for t in turns for e in t["cascade"]["escalations"]}
    assert "decisive" in reasons and reasons - {"decisive"}
    for t in turns:
        assert t["model"] == ("mock-strong" if t["cascade"]["tier"] == "strong" else "mock-flaky:0.4")
        assert all(c["model"] in ("mock-strong", "mock-flaky:0.4") for c in t["cascade"]["calls"])
    s = log["cascade_summary"]
    assert s["turns"] == len(turns) and 0 < s["strong_turns"] < len(turns)
    assert set(s["tokens_by_model"]) == {"mock-strong", "mock-flaky:0.4"}


def test_a_recorded_cascade_game_replays_exactly(tmp_path):
    from ..players.providers.recorded import Cassette
    from ..replay_llm import compare, load, replay_game
    out, log = _play(tmp_path, "r", "--cascade", "both", "--record", "--parallel", "8")
    assert any(e["reason"] != "decisive" for t in log["turns"] for e in t["cascade"]["escalations"])
    cas = Cassette.load(out.replace(".json", ".cassette.jsonl"))
    assert cas.clients["3"]["name"] == "mock-flaky:0.4" and cas.clients["3:strong"]["name"] == "mock-strong"
    models = {cas.request(k)["model"] for k in cas.order}
    assert models == {"mock-flaky:0.4", "mock-strong"}
    log2, cas2 = load(out)
    rep = replay_game(log2, cas2)
    assert compare(log2, rep) == [] and len(rep.used) == len(cas2.calls)
    # the same game played again records the same keys: escalated calls are numbered like any other
    out2, _ = _play(tmp_path, "r2", "--cascade", "both", "--record", "--parallel", "8")
    again = Cassette.load(out2.replace(".json", ".cassette.jsonl"))
    assert sorted(again.calls) == sorted(cas.calls)
    assert all(again.calls[k]["h"] == cas.calls[k]["h"] for k in cas.calls)


def test_replaying_a_call_on_the_wrong_tier_is_a_divergence(tmp_path):
    from ..players.providers.recorded import Cassette, ReplayClient, ReplayDivergence
    from ..players.providers.recorded import call_scope
    out, _log = _play(tmp_path, "w", "--cascade", "on-fail", "--record")
    cas = Cassette.load(out.replace(".json", ".cassette.jsonl"))
    key = next(k for k in cas.order if cas.request(k)["model"] == "mock-strong")
    seat, turn, task, _n, kind = key.split("/")
    req = cas.request(key)
    wrong = ReplayClient(cas, int(seat), counters={(int(seat), int(turn)): int(_n)})      # the cheap tier
    with call_scope(int(seat), int(turn), task, kind), pytest.raises(ReplayDivergence):
        wrong.chat(req["messages"], req["tools"], req["temperature"], req["max_tokens"])
    right = ReplayClient(cas, int(seat), counters={(int(seat), int(turn)): int(_n)}, tier="strong")
    with call_scope(int(seat), int(turn), task, kind):
        right.chat(req["messages"], req["tools"], req["temperature"], req["max_tokens"])


def test_an_old_log_without_cascade_settings_still_replays(tmp_path):
    from .. import cli
    out = str(tmp_path / "plain.json")
    assert cli.main(["play", "--variant", "8p", "--seed", "4", "--model", "mock", "--out", out, "--record"]) == 0
    log = json.load(open(out))
    assert "cascade" not in log["settings"] and "cascade" not in log
    assert cli.main(["replay-llm", out, "--strict"]) == 0


# ------------------------------------------------------------ the online table

def test_the_table_builds_cascade_seats_and_counts_escalations():
    from ..app.runtime.table import Table, TableConfig
    cfg = TableConfig(variant="8p", models=["mock"], cascade={"cheap": "mock-flaky", "strong": "mock-strong", "mode": "both"})
    t = Table("g1", 3, cfg)
    assert set(t.seat_models.values()) == {"cascade"} and t.models == ["mock-flaky", "mock-strong"]
    p = t.make_player(1, task="vote")
    assert p.client.name == "mock-flaky" and p.strong_client.name == "mock-strong" and p.cascade.mode == "both"
    t.note_cascade({"cascade": {"tier": "strong", "calls": [{"tier": "strong"}, {"tier": "cheap"}],
                                "escalations": [{"reason": "decisive"}]}})
    t.note_cascade({})
    s = t.summary()["cascade"]
    assert s["turns"] == 1 and s["strong_turns"] == 1 and s["escalations"] == {"decisive": 1} and s["mode"] == "both"
    assert "cascade" not in Table("g2", 3, TableConfig(variant="8p")).summary()
    assert TableConfig.from_dict(cfg.to_dict()).cascade == cfg.cascade


def test_an_online_cascade_turn_reports_its_record_in_the_result_meta():
    from ..app.runtime.agents import AgentJob, run_agent_turn
    from ..app.runtime.table import Table, TableConfig
    from ..players.providers.base import CancelToken
    t = Table("g3", 3, TableConfig(variant="8p", cascade={"cheap": "mock-flaky:0.9", "strong": "mock-strong",
                                                          "mode": "on-fail"}))
    t.reset()
    turn = t.turn
    view, spec = t.env.observe(turn.actor), t.env.action_space(turn.actor, turn.task)
    job = AgentJob("g3", turn.actor, 0, turn.task, view, spec, None, make_player=lambda on_step: t.make_player(
        turn.actor, on_step=on_step, task=turn.task, version=0), on_step=lambda *a: None)
    job.cancel = CancelToken()
    res = run_agent_turn(job)
    assert res.meta["cascade"]["calls"] and res.meta["model"] in ("mock-flaky:0.9", "mock-strong")


# ------------------------------------------------------------ paired statistics

def test_paired_difference_and_its_bootstrap():
    a = {s: 0.9 + 0.01 * (s % 3) for s in range(10)}
    b = {s: 0.8 for s in range(10)}
    d = pc.paired_diff(a, b)
    assert d["n"] == 10 and d["diff"] == pytest.approx(sum(a.values()) / 10 - 0.8)
    assert d["lo"] <= d["diff"] <= d["hi"] and d["lo"] >= 0.1 - 1e-9 and d["hi"] <= 0.12 + 1e-9
    # unpaired seeds and missing values are dropped, not imputed
    d2 = pc.paired_diff({1: 1.0, 2: None, 3: 0.5}, {1: 0.5, 2: 0.1, 4: 0.0})
    assert d2["n"] == 1 and d2["diff"] == 0.5 and d2["lo"] is None
    # identical arms: the CI collapses on zero
    z = pc.paired_diff(b, dict(b))
    assert z["diff"] == 0 and z["lo"] == 0 and z["hi"] == 0


def test_pairing_removes_the_between_seed_variance():
    # big deal-to-deal swings, a constant +0.05 effect: the paired CI is tight, an unpaired one would not be
    base = {s: (s * 37 % 11) / 11 for s in range(12)}
    arm = {s: v + 0.05 for s, v in base.items()}
    d = pc.paired_diff(arm, base)
    assert d["lo"] == pytest.approx(0.05) and d["hi"] == pytest.approx(0.05)


def test_ratio_and_non_inferiority():
    r = pc.paired_ratio({s: 2.0 for s in range(6)}, {s: 4.0 for s in range(6)})
    assert r["ratio"] == 0.5 and r["lo"] == 0.5 and r["hi"] == 0.5
    assert pc.non_inferiority({"lo": -0.01, "hi": 0.02}, 0.02)["verdict"] == "non_inferior"
    assert pc.non_inferiority({"lo": -0.03, "hi": 0.02}, 0.02)["verdict"] == "not_shown"
    assert pc.non_inferiority({"lo": None, "hi": None}, 0.02)["verdict"] == "insufficient"
    assert pc.non_inferiority({"lo": -0.5, "hi": 0.01}, 0.02, higher_is_better=False)["verdict"] == "non_inferior"


def test_cost_is_split_by_model_and_priced():
    log = {"model": "cheap", "cascade": "both", "usage": {"gateway_prompt_tokens": 1300, "gateway_completion_tokens": 120},
           "turns": [{"cascade": {"calls": [{"model": "cheap", "prompt": 1000, "completion": 100},
                                            {"model": "strong", "prompt": 200, "completion": 10}]}}]}
    tok = pc.tokens_by_model(log)
    assert tok == {"cheap": [1100, 110], "strong": [200, 10]}           # the unattributed rest goes to the cheap model
    prices = {"cheap": (1.0, 5.0), "strong": (3.0, 15.0)}
    assert pc.usd(tok, prices) == pytest.approx((1100 * 1 + 110 * 5 + 200 * 3 + 10 * 15) / 1e6)
    plain = {"model": "strong", "usage": {"gateway_prompt_tokens": 50, "gateway_completion_tokens": 5}}
    assert pc.tokens_by_model(plain) == {"strong": [50, 5]}
    assert pc.price_of("mock-flaky:0.3", {}) == pc.FLAKY_PRICE
    with pytest.raises(KeyError):
        pc.price_of("who-knows", {})


def test_seed_specs_and_arms():
    assert pc.parse_seeds("2001-2003,2010,2002") == [2001, 2002, 2003, 2010]
    assert pc.arm_spec("cascade", "c", "s", "both") == {"arm": "cascade", "model": "c", "strong": "s", "cascade": "both"}
    assert pc.arm_spec("cascade-decisive", "c", "s", "both")["cascade"] == "decisive"
    assert pc.arm_spec("strong", "c", "s", "both") == {"arm": "strong", "model": "s"}
    with pytest.raises(SystemExit):
        pc.arm_spec("medium", "c", "s", "both")


def test_the_runner_plays_pairs_resumes_and_reports(tmp_path, capsys):
    out = str(tmp_path / "pc")
    argv = ["--seeds", "301-305", "--cheap", "mock-flaky:0.4", "--strong", "mock-strong",
            "--arms", "strong,cheap,cascade", "--out", out, "--record"]
    res = pc.main(argv)
    for arm in ("strong", "cheap", "cascade"):
        assert sorted(os.listdir(os.path.join(out, arm))) == sorted(
            [f"g{s}.json" for s in range(301, 306)] + [f"g{s}.cassette.jsonl" for s in range(301, 306)])
    assert res["arms"]["cascade"]["escalated"]["p"] > 0 and res["arms"]["strong"]["escalated"]["p"] == 0
    assert res["paired"]["cascade"]["usd"]["n"] == 5
    assert res["paired"]["cascade"]["non_inferiority_legal"]["verdict"] in ("non_inferior", "not_shown")
    assert os.path.exists(os.path.join(out, "report.md")) and json.load(open(os.path.join(out, "report.json")))["paired"]
    # resume: nothing is played again; a log cut short is
    mt = {p: os.path.getmtime(p) for p in (os.path.join(out, "strong", "g301.json"), os.path.join(out, "cheap", "g302.json"))}
    open(os.path.join(out, "cheap", "g302.json"), "w").write('{"journal": [')
    capsys.readouterr()
    pc.main(argv)
    printed = capsys.readouterr().out
    assert "已完成 14 局" in printed
    assert os.path.getmtime(os.path.join(out, "strong", "g301.json")) == mt[os.path.join(out, "strong", "g301.json")]
    assert pc.done(os.path.join(out, "cheap", "g302.json"))


def test_the_plan_is_printed_with_a_time_and_token_estimate(tmp_path, capsys):
    pc.main(["--seeds", "1-10", "--cheap", "claude-cli:haiku", "--strong", "claude-cli:sonnet",
             "--out", str(tmp_path), "--dry-run"])
    text = capsys.readouterr().out
    assert "10 个种子 × 3 臂 = 30 局" in text
    assert "每局约 3.4 分钟" in text and "每局约 6.8 分钟" in text
    # 10 haiku (34 min) + 10 sonnet (68 min) + 10 cascade at the midpoint (51 min) = 153 min
    assert "合计约 2.5 小时" in text and "16.8M CLI token" in text
