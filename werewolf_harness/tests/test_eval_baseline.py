"""The baseline evaluator: CI math, the Chinese claim extractors, and one synthetic game end to end."""
from __future__ import annotations

import json

import pytest

from ..scripts import eval_baseline as eb

ROLES = {1: "seer", 2: "villager", 3: "villager", 4: "witch", 5: "villager", 6: "werewolf", 7: "villager", 8: "werewolf"}


# ------------------------------------------------------------ statistics

def test_wilson_matches_reference_values():
    p, lo, hi = eb.wilson(5, 10)
    assert p == 0.5 and lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    p, lo, hi = eb.wilson(0, 10)
    assert p == 0 and lo == 0 and hi == pytest.approx(0.2775, abs=1e-4)
    assert eb.wilson(0, 0) == (None, None, None)


def test_percentile_is_linear_interpolation():
    assert eb.percentile([1, 2, 3, 4], 50) == 2.5
    assert eb.percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
    assert eb.percentile([7], 95) == 7
    assert eb.percentile([], 50) is None


def test_cluster_bootstrap_resamples_games_not_units():
    # identical games: every resample is the same pool, the interval collapses on the point
    groups = [[1.0, 3.0]] * 6
    assert eb.cluster_bootstrap(groups, eb.mean) == (2.0, 2.0, 2.0)
    # too few games: point only
    assert eb.cluster_bootstrap([[1.0], [2.0]], eb.mean) == (1.5, None, None)
    # one huge game and five small ones: the game-level interval is wide even though units are many
    groups = [[1.0] * 100] + [[0.0]] * 5
    p, lo, hi = eb.cluster_bootstrap(groups, eb.mean, seed=1)
    assert p == pytest.approx(100 / 105) and lo < 0.5 < hi


def test_prop_reports_wilson_and_cluster_intervals():
    d = eb.prop([[1, 0], [1, 1], [0], [1], [1], [0, 0]])
    assert (d["k"], d["n"], d["n_games"]) == (5, 9, 6)
    assert d["wilson"][0] < d["p"] < d["wilson"][1]
    assert d["boot"][0] is not None


# ------------------------------------------------------------ say/do extractor

@pytest.mark.parametrize("text,intent", [
    ("我是8号。我这轮会把票投给2号，除非他给出依据。", 2),
    ("我投3号。", 3),
    ("请大家今天把票投给5号。", 5),
    ("第一天我投了1号，现在看是误投。", None),             # past tense
    ("我投7号，是因为他起跳却拿不出依据。", None),          # explains a ballot already cast
    ("如果他再回避，我就投4号。", None),                    # conditional
    ("2号起跳预言家并验出5号是狼。", None),                 # 出 inside 验出 is not a ballot
    ("我这轮投3号，也可能投4号。", None),                   # two targets: unreadable
])
def test_vote_intent(text, intent):
    assert eb.say_claims(text, speaker=8).intent == intent


def test_negated_intent_is_a_vouch_and_accusations_are_first_person():
    c = eb.say_claims("我不会投1号。我最怀疑3号，6号很可疑。", speaker=2)
    assert c.intent is None and c.vouch == {1} and c.accuse == {3, 6}
    # a seer report is not the speaker's own accusation; a seat in both sets is dropped
    c = eb.say_claims("2号验出5号是狼。我相信4号，但4号很可疑。", speaker=7)
    assert c.accuse == set() and c.vouch == set()
    # "没有理由怀疑" is not an accusation
    assert eb.say_claims("我暂时没有理由怀疑2号。", speaker=4).accuse == set()


def test_say_do_verdict_tiers():
    c = eb.SayClaims(intent=3)
    assert eb.say_do_verdict(c, 3)["any"] is True and eb.say_do_verdict(c, 4)["any"] is False
    assert eb.say_do_verdict(c, None)["intent"] is False                    # abstaining breaks an announced vote
    c = eb.SayClaims(accuse={3, 6}, vouch={1})
    assert eb.say_do_verdict(c, 6)["any"] is True
    assert eb.say_do_verdict(c, 1)["any"] is False
    assert eb.say_do_verdict(eb.SayClaims(), 5)["any"] is None


def test_restrict_drops_claims_about_seats_that_cannot_be_voted():
    c = eb.restrict(eb.SayClaims(intent=2, accuse={2, 3}, vouch={5}), alive={3, 4, 5})
    assert c.intent is None and c.accuse == {3} and c.vouch == {5}


# ------------------------------------------------------------ role / seer extractors

def test_self_role_claims():
    assert eb.self_role_claims("我是3号，村民，没有夜间信息。") == ["villager"]
    assert eb.self_role_claims("我是2号，预言家。") == ["seer"]
    assert eb.self_role_claims("我是真预言家") == ["seer"]
    assert eb.self_role_claims("如果我是预言家，我会先验1号。") == []
    assert eb.self_role_claims("我不是预言家。") == []


def test_check_claims_read_only_first_person_reports():
    assert eb.check_claims("我是预言家，首夜查验了2号，结果是好人。") == [(2, "good")]
    assert eb.check_claims("昨晚我查验了5号，结果是狼人，请大家投5号。") == [(5, "wolf")]
    assert eb.check_claims("我验了4号，他不是狼人。") == [(4, "good")]
    assert eb.check_claims("1号验了2号是好人。") == []
    assert eb.check_claims("他验过2号为好人。") == []


def test_role_assertions_refuse_hedges_and_reports():
    assert [(a.seat, a.role) for a in eb.role_assertions("3号就是狼。", speaker=1)] == [(3, "werewolf")]
    assert eb.role_assertions("我倾向5号是好人。", speaker=1) == []
    assert eb.role_assertions("2号验出5号是狼。", speaker=1) == []
    assert eb.role_assertions("如果3号是预言家，那就麻烦了。", speaker=1) == []
    assert eb.role_assertions("我是3号，3号是村民。", speaker=3) == []          # about themself
    assert eb.role_matches("good", "witch") and not eb.role_matches("good", "werewolf")


def test_guard_audit_measures_the_shipped_regex():
    # the audit reads through the guard's own patterns: a seer report is no longer a ballot
    # announcement (it was, before the 验出/给出 fix), an exile call still is
    assert eb.guard_declared_vote("1号起跳预言家并验出3号是狼。") == (None, None)
    target, trigger = eb.guard_declared_vote("1号验出3号是狼，今天出3号。")
    assert target == 3 and "出3号" in trigger


# ------------------------------------------------------------ synthetic game end to end

def _turn(seat, rnd, task, action, *, steps=None, tokens=(100, 10), lat=1000, note_ms=300, recovery=None, cites=None):
    steps = steps or [{"step": 1, "action": action["name"], "args": action.get("args", {}), "tokens": sum(tokens),
                       "latency_ms": lat, "guard_blocked": False, "block_reason": None}]
    return {"player_id": seat, "round": rnd, "task": task, "action": action, "prompt_tokens": tokens[0],
            "completion_tokens": tokens[1], "latency_ms": sum(s["latency_ms"] for s in steps) + note_ms, "note_ms": note_ms,
            "react_trace": steps, "recovery": recovery or {"retries": 0, "forced_terminal": False, "fallback_used": None},
            "hallucinated_citations": cites or [], "guard_blocks": [], "guard_detections": []}


def _game(tmp_path, name="g1.json", winner="village"):
    journal, turns = [], []

    def add(seat, rnd, task, action, **kw):
        journal.append({"actor": seat, "task": task, "action": action, "forced": False})
        turns.append(_turn(seat, rnd, task, action, **kw))

    add(1, 1, "night", {"name": "night_check", "args": {"target_id": 6}})
    add(6, 1, "night", {"name": "night_kill", "args": {"target_id": 1}})
    add(8, 1, "night", {"name": "night_kill", "args": {"target_id": 1}})
    speeches = {1: "我是1号，预言家。首夜我查验了6号，结果是狼人，请大家今天把票投给6号。",
                2: "我是2号，村民。我相信1号。我投6号。", 3: "我是3号，村民，先听发言。",
                4: "我是4号，村民，没有信息。", 5: "我是5号，村民。我最怀疑8号。",
                6: "我是6号，预言家。我验了2号，结果是狼人。", 7: "我是7号，村民。", 8: "我是8号，村民。我投3号。"}
    for s in range(1, 9):
        add(s, 1, "speak", {"name": "speak", "args": {"content": speeches[s]}}, cites=[{"kind": "no_lookup"}] if s == 7 else None)
    ballots = {1: 6, 2: 6, 3: 6, 4: 6, 5: 6, 6: 2, 7: 6, 8: 2}
    for s in range(1, 9):
        steps = None
        if s == 8:   # blocked once by the guard, then voted
            steps = [{"step": 1, "action": "vote", "args": {"target_id": 2}, "tokens": 50, "latency_ms": 500,
                      "guard_blocked": True, "block_reason": "say_do_consistency"},
                     {"step": 2, "action": "vote", "args": {"target_id": 2}, "tokens": 60, "latency_ms": 700,
                      "guard_blocked": False, "block_reason": None}]
        add(s, 1, "vote", {"name": "vote", "args": {"target_id": ballots[s]}}, steps=steps)
    # timings: strictly sequential, 1s per turn
    timings = [{"seq": i, "actor": j["actor"], "task": j["task"], "asked": float(i), "done": i + 1.0,
                "committed": i + 1.0, "ahead": False} for i, j in enumerate(journal)]
    # turns are stored grouped by seat, as `cli play --out` writes them
    turns.sort(key=lambda t: t["player_id"])
    n_calls = sum(len(t["react_trace"]) + 1 for t in turns)
    log = {"variant": "8p", "seed": 1, "model": "mock", "guard": "L1+L2+L3", "parallel": 8,
           "ground_truth": {"roles": {str(k): v for k, v in ROLES.items()}, "winner": winner},
           "outcome": {"winner": winner}, "duration_s": float(len(journal)), "crashed": None,
           "health": {"usable": True}, "journal": journal, "turns": turns, "timings": timings,
           "night": [{"round": 1, "player_id": 1, "action": "night_check", "target": 6, "outcome": "werewolf"},
                     {"round": 1, "player_id": 6, "action": "night_kill", "target": 1, "outcome": "noted"},
                     {"round": 1, "player_id": 8, "action": "night_kill", "target": 1, "outcome": "noted"}],
           "usage": {"calls": n_calls, "prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55,
                     "gateway_prompt_tokens": sum(t["prompt_tokens"] for t in turns),
                     "gateway_completion_tokens": sum(t["completion_tokens"] for t in turns)}}
    p = tmp_path / name
    p.write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")
    return str(p)


def test_alignment_puts_seat_grouped_turns_back_in_commit_order(tmp_path):
    g = eb.load_game(_game(tmp_path))
    assert g.aligned
    assert [e.task for e in g.entries[:4]] == ["night", "night", "night", "speak"]
    assert g.entries[3].actor == 1 and "预言家" in g.entries[3].content
    assert [t for t, _ in eb.phase_wall_s(g)] == ["night", "speak", "vote"]
    assert [w for _, w in eb.phase_wall_s(g)] == [3.0, 8.0, 8.0]


def test_end_to_end_on_a_synthetic_game(tmp_path):
    m = eb.evaluate(eb.load([_game(tmp_path)]))
    assert m["outcome"]["village_wins"] == 1
    assert m["cost"]["calls_accounted"] and m["cost"]["turn_sum_equals_cli"]
    # say/do: seat 2 said "投6号" and voted 6; seat 8 said "投3号" and voted 2; seat 5 accused 8, voted 6
    sd = m["say_do"]
    assert sd["pairs"] == 8
    assert (sd["consistent"]["intent"]["k"], sd["consistent"]["intent"]["n"]) == (2, 3)   # seats 1 and 2 kept their word, 8 did not
    assert sd["consistent"]["accuse"]["n"] == 1 and sd["consistent"]["accuse"]["k"] == 0  # 5 accused 8, voted 6
    # the seer's public check is honest; the wolf's fake check is a lie about a villager
    s = m["seer"]
    assert s["seer_check_honest"]["k"] == 1 and s["seer_check_honest"]["n"] == 1
    assert s["wolf_fake_seer_games"]["k"] == 1
    assert s["fake_claim_matches_truth"]["k"] == 0 and s["fake_claim_matches_truth"]["n"] == 1
    assert s["self_claim_true"]["werewolf"]["k"] == 0
    # reliability: one blocked step out of 20, one retried turn
    r = m["reliability"]
    assert r["block_reason_per_step"]["say_do_consistency"]["k"] == 1
    assert r["turn_rejected_and_retried"]["k"] == 1
    assert m["hallucination"]["speeches_with_unsupported_citation"]["k"] == 1
    # night-1 kill went to the seer
    assert m["structural_leak"]["night1_kill_hits_seer"]["k"] == 1


def test_cli_writes_report_and_json(tmp_path):
    p1, p2 = _game(tmp_path, "g1.json"), _game(tmp_path, "g2.json", winner="werewolf")
    out, js = tmp_path / "r.md", tmp_path / "m.json"
    eb.main([str(tmp_path / "g*.json"), "--compare", f"seq={p1}", "--out", str(out), "--json", str(js)])
    text = out.read_text(encoding="utf-8")
    assert "核心数字" in text and "好人胜率" in text and "seq" in text
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["base"]["outcome"]["n"] == 2 and data["base"]["outcome"]["village_wins"] == 1
