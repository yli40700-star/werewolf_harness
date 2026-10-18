"""Streaming speech: the incremental `content` extractor and the speech stream."""
from __future__ import annotations

import json
import random

import pytest

from werewolf_harness.players.llm.streaming import ContentExtractor, SpeechStream
from werewolf_harness.players.providers.base import Delta

SPEECH = '我是 3 号。\n他说"我是预言家"——我不信 😀 \\ 路径 C:\\tmp\t完'
THOUGHT = '其实我是狼，"别说出去" \\n 秘密'


def _chunks(s: str, n: int):
    return [s[i:i + n] for i in range(0, len(s), n)]


def _run(raw: str, n: int, paths=(("content",),)) -> tuple[str, ContentExtractor]:
    e = ContentExtractor(paths)
    return "".join(e.feed(c) for c in _chunks(raw, n)), e


@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 64, 10_000])
def test_every_split_boundary_decodes_the_same(n):
    raw = json.dumps({"thought": THOUGHT, "content": SPEECH, "n": 3}, ensure_ascii=True)   # \\uXXXX + surrogates
    out, e = _run(raw, n)
    assert out == SPEECH and e.done
    raw = json.dumps({"thought": THOUGHT, "content": SPEECH}, ensure_ascii=False)          # raw unicode
    assert _run(raw, n)[0] == SPEECH


def test_random_splits_never_leak_the_thought():
    raw = json.dumps({"thought": THOUGHT, "content": SPEECH, "extra": {"content": "nested, not mine"}},
                     ensure_ascii=True)
    rng = random.Random(7)
    for _ in range(200):
        cuts = sorted(rng.sample(range(1, len(raw)), rng.randint(1, 40)))
        parts = [raw[a:b] for a, b in zip([0] + cuts, cuts + [len(raw)])]
        e = ContentExtractor((("content",),))
        got = [e.feed(p) for p in parts]
        assert "".join(got) == SPEECH
        assert "秘密" not in "".join(got) and "nested" not in "".join(got)


def test_a_surrogate_pair_cut_between_its_halves():
    raw = '{"content": "a\\ud83d' + '\\ude00b"}'
    assert _run(raw, 1)[0] == "a😀b"
    assert _run('{"content": "x\\ud83dy"}', 1)[0] == "x\ufffdy"          # an unpaired half is not dropped silently


def test_the_json_protocol_envelope_and_prose_around_it():
    env = {"thought": THOUGHT, "action": "speak", "args": {"thought": "inner secret", "content": SPEECH}}
    raw = "Sure! ```json\n" + json.dumps(env, ensure_ascii=False) + "\n```"
    out, e = _run(raw, 4, paths=(("args", "content"),))
    assert out == SPEECH
    assert ContentExtractor((("args", "content"),)).feed('{"content": "top level is not args"}') == ""


def test_it_streams_as_it_goes_not_at_the_end():
    e = ContentExtractor((("content",),))
    assert e.feed('{"content": "我觉') == "我觉"
    assert e.feed('得 5 号') == "得 5 号"
    assert not e.done and e.feed('"}') == "" and e.done


def _speech(deltas):
    shown, voids = [], []
    s = SpeechStream(shown.append, voids.append)
    for d in deltas:
        s.feed(d)
    return "".join(shown), voids


def test_native_tool_call_fragments():
    raw = json.dumps({"thought": THOUGHT, "content": SPEECH}, ensure_ascii=False)
    frags = [Delta("tool", c, "speak" if i == 0 else None) for i, c in enumerate(_chunks(raw, 3))]
    assert _speech(frags) == (SPEECH, [])
    # a vote has no speech, whatever its arguments say
    assert _speech([Delta("tool", '{"content": "x", "target_id": 3}', "vote")]) == ("", [])
    # the name may arrive after the first fragments: they are held, not lost
    late = [Delta("tool", raw[:10]), Delta("tool", raw[10:], "pack_say")]
    assert _speech(late)[0] == SPEECH


def test_json_protocol_that_turns_out_not_to_be_a_speech_is_voided():
    raw = '{"args": {"content": "早早开口"}, "action": "self_detonate"}'
    shown, voids = _speech([Delta("text", c) for c in _chunks(raw, 5)])
    assert shown == "早早开口" and voids and "self_detonate" in voids[0]


def _play_with_drafts(seed: int, variant: str = "8p", turns: int = 400):
    """A whole game of streaming LLM seats; yields (turn, action, player, draft events of that turn)."""
    from werewolf_harness.arena import Terminal
    from werewolf_harness.envs import Werewolf
    from werewolf_harness.players import GuardStack, LLMPlayer, MockClient
    env = Werewolf(variant)
    turn = env.reset(seed)
    guard = GuardStack(("L1", "L2", "L3"))
    for _ in range(turns):
        if isinstance(turn, Terminal):
            return
        events: list[tuple[str, int, str]] = []
        p = LLMPlayer(MockClient(seed=seed * 100 + turn.actor, speech_tokens=40, chunk_tokens=3), guard=guard,
                      on_draft=lambda k, a, t, ev=events: ev.append((k, a, t)))
        view, spec = env.observe(turn.actor), env.action_space(turn.actor, turn.task)
        action = p.act(view, turn.task, spec, None)
        if spec.check(action) is not None:
            p.abandon_draft(p.last_turn.draft_attempt, "spec")
            action = spec.fallback
        yield turn, action, p, events
        turn = env.step(turn.actor, turn.task, action)


def test_every_streamed_attempt_ends_as_the_action_or_an_abandon():
    shown = 0
    for seed in range(3):
        for turn, action, p, events in _play_with_drafts(seed):
            attempts = {a for k, a, _ in events if k == "delta"}
            abandoned = [a for k, a, _ in events if k == "abandon"]
            kept = p.last_turn.draft_attempt
            assert len(abandoned) == len(set(abandoned)), "an attempt was abandoned twice"
            for a in attempts:
                assert (a in abandoned) != (a == kept), f"attempt {a} of {turn} is neither kept nor abandoned"
            if kept is not None:
                text = "".join(t for k, a, t in events if k == "delta" and a == kept)
                assert text == action["args"]["content"]
                shown += 1
            assert not any("says something about" in t for k, _, t in events if k == "delta"), "a thought leaked"
            if turn.task not in ("speak", "last_words", "campaign", "pack_chat", "review"):
                assert not events
    assert shown > 10


def test_a_reset_voids_and_starts_over():
    a = '{"content": "first try'
    b = '{"content": "second"}'
    shown, voids = _speech([Delta("tool", a, "speak"), Delta("reset"), Delta("tool", b, "speak")])
    assert voids == ["reset"] and shown == "first try" + "second"
