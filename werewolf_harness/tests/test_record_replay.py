"""Record/replay of model I/O (players/providers/recorded.py, replay_llm.py)."""
from __future__ import annotations

import json
import os
import random
import time

import pytest

from werewolf_harness.arena import run
from werewolf_harness.cli import main as cli_main
from werewolf_harness.envs import Werewolf
from werewolf_harness.players.llm import loop
from werewolf_harness.players.providers import recorded
from werewolf_harness.players.providers.base import (Cancelled, CancelToken, Delta, LLMClient, LLMResponse,
                                                     ProviderError, ToolCall, collect)
from werewolf_harness.players.providers.mock import ChaosMockClient, MockClient
from werewolf_harness.players.providers.recorded import (Cassette, CassetteWriter, RecordingClient, ReplayClient,
                                                        ReplayDivergence, call_scope)
from werewolf_harness.replay_llm import build_players, check_expectations, compare, load, replay_game
from werewolf_harness.scripts.make_regression import make_case, run_case

SETTINGS = {"guard": "L1,L2,L3", "evidence": False, "playbook": "static", "temperature": 0.7, "lang": "zh",
            "max_tokens": 6_000_000, "max_usd": None, "parallel": 8}


class Jitter(LLMClient):
    """A deterministic client that finishes in a random order: the replies depend only on
    the request, the timing does not."""

    def __init__(self, inner):
        self.inner, self.name, self.tool_mode = inner, inner.name, inner.tool_mode
        self.rng = random.Random()

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0):
        time.sleep(self.rng.random() * 0.004)
        return self.inner.chat(messages, tools, temperature, max_tokens, timeout)


def record_game(tmp_path, name, make_client, *, variant="8p", seed=3, parallel=8, settings=None,
                engine_version=None, on_draft=None):
    """What `play --record --out` does, returning the log dict and its path.

    on_draft: stream the seats' speech as the online service does (a cancel token and a
    draft watcher on every seat), so the calls go through chat_stream."""
    s = dict(SETTINGS, parallel=parallel, **(settings or {}))
    out = str(tmp_path / f"{name}.json")
    w = CassetteWriter(recorded.cassette_path_for(out), meta={"variant": variant, "seed": seed})
    players, _t, guard = build_players(s, 8 if variant == "8p" else 12,
                                       lambda seat: RecordingClient(make_client(seat), w, seat=seat))
    for p in players.values():                     # no real backoff in tests
        p.policy.sleep = lambda _s: None
        if on_draft is not None:
            p.cancel = CancelToken()
            p.on_draft = (lambda seat: lambda kind, attempt, text: on_draft(seat, kind, attempt, text))(p)
    env = Werewolf(variant, playbook_mode=s["playbook"], engine_version=engine_version)
    t0 = time.time()
    ep = run(env, players, seed, parallel=parallel)
    w.close()
    log = {"variant": variant, "seed": seed, "model": "test", "guard": guard.label(), "playbook": s["playbook"],
           "engine_version": env.engine_version, "settings": s, "parallel": parallel, "outcome": ep.outcome, "duration_s": time.time() - t0,
           "journal": [r.__dict__ for r in ep.journal], "misses": ep.misses,
           "turns": [t.to_dict() for p in players.values() for t in p.history],
           "cassette": os.path.basename(w.path)}
    with open(out, "w") as f:
        json.dump(log, f, ensure_ascii=False, default=str)
    return log, out


def keys_and_hashes(path):
    c = Cassette.load(path)
    return {k: v["h"] for k, v in c.calls.items()}


# ------------------------------------------------------------------ keys
def test_keys_do_not_depend_on_thread_timing(tmp_path):
    """Same deterministic replies, different interleavings: same keys, same request hashes."""
    _, a = record_game(tmp_path, "a", lambda s: Jitter(MockClient(seed=300 + s)), variant="12p")
    _, b = record_game(tmp_path, "b", lambda s: Jitter(MockClient(seed=300 + s)), variant="12p")
    ka, kb = keys_and_hashes(recorded.cassette_path_for(a)), keys_and_hashes(recorded.cassette_path_for(b))
    assert len(ka) > 100 and ka == kb
    kinds = {k.rsplit("/", 1)[1] for k in ka}
    assert {"step", "note"} <= kinds


def test_scope_survives_the_model_call_pool():
    """call_model runs the client on a pool thread; the scope must still be there."""
    from werewolf_harness.players.recovery import RecoveryPolicy, RecoveryStats, call_model
    seen = []

    class Spy(LLMClient):
        def chat(self, *a, **k):
            seen.append(recorded.current_scope())
            return LLMResponse(text="x")
    with call_scope(4, 7, "vote", "forced"):
        call_model(Spy(), [{"role": "user", "content": "hi"}], None, policy=RecoveryPolicy(), stats=RecoveryStats(),
                   temperature=0.7, max_tokens=10)
    assert seen == [recorded.Scope(4, 7, "vote", "forced")]


# ------------------------------------------------------------------ exact replay
def test_parallel_chaos_game_replays_exactly(tmp_path):
    """A game no seed can reproduce (unseeded choices, random latency, 503s, junk replies),
    played 8 seats in parallel with background notes, replays to the same journal,
    outcome, beliefs and notes -- and the replay calls no model."""
    log, path = record_game(tmp_path, "chaos", lambda s: ChaosMockClient(seed=s), variant="12p", seed=5)
    log2, cas = load(path)
    rep = replay_game(log2, cas)
    assert compare(log2, rep) == []
    assert rep.used == set(cas.calls) and rep.calls == len(cas.calls)
    errs = sum(len(t["recovery"]["errors"]) for t in log2["turns"])
    assert errs > 0, "the chaos client should have failed at least once"
    assert all(t["note_after"] is not None for t in rep.turns)


def test_replay_uses_the_engine_version_the_game_was_recorded_under(tmp_path):
    """The rules text is in every prompt: an old-rules game must be replayed under its own
    rules (and a log without the field is the baseline), or every call would drift."""
    from werewolf_harness.envs import BASELINE_ENGINE_VERSION, ENGINE_VERSION
    assert BASELINE_ENGINE_VERSION != ENGINE_VERSION
    log, out = record_game(tmp_path, "old", lambda s: MockClient(seed=40 + s), engine_version=BASELINE_ENGINE_VERSION)
    rep = replay_game(*load(out))
    assert not rep.error and not rep.drifts and compare(log, rep) == []
    log.pop("engine_version")                      # a pre-versioning log: still the baseline
    rep = replay_game(log, load(out)[1])
    assert not rep.error and compare(log, rep) == []


def test_replay_matches_without_a_recording_it_would_not(tmp_path):
    """Control: the same seed played again live is a different game."""
    log_a, _ = record_game(tmp_path, "x", lambda s: ChaosMockClient(seed=s), seed=9)
    log_b, _ = record_game(tmp_path, "y", lambda s: ChaosMockClient(seed=s), seed=9)
    assert log_a["journal"] != log_b["journal"]


def test_errors_replay_as_the_same_errors(tmp_path):
    """Heavy failure rates: retries, forced terminals, transport-failure fallbacks and default
    actions all come back in the replay exactly, because the errors are recorded too."""
    # a seeded generator and one seat at a time: this game is the same every run
    log, path = record_game(tmp_path, "err", lambda s: ChaosMockClient(seed=s, error_rate=0.25, malformed_rate=0.35,
                                                                       rng=random.Random(1000 + s)),
                            parallel=1, seed=2)
    fb = [t["recovery"]["fallback_used"] for t in log["turns"] if t["recovery"]["fallback_used"]]
    assert fb, "expected fallbacks at these failure rates"
    log2, cas = load(path)
    assert any("err" in c for c in cas.calls.values())
    assert compare(log2, replay_game(log2, cas)) == []


def test_error_types_and_timeouts_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(recorded, "_hard_deadline", lambda t: 0.3)
    monkeypatch.setattr(recorded, "RECORD_SLACK_S", 0.1)

    class Bad(LLMClient):
        name = "bad"

        def __init__(self):
            self.n = 0

        def chat(self, *a, **k):
            self.n += 1
            if self.n == 1:
                raise ProviderError("quota 400", status=400, hint="h")
            if self.n == 2:
                raise KeyError("boom")
            time.sleep(1.0)
    path = str(tmp_path / "t.cassette.jsonl")
    w = CassetteWriter(path)
    rc = RecordingClient(Bad(), w, seat=1)
    got = []
    with call_scope(1, 0, "vote", "step"):
        for _ in range(3):
            try:
                rc.chat([{"role": "user", "content": "x"}], timeout=0.01)
            except BaseException as exc:      # noqa: BLE001
                got.append(exc)
    w.close()
    assert isinstance(got[0], ProviderError) and got[0].status == 400
    assert "timed out" in str(got[2]) and isinstance(got[2], ProviderError)
    rp = ReplayClient(Cassette.load(path), 1)
    again = []
    with call_scope(1, 0, "vote", "step"):
        for _ in range(3):
            try:
                rp.chat([{"role": "user", "content": "x"}], timeout=0.01)
            except BaseException as exc:      # noqa: BLE001
                again.append(exc)
    assert [type(e).__name__ for e in again] == ["ProviderError", "KeyError", "ProviderError"]
    assert [str(e) for e in again] == [str(e) for e in got]
    assert again[0].status == 400 and again[0].hint == "h"


def test_replayed_replies_are_fresh_objects(tmp_path):
    """The loop writes `thought` into a call's arguments; a second replay must not see it."""
    path = str(tmp_path / "f.cassette.jsonl")
    w = CassetteWriter(path)

    class One(LLMClient):
        def chat(self, *a, **k):
            return LLMResponse(tool_calls=[ToolCall("vote", {"target_id": 2, "b": 1, "a": 0})])
    with call_scope(1, 0, "vote", "step"):
        RecordingClient(One(), w, seat=1).chat([{"role": "user", "content": "x"}])
    w.close()
    c = Cassette.load(path)
    with call_scope(1, 0, "vote", "step"):
        r = ReplayClient(c, 1).chat([{"role": "user", "content": "x"}])
    assert list(r.tool_calls[0].arguments) == ["target_id", "b", "a"]      # order kept, not sorted
    r.tool_calls[0].arguments["thought"] = "x"
    assert "thought" not in c.calls["1/0/vote/0/step"]["resp"]["tool_calls"][0]["arguments"]


# ------------------------------------------------------------------ mismatch detection
def test_a_prompt_change_is_named_and_diffed(tmp_path, monkeypatch):
    log, path = record_game(tmp_path, "p", lambda s: MockClient(seed=s))
    monkeypatch.setattr(loop, "NOTE_PROMPT", loop.NOTE_PROMPT + " 另外写一句天气。")
    log2, cas = load(path)
    rep = replay_game(log2, cas)
    assert rep.error and "请求和录制时不一样" in rep.error
    assert "[note]" in rep.error and "座位" in rep.error
    assert "天气" in rep.error                                    # the diff shows the changed line
    drift = replay_game(log2, cas, allow_drift=True)
    assert drift.error is None and drift.drifts
    assert compare(log2, drift) == []                              # the replies still decide the same game
    assert all(d.key.endswith("/note") for d in drift.drifts)


def test_a_call_the_recording_never_made_is_an_error(tmp_path):
    c = Cassette.load(_one_call_cassette(tmp_path))
    rp = ReplayClient(c, 1)
    with call_scope(1, 0, "vote", "step"):
        rp.chat([{"role": "user", "content": "x"}])
        with pytest.raises(ReplayDivergence, match="录像里没有这一次调用"):
            rp.chat([{"role": "user", "content": "x"}])


def test_divergence_is_not_swallowed_by_recovery(tmp_path):
    """A mismatch must stop the game, not become a transport-failure fallback."""
    from werewolf_harness.players.recovery import RecoveryPolicy, RecoveryStats, call_model
    rp = ReplayClient(Cassette.load(_one_call_cassette(tmp_path)), 1)
    with call_scope(1, 0, "vote", "step"), pytest.raises(ReplayDivergence):
        call_model(rp, [{"role": "user", "content": "different"}], None, policy=RecoveryPolicy(),
                   stats=RecoveryStats(), temperature=0.7, max_tokens=800)


def _one_call_cassette(tmp_path) -> str:
    path = str(tmp_path / "one.cassette.jsonl")
    w = CassetteWriter(path)
    with call_scope(1, 0, "vote", "step"):
        RecordingClient(MockClient(), w, seat=1).chat([{"role": "user", "content": "x"}], max_tokens=800)
    w.close()
    return path


# ------------------------------------------------------------------ regression cases + CLI
def test_make_case_and_run_it(tmp_path):
    _, path = record_game(tmp_path, "g", lambda s: ChaosMockClient(seed=s), variant="12p", seed=4)
    folder, failing = make_case(path, "c1", upto=25, expect=["max_forced:0", "action:0:night_kill"],
                                dest=str(tmp_path / "cases"))
    assert run_case(folder) == check_expectations_of(folder)
    full = Cassette.load(recorded.cassette_path_for(path))
    small = Cassette.load(os.path.join(folder, "cassette.jsonl.gz"))
    assert 0 < len(small.calls) < len(full.calls)
    # an expectation that does not hold is reported by the runner
    case = json.load(open(os.path.join(folder, "case.json")))
    case["expect"] = ["winner:nobody"]
    json.dump(case, open(os.path.join(folder, "case.json"), "w"))
    assert any("winner:nobody" in p for p in run_case(folder))


def check_expectations_of(folder):
    case = json.load(open(os.path.join(folder, "case.json")))
    log, cas = load(os.path.join(folder, "game.json.gz"))
    return check_expectations(replay_game(log, cas, upto=case["upto"]), case["expect"])


def test_cli_record_then_replay(tmp_path, capsys):
    out = str(tmp_path / "rec" / "g.json")
    assert cli_main(["play", "--variant", "8p", "--seed", "11", "--model", "mock-chaos", "--parallel", "8",
                     "--record", "--out", out]) == 0
    assert os.path.exists(recorded.cassette_path_for(out))
    assert cli_main(["replay-llm", out, "--strict"]) == 0
    assert "一致 ✓" in capsys.readouterr().out


def test_old_logs_without_settings_still_parse():
    from werewolf_harness.replay_llm import settings_of
    s = settings_of({"guard": "L1+L3+E", "playbook": "none", "parallel": 4})
    assert s["guard"] == "L1,L3" and s["evidence"] and s["playbook"] == "none" and s["parallel"] == 4


# ------------------------------------------------------------------ server path
def test_online_table_records_when_asked(tmp_path, monkeypatch):
    from werewolf_harness.app.runtime.table import Table, TableConfig
    monkeypatch.setenv("LLM_RECORD_DIR", str(tmp_path / "cas"))
    t = Table("gREC", 7, TableConfig(variant="8p", record=True))
    t.reset()
    turn = t.turn
    p = t.make_player(turn.actor, task=turn.task, version=0)
    p.act(t.env.observe(turn.actor), turn.task, t.env.action_space(turn.actor, turn.task), None)
    c = Cassette.load(str(tmp_path / "cas" / "gREC.cassette.jsonl"))
    assert c.meta["game_id"] == "gREC" and c.calls
    assert all(k.startswith(f"{turn.actor}/0.0/{turn.task}/") for k in c.calls)
    off = Table("gOFF", 7, TableConfig(variant="8p"))
    assert off.cassette() is None
    # the service streams every model seat (a cancel token on each job): same lines on the cassette
    ts = Table("gRECs", 7, TableConfig(variant="8p", record=True))
    ts.reset()
    ps = ts.make_player(turn.actor, task=turn.task, version=0)
    ps.cancel, ps.on_draft = CancelToken(), lambda *a: None
    ps.act(ts.env.observe(turn.actor), turn.task, ts.env.action_space(turn.actor, turn.task), None)
    cs = Cassette.load(str(tmp_path / "cas" / "gRECs.cassette.jsonl"))
    assert {k: v["h"] for k, v in cs.calls.items()} == {k: v["h"] for k, v in c.calls.items()}


# ------------------------------------------------------------------ streaming (docs/adr/0009)
def test_a_game_recorded_while_streaming_replays_identically(tmp_path):
    """Streaming is delivery, not content: the cassette of a streamed game has the same keys
    and request hashes as the same game played without streaming, and it replays (through
    the non-streaming path) to the same journal and turn records."""
    drafts: list = []
    mk = lambda s: MockClient(seed=60 + s, speech_tokens=40, chunk_tokens=3)      # noqa: E731
    log_s, path_s = record_game(tmp_path, "streamed", mk, on_draft=lambda *a: drafts.append(a))
    log_p, path_p = record_game(tmp_path, "plain", mk)
    assert sum(1 for d in drafts if d[1] == "delta") > 20, "the speech should have been streamed in pieces"
    assert keys_and_hashes(recorded.cassette_path_for(path_s)) == keys_and_hashes(recorded.cassette_path_for(path_p))
    assert log_s["journal"] == log_p["journal"]
    log2, cas = load(path_s)
    rep = replay_game(log2, cas)
    assert compare(log2, rep) == []
    assert rep.used == set(cas.calls)


def test_a_streamed_game_replays_through_the_streaming_path(tmp_path):
    """And the other way round: the replay itself streamed (seats with drafts on)."""
    log, path = record_game(tmp_path, "s2", lambda s: ChaosMockClient(seed=s, rng=random.Random(70 + s)),
                            parallel=1, seed=4, on_draft=lambda *a: None)
    log2, cas = load(path)
    seen: list = []
    rep = replay_game(log2, cas, on_player=lambda p: (setattr(p, "cancel", CancelToken()),
                                                      setattr(p, "on_draft", lambda *a: seen.append(a))))
    assert compare(log2, rep) == []
    assert any(d[0] == "delta" for d in seen)


class _SlowStream(LLMClient):
    name = "slow"

    def __init__(self, stall: float):
        self.stall = stall

    def chat(self, *a, **k):
        return LLMResponse(text="hi")

    def chat_stream(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0, cancel=None):
        yield Delta("text", "h")
        if cancel.wait(self.stall):
            raise Cancelled(cancel.reason or "cancelled", 1)
        yield Delta("text", "i")
        yield LLMResponse(text="hi", completion_tokens=1)


def test_streamed_calls_record_the_same_lines_as_chat(tmp_path, monkeypatch):
    monkeypatch.setattr(recorded, "_hard_deadline", lambda t: 0.3)
    monkeypatch.setattr(recorded, "RECORD_SLACK_S", 0.1)
    path = str(tmp_path / "s.cassette.jsonl")
    w = CassetteWriter(path)
    msgs = [{"role": "user", "content": "x"}]
    got = []
    with call_scope(1, 0, "speak", "step"):
        pieces: list = []
        got.append(collect(RecordingClient(_SlowStream(0.0), w, seat=1).chat_stream(msgs), pieces.append))
        assert [p.text for p in pieces] == ["h", "i"]
        with pytest.raises(ProviderError, match="timed out"):          # past the deadline: the plug is pulled
            collect(RecordingClient(_SlowStream(5.0), w, seat=1).chat_stream(msgs, cancel=CancelToken()))
        outer = CancelToken()
        stream = RecordingClient(_SlowStream(5.0), w, seat=1).chat_stream(msgs, cancel=outer)
        next(stream)
        outer.cancel("turn_gone")
        with pytest.raises(Cancelled):
            list(stream)
    w.close()
    c = Cassette.load(path)
    assert c.order == ["1/0/speak/0/step", "1/0/speak/1/step", "1/0/speak/2/step"]
    rp = ReplayClient(c, 1)
    with call_scope(1, 0, "speak", "step"):
        assert collect(rp.chat_stream(msgs)).text == got[0].text
        with pytest.raises(ProviderError, match="timed out"):
            collect(rp.chat_stream(msgs))
        with pytest.raises(Cancelled):                                 # comes back as itself, not a look-alike
            collect(rp.chat_stream(msgs))
