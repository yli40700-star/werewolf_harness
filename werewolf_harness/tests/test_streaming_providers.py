"""chat_stream for every provider: SSE (OpenAI-compatible), `claude -p` stream-json, the mock;
and cancellation that actually stops the upstream call."""
from __future__ import annotations

import http.server
import json
import os
import stat
import sys
import textwrap
import threading
import time

import pytest

from werewolf_harness.players.providers.base import (Cancelled, CancelToken, Delta, LLMClient, LLMResponse,
                                                     ProviderError, ToolCall, collect)
from werewolf_harness.players.providers.claude_cli import ClaudeCLIClient, StreamJsonParser
from werewolf_harness.players.providers.mock import STREAM_PROFILES, MockClient
from werewolf_harness.players.providers.openai_compat import OpenAICompatClient, StreamAssembler, iter_sse
from werewolf_harness.players.recovery import RecoveryPolicy, RecoveryStats, call_model

SPEAK = [{"type": "function", "function": {"name": "speak", "description": "Say something. ENDS YOUR TURN.",
                                           "parameters": {"type": "object", "properties": {
                                               "content": {"type": "string"}, "thought": {"type": "string"}},
                                               "required": ["content"]}}}]
MSGS = [{"role": "system", "content": "You are player 3, role: villager"}, {"role": "user", "content": "=== ROUND 1"}]


# ---------------------------------------------------------------- SSE framing
def test_sse_framing_and_its_tolerances():
    lines = [b": keep-alive\n", b"data: {\"a\": 1}\n", b"\n",
             b"data: {\"b\":\n", b"data:  2}\n", b"\n",               # one event over two data lines
             b"data: {\"c\": 3}\r\n", b"data: {\"d\": 4}\n",           # no blank line between events
             b"event: ping\n", b"\n", "data: {\"e\": \"中文\"}\n".encode(), b"data: [DONE]\n"]
    got = list(iter_sse(lines))
    assert got[0] == '{"a": 1}' and json.loads(got[1]) == {"b": 2}
    assert got[2:4] == ['{"c": 3}', '{"d": 4}'] and json.loads(got[4]) == {"e": "中文"} and got[-1] == "[DONE]"


def test_the_assembler_rebuilds_tool_calls_from_argument_deltas():
    acc = StreamAssembler()
    chunks = [
        {"model": "m1", "choices": [{"delta": {"role": "assistant", "tool_calls": [
            {"index": 0, "id": "call_9", "type": "function", "function": {"name": "speak", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"thought": "秘'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '密", "content": "我'}}]}}]},
        {"choices": [{"delta": {"reasoning_content": "private scratchpad"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '是好人"}'}}]},
                      "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 12}},
    ]
    deltas = [d for c in chunks for d in acc.feed(c)]
    assert deltas[0].name == "speak" and all(d.kind == "tool" for d in deltas)
    assert "private scratchpad" not in "".join(d.text for d in deltas)
    msg = acc.message()
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"thought": "秘密", "content": "我是好人"}
    assert acc.finish_reason == "tool_calls" and acc.usage["completion_tokens"] == 12 and acc.model == "m1"
    with pytest.raises(ProviderError):
        acc.feed({"error": {"message": "upstream overloaded", "code": 503}})


# ---------------------------------------------------------------- a real HTTP stream
class _Relay(http.server.BaseHTTPRequestHandler):
    chunks: list = []
    gap = 0.0
    seen: list = []
    closed = threading.Event()

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(body)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            for c in type(self).chunks:
                self.wfile.write(f"data: {json.dumps(c, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
                time.sleep(type(self).gap)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            type(self).closed.set()                   # the client hung up: what a cancel looks like upstream


@pytest.fixture()
def relay():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Relay)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    _Relay.seen, _Relay.gap = [], 0.0
    _Relay.closed = threading.Event()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def _speech_chunks(text: str, n: int = 3) -> list[dict]:
    raw = json.dumps({"thought": "不能说的", "content": text}, ensure_ascii=False)
    out = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "speak",
                                                                                        "arguments": ""}}]}}]}]
    out += [{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": raw[i:i + n]}}]}}]}
            for i in range(0, len(raw), n)]
    out.append({"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 30}})
    return out


def test_openai_stream_yields_fragments_then_the_same_response(relay):
    _Relay.chunks = _speech_chunks("我是 3 号，我觉得 5 号可疑。")
    c = OpenAICompatClient("m", api_key="k", base_url=relay)
    items = list(c.chat_stream(MSGS, tools=SPEAK, timeout=10))
    assert isinstance(items[-1], LLMResponse) and all(isinstance(i, Delta) for i in items[:-1])
    resp = items[-1]
    assert resp.tool_calls[0].name == "speak" and resp.tool_calls[0].arguments["content"] == "我是 3 号，我觉得 5 号可疑。"
    assert resp.completion_tokens == 30 and resp.raw["stream"] is True
    sent = _Relay.seen[-1]
    assert sent["stream"] is True and sent["tools"] and sent["tool_choice"] == "required"


def test_cancelling_an_openai_stream_hangs_up_on_the_relay(relay):
    _Relay.chunks = _speech_chunks("长" * 300, n=2)
    _Relay.gap = 0.01
    c = OpenAICompatClient("m", api_key="k", base_url=relay)
    tok = CancelToken()
    got = []
    t0 = time.monotonic()
    with pytest.raises(Cancelled) as e:
        for item in c.chat_stream(MSGS, tools=SPEAK, timeout=10, cancel=tok):
            got.append(item)
            if len(got) == 20:
                threading.Timer(0.05, tok.cancel, args=("deadline",)).start()
    assert time.monotonic() - t0 < 3 and e.value.reason == "deadline" and e.value.produced > 0
    assert _Relay.closed.wait(5), "the relay never saw the client go"


def test_a_profile_can_switch_streaming_off(relay):
    c = OpenAICompatClient("m", api_key="k", base_url=relay, stream=False)
    assert c.stream is False


# ---------------------------------------------------------------- claude -p stream-json
FAKE = textwrap.dedent("""\
    #!{python}
    import json, os, sys, time
    log = os.environ["FAKE_CLAUDE_LOG"]
    prompt = sys.stdin.read()
    with open(log, "w") as f:
        json.dump({{"argv": sys.argv[1:], "stdin": prompt, "pid": os.getpid()}}, f)
    mode = os.environ.get("FAKE_CLAUDE_MODE", "stream")
    def out(o):
        print(json.dumps(o, ensure_ascii=False), flush=True)
    reply = json.dumps({{"thought": "我是狼，别说", "action": "speak",
                        "args": {{"content": "我是 3 号，昨晚 5 号的发言有问题。"}}}}, ensure_ascii=False)
    out({{"type": "system", "subtype": "init", "model": "claude-x", "tools": []}})
    print("some warning that is not json", flush=True)
    if mode == "error":
        out({{"type": "result", "subtype": "success", "is_error": True, "result": "Claude AI usage limit reached",
             "api_error_status": 429}})
        sys.exit(1)
    if mode == "stream" or mode == "slow":
        out({{"type": "stream_event", "event": {{"type": "message_start", "message": {{"usage": {{}}}}}}}})
        out({{"type": "stream_event", "event": {{"type": "content_block_delta", "index": 0,
             "delta": {{"type": "thinking_delta", "thinking": "SECRET REASONING"}}}}}})
        for i in range(0, len(reply), 4):
            out({{"type": "stream_event", "event": {{"type": "content_block_delta", "index": 0,
                 "delta": {{"type": "text_delta", "text": reply[i:i + 4]}}}}}})
            time.sleep(0.5 if mode == "slow" else 0.01)
    out({{"type": "assistant", "message": {{"content": [{{"type": "text", "text": reply}}]}}}})
    with open(log + ".done", "w") as f:
        f.write("finished")
    out({{"type": "result", "subtype": "success", "is_error": False, "result": reply, "stop_reason": "end_turn",
         "usage": {{"input_tokens": 10, "cache_read_input_tokens": 100, "output_tokens": 33}},
         "modelUsage": {{"claude-x": {{}}}}}})
    """)


@pytest.fixture()
def fake_cli(tmp_path, monkeypatch):
    exe = tmp_path / "claude"
    exe.write_text(FAKE.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "call.json"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "stream")
    return {"env": {"CLAUDE_CLI_BIN": str(exe)}, "log": log}


def test_claude_stream_json_yields_text_deltas_and_the_same_result(fake_cli):
    c = ClaudeCLIClient("claude-cli:sonnet", env=fake_cli["env"])
    items = list(c.chat_stream(MSGS, tools=SPEAK, timeout=20))
    resp = items[-1]
    text = "".join(d.text for d in items[:-1])
    assert len(items) > 5 and "SECRET REASONING" not in text
    assert resp.tool_calls[0].name == "speak" and resp.tool_calls[0].arguments["content"].startswith("我是 3 号")
    assert text == resp.text and resp.prompt_tokens == 110 and resp.completion_tokens == 33
    argv = json.loads(fake_cli["log"].read_text())["argv"]
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--include-partial-messages" in argv and "--verbose" in argv


def test_claude_without_partials_still_streams_once(fake_cli, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "whole")
    items = list(ClaudeCLIClient("claude-cli", env=fake_cli["env"]).chat_stream(MSGS, tools=SPEAK, timeout=20))
    assert len(items) == 2 and items[0].kind == "text" and items[0].text == items[1].text


def test_claude_stream_errors_map_like_the_json_output(fake_cli, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "error")
    with pytest.raises(ProviderError) as e:
        list(ClaudeCLIClient("claude-cli", env=fake_cli["env"]).chat_stream(MSGS, tools=SPEAK, timeout=20))
    assert e.value.status == 429


def test_cancelling_claude_kills_the_process(fake_cli, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "slow")
    tok = CancelToken()
    c = ClaudeCLIClient("claude-cli", env=fake_cli["env"])
    t0 = time.monotonic()
    with pytest.raises(Cancelled) as e:
        for item in c.chat_stream(MSGS, tools=SPEAK, timeout=60, cancel=tok):
            tok.cancel("turn deadline")
    assert time.monotonic() - t0 < 5 and e.value.reason == "turn deadline"
    pid = json.loads(fake_cli["log"].read_text())["pid"]
    time.sleep(0.2)
    with pytest.raises(OSError):
        os.kill(pid, 0)                                        # gone
    assert not os.path.exists(str(fake_cli["log"]) + ".done")  # and it never finished its reply


def test_a_slow_claude_stream_times_out(fake_cli, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "slow")
    with pytest.raises(ProviderError) as e:
        list(ClaudeCLIClient("claude-cli", env=fake_cli["env"]).chat_stream(MSGS, tools=SPEAK, timeout=1.2))
    assert "timed out" in str(e.value)


def test_the_stream_json_parser_skips_what_it_does_not_know():
    p = StreamJsonParser()
    assert p.feed("not json") == [] and p.feed('{"type": "rate_limit_event"}') == []
    assert p.feed('{"type": "stream_event", "event": {"type": "content_block_delta", "delta": {}}}') == []
    d = p.feed('{"type":"stream_event","event":{"type":"content_block_delta","index":0,'
               '"delta":{"type":"text_delta","text":"hi"}}}')
    assert d[0].text == "hi" and p.text() == "hi"


# ---------------------------------------------------------------- the mock and the default
def test_mock_stream_is_the_same_decision_as_chat():
    for seed in range(5):
        a = MockClient(seed=seed).chat(MSGS, tools=SPEAK)
        items = list(MockClient(seed=seed).chat_stream(MSGS, tools=SPEAK))
        b = items[-1]
        assert b.tool_calls[0].arguments == a.tool_calls[0].arguments
        assert "".join(d.text for d in items[:-1]) == b.tool_calls[0].raw_arguments


def test_mock_stream_timing_and_cancel():
    c = MockClient(seed=1, ttft_s=0.05, tokens_per_s=400, chunk_tokens=4, speech_tokens=80)
    t0 = time.monotonic()
    first = None
    for item in c.chat_stream(MSGS, tools=SPEAK):
        if first is None:
            first = time.monotonic() - t0
    full = time.monotonic() - t0
    assert 0.05 <= first < full and full >= 80 / 400
    tok = CancelToken()
    threading.Timer(0.1, tok.cancel, args=("stop",)).start()
    with pytest.raises(Cancelled) as e:
        collect(MockClient(seed=1, ttft_s=0.0, tokens_per_s=100, speech_tokens=400).chat_stream(
            MSGS, tools=SPEAK, cancel=tok))
    assert 0 < e.value.produced < 100
    assert set(STREAM_PROFILES) >= {"mock-stream", "mock-stream-fast"}


class _Plain(LLMClient):
    name = "plain"

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0):
        return LLMResponse(tool_calls=[ToolCall("speak", {"content": "x"}, raw_arguments='{"content": "x"}')])


def test_the_default_chat_stream_yields_once():
    items = list(_Plain().chat_stream(MSGS, tools=SPEAK))
    assert [type(i) for i in items] == [Delta, LLMResponse] and items[0].name == "speak"


# ---------------------------------------------------------------- call_model
class _Flaky(LLMClient):
    """Streams half a speech, then drops the connection; answers the second time."""
    name = "flaky"

    def __init__(self):
        self.calls = 0

    def chat(self, *a, **k):
        raise AssertionError("streaming path expected")

    def chat_stream(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0, cancel=None):
        self.calls += 1
        yield Delta("tool", '{"content": "half', "speak")
        if self.calls == 1:
            raise ProviderError("connection dropped", status=502)
        yield Delta("tool", '"}')
        yield LLMResponse(tool_calls=[ToolCall("speak", {"content": "half"}, raw_arguments='{"content": "half"}')])


def test_a_transport_retry_after_a_partial_stream_resets_it():
    got: list[Delta] = []
    pol = RecoveryPolicy(backoff_s=0.0)
    resp = call_model(_Flaky(), MSGS, SPEAK, policy=pol, stats=RecoveryStats(), temperature=0.7, max_tokens=100,
                      on_delta=got.append)
    assert resp is not None and [d.kind for d in got] == ["tool", "reset", "tool", "tool"]


def test_cancel_is_raised_through_call_model_not_retried():
    tok = CancelToken()
    tok.cancel("game stopped")
    with pytest.raises(Cancelled):
        call_model(MockClient(), MSGS, SPEAK, policy=RecoveryPolicy(), stats=RecoveryStats(), temperature=0.7,
                   max_tokens=100, cancel=tok)


class _Stuck(LLMClient):
    """A client that ignores cancel and blocks: the caller must still get out at once."""
    name = "stuck"

    def chat(self, *a, **k):
        time.sleep(3)
        return LLMResponse(text="late")


def test_a_cancel_reaches_the_caller_even_when_the_client_cannot_be_interrupted():
    tok = CancelToken()
    threading.Timer(0.2, tok.cancel, args=("stop",)).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        call_model(_Stuck(), MSGS, None, policy=RecoveryPolicy(), stats=RecoveryStats(), temperature=0.7,
                   max_tokens=10, cancel=tok)
    assert time.monotonic() - t0 < 1.5
