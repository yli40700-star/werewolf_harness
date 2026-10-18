"""The `claude -p` seat, against a fake executable that records what it was given."""
from __future__ import annotations

import json
import os
import stat
import sys
import textwrap

import pytest

from werewolf_harness.app.llm.gateway import classify
from werewolf_harness.players.providers import offered, profile
from werewolf_harness.players.providers.base import ProviderError
from werewolf_harness.players.providers.claude_cli import (MENU, ClaudeCLIClient, cli_model_arg, is_cli_model,
                                                          render_transcript)
from werewolf_harness.players.recovery import is_terminal_refusal
from werewolf_harness.server.service import _client

FAKE = textwrap.dedent("""\
    #!{python}
    import json, os, sys, time
    log = os.environ["FAKE_CLAUDE_LOG"]
    prompt = sys.stdin.read()
    with open(log, "w") as f:
        json.dump({{"argv": sys.argv[1:], "stdin": prompt, "cwd": os.getcwd(),
                   "api_key": os.environ.get("ANTHROPIC_API_KEY")}}, f)
    mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
    if mode == "slow":
        time.sleep(5)
    if mode == "auth":
        print(json.dumps({{"type": "result", "subtype": "success", "is_error": True,
                          "result": "Not logged in · Please run /login"}}))
        sys.exit(1)
    if mode == "limit":
        print(json.dumps({{"type": "result", "subtype": "success", "is_error": True,
                          "result": "Claude AI usage limit reached", "api_error_status": 429}}))
        sys.exit(1)
    if mode == "garbage":
        print("segfault-ish noise"); sys.exit(3)
    reply = '{{"thought": "5 keeps dodging", "action": "vote", "args": {{"target_id": 5, "thought": "dodging"}}}}'
    print(json.dumps({{"type": "result", "subtype": "success", "is_error": False, "result": reply,
                      "stop_reason": "end_turn", "total_cost_usd": 0.01, "duration_ms": 900,
                      "usage": {{"input_tokens": 12, "cache_creation_input_tokens": 300,
                                "cache_read_input_tokens": 4000, "output_tokens": 40}},
                      "modelUsage": {{"claude-sonnet-x": {{}}}}}}))
    """)

VOTE = [{"type": "function", "function": {"name": "vote", "description": "Cast your vote.", "parameters": {
    "type": "object", "properties": {"target_id": {"type": "integer", "enum": [2, 5]},
                                     "thought": {"type": "string"}}, "required": ["target_id"]}}}]
MSGS = [{"role": "system", "content": "RULES OF THE TABLE"}, {"role": "user", "content": "Vote now."}]


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    exe = tmp_path / "claude"
    exe.write_text(FAKE.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "call.json"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")

    def last_call() -> dict:
        return json.loads(log.read_text())

    return {"bin": str(exe), "env": {"CLAUDE_CLI_BIN": str(exe)}, "last": last_call}


def test_model_names():
    assert is_cli_model("claude-cli") and is_cli_model("claude-cli:opus")
    assert not is_cli_model("claude-opus-5") and not is_cli_model("claude-clix")
    assert cli_model_arg("claude-cli") is None and cli_model_arg("claude-cli:sonnet") == "sonnet"
    assert profile("claude-cli:haiku")["tool_mode"] == "json_prompt"


def test_a_vote_comes_back_as_a_tool_call(fake):
    c = ClaudeCLIClient("claude-cli:sonnet", env=fake["env"])
    r = c.chat(MSGS, tools=VOTE, timeout=20)
    assert r.tool_calls[0].name == "vote" and r.tool_calls[0].arguments["target_id"] == 5
    assert not r.tool_calls[0].malformed
    assert r.prompt_tokens == 12 + 300 + 4000 and r.completion_tokens == 40
    assert r.raw["usage"]["cache_read_input_tokens"] == 4000 and r.model == "claude-sonnet-x"


def test_the_command_line_switches_claude_code_off(fake):
    ClaudeCLIClient("claude-cli:sonnet", env=fake["env"]).chat(MSGS, tools=VOTE, timeout=20)
    call = fake["last"]()
    argv = call["argv"]
    assert argv[0] == "-p" and argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--tools") + 1] == "" and "--strict-mcp-config" in argv
    assert "--no-session-persistence" in argv and argv[argv.index("--model") + 1] == "sonnet"
    system = argv[argv.index("--system-prompt") + 1]
    assert system.startswith("RULES OF THE TABLE") and "TOOL PROTOCOL" in system and "vote(" in system
    assert call["stdin"] == "Vote now."
    assert "--bare" not in argv and not os.listdir(call["cwd"])     # an empty working directory
    assert "--setting-sources=" in argv                              # lean start: no settings files


def test_lean_start_can_be_switched_off(fake):
    ClaudeCLIClient("claude-cli", env={**fake["env"], "CLAUDE_CLI_LEAN": "0"}).chat(MSGS, tools=VOTE, timeout=20)
    assert "--setting-sources=" not in fake["last"]()["argv"]


def test_plain_claude_cli_uses_the_cli_default_model(fake):
    ClaudeCLIClient("claude-cli", env=fake["env"]).chat(MSGS, tools=VOTE, timeout=20)
    assert "--model" not in fake["last"]()["argv"]


def test_a_react_transcript_is_rendered_in_order():
    msgs = MSGS + [{"role": "assistant", "content": '{"action": "query_history", "args": {"player_id": 5}}'},
                   {"role": "user", "content": "OBSERVATION: 5 said he was the seer"}]
    text = render_transcript(msgs)
    assert "RULES" not in text
    assert text.index("Vote now.") < text.index("[you]\n{") < text.index("OBSERVATION: 5 said")
    assert text.rstrip().endswith("exactly one JSON object, nothing else.")


def test_not_logged_in_is_terminal(fake, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "auth")
    with pytest.raises(ProviderError) as e:
        ClaudeCLIClient("claude-cli", env=fake["env"]).chat(MSGS, tools=VOTE, timeout=20)
    assert e.value.status == 401 and is_terminal_refusal(e.value)


def test_a_usage_limit_moves_to_the_next_model(fake, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "limit")
    with pytest.raises(ProviderError) as e:
        ClaudeCLIClient("claude-cli", env=fake["env"]).chat(MSGS, tools=VOTE, timeout=20)
    assert e.value.status == 429 and classify(e.value) == "rate_limited" and not is_terminal_refusal(e.value)


def test_a_crash_is_an_upstream_error(fake, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "garbage")
    with pytest.raises(ProviderError) as e:
        ClaudeCLIClient("claude-cli", env=fake["env"]).chat(MSGS, tools=VOTE, timeout=20)
    assert classify(e.value) == "upstream_error"


def test_a_slow_cli_times_out(fake, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "slow")
    with pytest.raises(ProviderError) as e:
        ClaudeCLIClient("claude-cli", env=fake["env"]).chat(MSGS, tools=VOTE, timeout=1.5)
    assert classify(e.value) == "timeout"


def test_a_missing_executable_stops_the_game(tmp_path):
    c = ClaudeCLIClient("claude-cli", env={"CLAUDE_CLI_BIN": str(tmp_path / "nope")})
    with pytest.raises(ProviderError) as e:
        c.chat(MSGS, tools=VOTE, timeout=5)
    assert is_terminal_refusal(e.value)


def test_an_api_key_in_env_reaches_the_cli_only_in_bare_mode(fake):
    env = {**fake["env"], "ANTHROPIC_API_KEY": "sk-test"}
    ClaudeCLIClient("claude-cli", env=env).chat(MSGS, tools=VOTE, timeout=20)
    assert fake["last"]()["api_key"] in (None, os.environ.get("ANTHROPIC_API_KEY"))
    ClaudeCLIClient("claude-cli", env={**env, "CLAUDE_CLI_BARE": "1"}).chat(MSGS, tools=VOTE, timeout=20)
    call = fake["last"]()
    assert call["api_key"] == "sk-test" and "--bare" in call["argv"]


def test_offered_only_when_the_executable_is_found(fake, tmp_path):
    assert set(MENU) <= set(offered(fake["env"]))
    assert not set(MENU) & set(offered({"CLAUDE_CLI_BIN": str(tmp_path / "nope")}))


def test_the_service_builds_a_cli_seat(fake):
    c = _client("claude-cli:opus", 1, fake["env"])
    assert isinstance(c, ClaudeCLIClient) and c.cli_model == "opus" and c.tool_mode == "json_prompt"


def test_ai_seats_speak_chinese_unless_told_otherwise():
    from werewolf_harness.envs import Werewolf
    from werewolf_harness.players.guard import GuardStack
    from werewolf_harness.players.llm.context import ContextBuilder
    env = Werewolf("8p")
    env.reset(1)
    view = env.observe(1)
    zh = ContextBuilder(GuardStack()).system_message(view, ["speak"], [])
    en = ContextBuilder(GuardStack(), lang="en").system_message(view, ["speak"], [])
    assert "简体中文" in zh and "简体中文" not in en


def test_split_system_sends_only_the_shared_part_as_the_system_prompt():
    from werewolf_harness.players.providers.claude_cli import ClaudeCLIClient
    msgs = [{"role": "system", "content": "RULES\nCONTRACT\nYOUR IDENTITY (private): seat 3\nTHIS TURN: vote"},
            {"role": "user", "content": "go"}]
    plain = ClaudeCLIClient("claude-cli:haiku", env={})
    split = ClaudeCLIClient("claude-cli:haiku", env={"CLAUDE_CLI_SPLIT_SYSTEM": "1"})
    assert "seat 3" in plain.prompts(msgs, None)[0]
    system, stdin = split.prompts(msgs, None)
    assert system == "RULES\nCONTRACT" and stdin.startswith("YOUR IDENTITY (private): seat 3") and stdin.endswith("go")
