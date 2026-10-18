from .base import LLMClient, LLMResponse, ProviderError, ToolCall
from . import claude_cli
from .claude_cli import ClaudeCLIClient, is_cli_model
from .china import CANDIDATES, CANDIDATE_BY_MODEL, PROVIDERS, Candidate, Provider, provider_of
from .mock import MockClient
from .openai_compat import OpenAICompatClient


# How each model has to be spoken to on this gateway, measured rather than assumed.
#
# Two things vary and both were found the hard way:
#   tool_mode  the relay loses claude's native tool calls -- finish_reason says
#              "tool_calls" and the list comes back empty, billed. The JSON tool
#              protocol goes through the same relay and works, so claude speaks JSON.
#   timeout    on the native path claude-opus-5 takes 90-100s, so a 75s timeout read
#              it as a dead channel for a whole afternoon. On the JSON path the same
#              model answers in 4s -- the slowness was the relay's, not the model's.
# Latency, measured 2026-09-03 on the largest prompt the harness sends (4383
# chars, 13 tool schemas), 12 calls per model:
#
#   gemini-3.8-flash   中位  2.9s   p90   5.2s   最慢   5.8s
#   claude-opus-5      中位  6.8s   p90   9.4s   最慢  10.7s
#   gpt-5.6-terra      中位 40.1s   p90 118.1s   最慢 173.5s
#   gpt-5.6-sol        中位 53.9s   p90 105.2s   最慢 134.7s
#
# The gpt-5.6 pair is not slow so much as heavy-tailed: three separate runs put
# terra's median at 56.9s, 13.6s and 40.1s. There is no stable number to fit, so
# the timeout is set from the observed worst case, not the median -- a timeout
# below it does not save time, it converts a slow answer into a forced turn and
# a retry.
PROFILES: dict[str, dict] = {
    # Retired at the user's call. The 400s that killed a game were a schema
    # dialect bug, since fixed (see _as_string_enums -- Google allows `enum`
    # only on a string, and a seat number is an int), so this row is one key
    # away from working: delete `retired` to put it back.
    "gemini-3.8-flash":  {"tool_mode": "native",      "timeout_s": 60.0,  "note": "中位 3s，最快",
                          "retired": "用户下架"},
    # Retired: a seat that takes 40-170s per model call, several calls to a turn,
    # makes a game nobody sits through. The rows stay so the measurement is not
    # lost and the timeout is right if they are ever put back -- delete the
    # `retired` key to offer them again.
    "gpt-5.6-terra":     {"tool_mode": "native",      "timeout_s": 180.0, "note": "中位 40s，尾巴到 174s",
                          "retired": "太慢：中位 40s/次，最慢 174s"},
    "gpt-5.6-sol":       {"tool_mode": "native",      "timeout_s": 180.0, "note": "中位 54s，尾巴到 135s",
                          "retired": "太慢：中位 54s/次，最慢 135s"},
    "claude-sonnet-5":   {"tool_mode": "json_prompt", "timeout_s": 120.0, "note": "原生工具调用会被网关丢掉，走 JSON 协议"},
    "claude-opus-5":     {"tool_mode": "json_prompt", "timeout_s": 120.0, "note": "中位 7s（原生工具调用那条路要 90-100s 且会丢结果）"},
}


def profile(model: str) -> dict:
    """How to speak to one model.

    A model on a Chinese provider is described by its Candidate row rather than
    by PROFILES: those rows carry the vendor's quirks and, deliberately, no
    latency number -- none has been measured yet.
    """
    if model in PROFILES:
        return PROFILES[model]
    if is_cli_model(model):
        # A CLI call pays a process start (1-2 s) on top of the model; 120 s matches the
        # relay's claude rows, which take the same JSON path.
        return {"tool_mode": "json_prompt", "timeout_s": 120.0, "note": "本机 Claude Code（claude -p），走 JSON 协议"}
    c = CANDIDATE_BY_MODEL.get(model)
    if c is not None:
        return {"tool_mode": "native", "timeout_s": c.timeout_s, "note": c.note,
                "tool_choice": c.tool_choice, "extra_body": dict(c.extra_body),
                "temperature": c.temperature}
    return {"tool_mode": "native", "timeout_s": 90.0, "note": ""}


def offered(env_keys: dict | None = None) -> list[str]:
    """The models a table may be built from.

    A retired one is not on the menu. Neither is a Chinese model whose provider
    has no key in the environment -- offering a seat that cannot be filled just
    moves the failure to the middle of a game.
    """
    out = [m for m, p in PROFILES.items() if not p.get("retired")]
    if env_keys is not None:
        # The relay's models need the relay's key, same rule as the Chinese ones below.
        out = [m for m in out if relay_key(m, env_keys)]
        out += [c.model for c in CANDIDATES
                if env_keys.get(PROVIDERS[c.provider].env_key) and not c.retired]
        # The local CLI needs no key, only the executable: offered when it can be found.
        if claude_cli.available(env_keys):
            out += list(claude_cli.MENU)
    return out


def relay_key(model: str, env_keys: dict) -> str:
    """The key a relay model is called with ('' when there is none): per vendor group, then the shared one."""
    group = "CLAUDE" if model.startswith("claude") else "GEMINI" if model.startswith("gemini") else "OPENAI"
    return env_keys.get(f"LLM_KEY_{group}_GROUP") or env_keys.get("LLM_API_KEY", "")


def retired(model: str) -> str | None:
    """Why this model is off the menu, or None if it is on it."""
    if model in PROFILES:
        return PROFILES[model].get("retired")
    c = CANDIDATE_BY_MODEL.get(model)
    return (c.retired or None) if c else None
