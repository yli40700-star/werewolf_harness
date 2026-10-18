"""A seat that calls the Anthropic Messages API directly, with explicit cache breakpoints.

Why: through `claude -p` the CLI decides where the cache breakpoints go, and the
real runs read back only ~8% of their prompts from the cache (docs/numbers.md).
Here the breakpoints are ours:

  shared   after the part of the system message every seat and turn shares
           (rules, guard preamble, output contract) -- only when the prompt uses
           the cache layout (ContextBuilder.cache_layout), where that part is a
           prefix; it ends right before "YOUR IDENTITY"
  system   at the end of the system message (identity, this turn's tools)
  turn     at the end of the user message, so the next ReAct step of the same
           turn reads the whole earlier conversation

The prompt is built exactly as for the CLI seat (system_prompt / render_transcript,
JSON tool protocol), so the only difference from those runs is who places the
breakpoints. Usage: input + cache writes + cache reads is the prompt.
"""
from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request

from .base import LLMClient, LLMResponse, ProviderError, parse_json_action
from .claude_cli import SHARED_END, render_transcript, system_prompt

URL = "https://api.anthropic.com/v1/messages"
EPHEMERAL = {"type": "ephemeral"}


def system_blocks(system: str, breakpoints: bool, shared: bool) -> list[dict]:
    """The system message as content blocks, a breakpoint after the shared part when asked."""
    cut = system.find(SHARED_END) if shared else -1
    parts = [system[:cut], system[cut:]] if cut > 0 else [system]
    blocks = [{"type": "text", "text": p} for p in parts]
    if breakpoints:
        for b in blocks:
            b["cache_control"] = EPHEMERAL
    return blocks


class AnthropicClient(LLMClient):
    tool_mode = "json_prompt"

    def __init__(self, model: str, api_key: str, *, breakpoints: bool = True, shared_breakpoint: bool = True,
                 retries: int = 6):
        if not api_key:
            raise ProviderError("no ANTHROPIC_API_KEY", status=401, hint="put ANTHROPIC_API_KEY in .env")
        self.name = self.model = model
        self.api_key = api_key
        self.breakpoints = breakpoints
        self.shared_breakpoint = shared_breakpoint
        self.retries = retries

    def payload(self, messages: list[dict], tools: list[dict] | None, temperature: float, max_tokens: int) -> dict:
        user = {"type": "text", "text": render_transcript(messages)}
        if self.breakpoints:
            user["cache_control"] = EPHEMERAL
        return {"model": self.model, "max_tokens": max_tokens,   # temperature: rejected by current models
                "system": system_blocks(system_prompt(messages, tools), self.breakpoints, self.shared_breakpoint),
                "messages": [{"role": "user", "content": [user]}]}

    def _post(self, payload: dict, timeout: float) -> dict:
        body = json.dumps(payload).encode()
        headers = {"content-type": "application/json", "x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(URL, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:600]
                if exc.code in (429, 500, 502, 503, 529) and attempt < self.retries:
                    time.sleep(min(30.0, float(exc.headers.get("retry-after") or 2 ** attempt)))
                    continue
                raise ProviderError(f"HTTP {exc.code} from Anthropic: {detail}", status=exc.code) from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, http.client.HTTPException) as exc:
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
                    continue
                raise ProviderError(f"cannot reach Anthropic: {exc}") from exc
        raise AssertionError("unreachable")

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=60.0) -> LLMResponse:
        started = time.time()
        data = self._post(self.payload(messages, tools, temperature, max_tokens), timeout)
        usage = data.get("usage") or {}
        read = int(usage.get("cache_read_input_tokens") or 0)
        write = int(usage.get("cache_creation_input_tokens") or 0)
        prompt = int(usage.get("input_tokens") or 0) + read + write
        text = "".join(b.get("text", "") for b in data.get("content") or [] if b.get("type") == "text").strip()
        resp = LLMResponse(
            text=text, prompt_tokens=prompt, completion_tokens=int(usage.get("output_tokens") or 0),
            latency_ms=int((time.time() - started) * 1000), finish_reason=str(data.get("stop_reason") or "stop"),
            model=str(data.get("model") or self.model),
            raw={"usage": {"prompt_tokens": prompt, "completion_tokens": int(usage.get("output_tokens") or 0),
                           "cache_read_input_tokens": read, "cache_creation_input_tokens": write}},
        )
        if tools and text:
            resp.tool_calls.append(parse_json_action(text))
        return resp
