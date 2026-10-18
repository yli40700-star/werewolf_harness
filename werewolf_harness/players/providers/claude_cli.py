"""A seat driven by the local Claude Code CLI (`claude -p`) instead of an HTTP API.

Why: a person who has Claude Code signed in can fill seats from their own
subscription without an API key, or point the CLI at an API key, Bedrock or
Vertex -- whatever the CLI is configured for, the seat does not care.

How it maps onto the LLMClient contract:

  messages   the system message goes to --system-prompt (replacing Claude Code's
             default prompt), with the JSON tool protocol appended; the rest of
             the transcript is rendered as text and sent on stdin. Every call is
             stateless (--no-session-persistence), exactly like an HTTP call.
  tools      the CLI has no custom function calling, so tools are described in
             the prompt and the reply is parsed as one JSON object -- the same
             path the relay's claude models already take (tool_mode json_prompt).
             Claude Code's own tools are all switched off (--tools "").
  usage      read from the CLI's JSON result: input + cache writes + cache reads
             is the prompt, output is the completion.
  streaming  chat_stream runs --output-format stream-json --include-partial-messages
             --verbose and yields each text_delta as it is printed; cancelling the
             call kills the process (see StreamJsonParser for the lines it reads).
  failures   mapped onto the statuses the gateway already classifies: not
             installed / not logged in -> 401 (terminal), usage or rate limit ->
             429, timeout -> transport, anything else -> 502.

Not supported by the CLI and therefore ignored: temperature and max_tokens.

Model names: ``claude-cli`` (whatever the CLI defaults to) or
``claude-cli:<model>``, where <model> is anything ``claude --model`` accepts --
``sonnet``, ``opus``, ``haiku`` or a full model id.

Settings (read from the .env dict first, then the process environment):

  CLAUDE_CLI_BIN          path to the executable (default: ``claude`` on PATH)
  CLAUDE_CLI_CONCURRENCY  at most this many CLI processes at once (default 4;
                          each one is a Node process of a few hundred MB)
  CLAUDE_CLI_BARE         1 = add --bare: skip hooks, plugins and CLAUDE.md
                          discovery. Bare mode authenticates ONLY with
                          ANTHROPIC_API_KEY, never with a subscription login.
  CLAUDE_CLI_EXTRA_ARGS   extra flags appended verbatim, e.g. ``--effort low``
  CLAUDE_CLI_LEAN         1 (default) = start the CLI lean: no user/project settings files
                          (--setting-sources=) and no update check or telemetry
                          (CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC, DISABLE_AUTOUPDATER).
                          Measured on an M-series Mac, the time a call spends outside the
                          model fell from 1.58 s to 0.23 s; at ~150 calls a game that is
                          over three minutes. Login still comes from the keychain.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import time

from .base import (JSON_TOOL_INSTRUCTIONS, Cancelled, CancelToken, Delta, LLMClient, LLMResponse, ProviderError,
                   describe_tools_for_prompt, parse_json_action)

PREFIX = "claude-cli"
MENU = ("claude-cli:sonnet", "claude-cli:haiku", "claude-cli:opus")

_AUTH_HINTS = ("not logged in", "please run /login", "invalid api key", "authentication",
               "oauth token", "unauthorized", "credit balance is too low")
_LIMIT_HINTS = ("usage limit", "rate limit", "rate_limit", "limit reached", "overloaded",
                "too many requests", "429")

_slots_lock = threading.Lock()
_slots: dict[int, threading.BoundedSemaphore] = {}


def is_cli_model(model: str) -> bool:
    return model == PREFIX or model.startswith(PREFIX + ":")


def cli_model_arg(model: str) -> str | None:
    """`claude-cli:sonnet` -> `sonnet`; plain `claude-cli` -> None (the CLI's default)."""
    if ":" not in model:
        return None
    return model.split(":", 1)[1] or None


def _setting(name: str, env: dict | None, default: str = "") -> str:
    if env and env.get(name):
        return str(env[name])
    return os.environ.get(name, default)


def find_binary(env: dict | None = None) -> str | None:
    return shutil.which(_setting("CLAUDE_CLI_BIN", env, "claude"))


def available(env: dict | None = None) -> bool:
    return find_binary(env) is not None


def _slot(n: int) -> threading.BoundedSemaphore:
    with _slots_lock:
        if n not in _slots:
            _slots[n] = threading.BoundedSemaphore(n)
        return _slots[n]


def render_transcript(messages: list[dict]) -> str:
    """Everything after the system message, as one prompt.

    A single user message goes as it is. A longer exchange (the ReAct loop's
    earlier steps: the model's JSON replies and the observations that came
    back) is laid out in order, tagged by speaker, and ends by asking for the
    next step -- the CLI has no way to take prior assistant turns directly.
    """
    rest = [m for m in messages if m.get("role") != "system"]
    if len(rest) == 1 and rest[0].get("role") == "user":
        return str(rest[0].get("content") or "")
    parts = ["The conversation so far, oldest first. Lines you wrote are tagged [you]."]
    for m in rest:
        role = m.get("role")
        content = m.get("content")
        if content is None and m.get("tool_calls"):
            content = json.dumps([c.get("function") for c in m["tool_calls"]], ensure_ascii=False)
        tag = {"assistant": "[you]", "tool": "[tool result]"}.get(role, "[harness]")
        parts.append(f"{tag}\n{content or ''}")
    parts.append("Now reply with your next step: exactly one JSON object, nothing else.")
    return "\n\n".join(parts)


SHARED_END = "YOUR IDENTITY (private"     # where the shared part of the cache-layout system message ends


def system_prompt(messages: list[dict], tools: list[dict] | None) -> str:
    sys_parts = [str(m.get("content") or "") for m in messages if m.get("role") == "system"]
    if tools:
        sys_parts.append(JSON_TOOL_INSTRUCTIONS.format(tools=describe_tools_for_prompt(tools)))
    return "\n\n".join(p for p in sys_parts if p)


class ClaudeCLIClient(LLMClient):
    """One seat's view of `claude -p`. Thread-safe; holds no conversation state."""

    tool_mode = "json_prompt"

    def __init__(self, model: str = PREFIX, *, env: dict | None = None, binary: str | None = None,
                 concurrency: int | None = None, bare: bool | None = None,
                 extra_args: list[str] | None = None, cwd: str | None = None):
        self.name = model
        self.model = model
        self.cli_model = cli_model_arg(model)
        self.env = dict(env or {})
        self.binary = binary or _setting("CLAUDE_CLI_BIN", env, "claude")
        n = concurrency if concurrency is not None else int(_setting("CLAUDE_CLI_CONCURRENCY", env, "4") or 4)
        self.slots = _slot(max(1, n))
        self.bare = bare if bare is not None else _setting("CLAUDE_CLI_BARE", env) in ("1", "true", "yes")
        self.extra_args = list(extra_args) if extra_args is not None else shlex.split(_setting("CLAUDE_CLI_EXTRA_ARGS", env))
        # An empty working directory, so no project CLAUDE.md or settings leak into a seat.
        self.cwd = cwd or tempfile.mkdtemp(prefix="werewolf-claude-cli-")
        self.lean = _setting("CLAUDE_CLI_LEAN", env, "1") not in ("0", "false", "no")
        # CLAUDE_CLI_SPLIT_SYSTEM=1: only the part every seat shares goes to --system-prompt, the
        # seat's identity and this turn go on stdin. The CLI caches at the end of the system
        # prompt, so the shared rules become one cache entry for every seat and turn.
        self.split = _setting("CLAUDE_CLI_SPLIT_SYSTEM", env) in ("1", "true", "yes")

    # ------------------------------------------------------------ command line
    def prompts(self, messages: list[dict], tools: list[dict] | None) -> tuple[str, str]:
        """(system prompt, stdin) for one call."""
        system, prompt = system_prompt(messages, tools), render_transcript(messages)
        cut = system.find(SHARED_END) if self.split else -1
        if cut > 0:
            system, prompt = system[:cut].rstrip(), system[cut:] + "\n\n" + prompt
        return system, prompt

    def command(self, system: str, stream: bool = False) -> list[str]:
        # stream-json prints one event per line; --verbose is required with it in -p mode, and
        # --include-partial-messages adds the token-level deltas (content_block_delta).
        fmt = ["stream-json", "--verbose", "--include-partial-messages"] if stream else ["json"]
        cmd = [self.binary, "-p", "--output-format", *fmt, "--no-session-persistence",
               "--tools", "", "--strict-mcp-config", "--max-turns", "1", "--system-prompt", system]
        if self.cli_model:
            cmd += ["--model", self.cli_model]
        if self.bare:
            cmd.append("--bare")
        if self.lean:
            cmd.append("--setting-sources=")
        return cmd + self.extra_args

    def _env(self) -> dict:
        env = dict(os.environ)
        if self.lean:
            env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
            env.setdefault("DISABLE_AUTOUPDATER", "1")
        # A key put in .env reaches the CLI only when asked for: without --bare the CLI prefers
        # an ANTHROPIC_API_KEY over the subscription login, which would move the bill silently.
        if self.bare and self.env.get("ANTHROPIC_API_KEY"):
            env["ANTHROPIC_API_KEY"] = self.env["ANTHROPIC_API_KEY"]
        return env

    def _acquire(self, timeout: float, started: float) -> float:
        if not self.slots.acquire(timeout=timeout):
            raise ProviderError(f"{self.model}: all {self.slots._initial_value} CLI slots busy for {timeout:.0f}s",
                                status=429, hint="raise CLAUDE_CLI_CONCURRENCY or seat fewer claude-cli players")
        return max(1.0, timeout - (time.time() - started))

    # ------------------------------------------------------------ chat
    def chat(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0) -> LLMResponse:
        system, prompt = self.prompts(messages, tools)
        started = time.time()
        remaining = self._acquire(timeout, started)
        try:
            try:
                proc = subprocess.run(self.command(system), input=prompt, capture_output=True, text=True,
                                      timeout=remaining, cwd=self.cwd, env=self._env())
            except FileNotFoundError as exc:
                raise self._not_found() from exc
            except subprocess.TimeoutExpired as exc:
                raise self._timed_out(remaining) from exc
        finally:
            self.slots.release()
        data = self._parse(proc)
        return self._response(data, tools, int((time.time() - started) * 1000))

    def chat_stream(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0,
                    cancel: CancelToken | None = None):
        """`claude -p --output-format stream-json --include-partial-messages`, read line by line.

        Each text_delta is yielded as it is printed. Cancelling kills the process
        (and with it the request the CLI has open), so a turn whose deadline fired
        stops costing tokens at once. The final `result` line is read exactly as
        the one-shot JSON output is, so usage, errors and statuses are identical.
        """
        cancel = cancel or CancelToken()
        cancel.raise_if()
        system, prompt = self.prompts(messages, tools)
        started = time.time()
        remaining = self._acquire(timeout, started)
        proc = None
        try:
            try:
                proc = subprocess.Popen(self.command(system, stream=True), stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.cwd, env=self._env())
            except FileNotFoundError as exc:
                raise self._not_found() from exc
            err: list[bytes] = []
            threading.Thread(target=_feed_stdin, args=(proc, prompt), daemon=True).start()
            drain = threading.Thread(target=lambda: err.append(proc.stderr.read()), daemon=True)
            drain.start()
            timed_out = threading.Event()

            def deadline():
                timed_out.set()
                _kill(proc)
            timer = threading.Timer(remaining, deadline)
            timer.daemon = True
            timer.start()
            unhook = cancel.on_cancel(lambda: _kill(proc))
            parser = StreamJsonParser()
            out: list[str] = []
            try:
                for raw in proc.stdout:
                    for d in parser.feed(raw.decode("utf-8", errors="replace")):
                        out.append(d.text)
                        yield d
                proc.wait()
            finally:
                timer.cancel()
                unhook()
            drain.join(2.0)
            if cancel.is_set():
                from ..budget import estimate_tokens
                raise Cancelled(cancel.reason or "cancelled", estimate_tokens("".join(out)))
            if timed_out.is_set():
                raise self._timed_out(remaining)
            stdout = json.dumps(parser.result) if parser.result is not None else ""
            data = self._parse(subprocess.CompletedProcess(proc.args, proc.returncode, stdout,
                                                           b"".join(err).decode("utf-8", errors="replace")),
                               fallback_text=parser.text())
            for d in parser.tail():
                yield d                                    # a CLI that printed no partials: the whole text once
        finally:
            if proc is not None and proc.poll() is None:
                _kill(proc)                                # the consumer walked away: do not leave it running
            self.slots.release()
        yield self._response(data, tools, int((time.time() - started) * 1000))

    def _not_found(self) -> ProviderError:
        return ProviderError(f"claude CLI not found: {self.binary!r}", status=401,
                             hint="install Claude Code and sign in (`claude`), or set CLAUDE_CLI_BIN")

    def _timed_out(self, remaining: float) -> ProviderError:
        return ProviderError(f"{self.model}: claude -p timed out after {remaining:.0f}s",
                             hint="the CLI did not answer in time; retried on the next model")

    def _response(self, data: dict, tools, latency_ms: int) -> LLMResponse:
        usage = data.get("usage") or {}
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        cache_write = int(usage.get("cache_creation_input_tokens") or 0)
        prompt_tokens = int(usage.get("input_tokens") or 0) + cache_read + cache_write
        model_used = next(iter(data.get("modelUsage") or {}), "") or (self.cli_model or PREFIX)
        text = str(data.get("result") or "").strip()
        response = LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(usage.get("output_tokens") or 0),
            latency_ms=latency_ms,
            finish_reason=str(data.get("stop_reason") or "stop"),
            model=model_used,
            raw={"usage": {"prompt_tokens": prompt_tokens, "completion_tokens": int(usage.get("output_tokens") or 0),
                           "cache_read_input_tokens": cache_read, "cache_creation_input_tokens": cache_write},
                 "total_cost_usd": data.get("total_cost_usd"), "duration_ms": data.get("duration_ms"),
                 "session_id": data.get("session_id")},
        )
        if tools and text:
            response.tool_calls.append(parse_json_action(text))   # malformed calls are reported by the loop
        return response

    def _parse(self, proc: subprocess.CompletedProcess, fallback_text: str = "") -> dict:
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        data: dict = {}
        if out:
            try:
                data = json.loads(out.splitlines()[-1] if not out.startswith("{") else out)
            except json.JSONDecodeError:
                data = {}
        if not data and fallback_text and proc.returncode == 0:
            data = {"result": fallback_text}               # a stream that ended without its result line
        failed = proc.returncode != 0 or not data or data.get("is_error") or data.get("subtype") not in (None, "success")
        if not failed:
            return data
        detail = str(data.get("result") or err or out or f"exit code {proc.returncode}")[:500]
        low = detail.lower()
        status = data.get("api_error_status")
        if any(k in low for k in _AUTH_HINTS):
            raise ProviderError(f"{self.model}: {detail}", status=401,
                                hint="the CLI is not signed in: run `claude` once and log in, or set ANTHROPIC_API_KEY")
        if status == 429 or any(k in low for k in _LIMIT_HINTS):
            raise ProviderError(f"{self.model}: rate limit / usage limit: {detail}", status=429,
                                hint="the subscription or key hit its limit; seats move to the next model")
        raise ProviderError(f"{self.model}: {detail}", status=int(status) if isinstance(status, int) else 502)


class StreamJsonParser:
    """`claude -p --output-format stream-json` lines -> text fragments and the result.

    The lines this reads (Claude Code 2.x, with --verbose --include-partial-messages):

      {"type": "system", "subtype": "init", ...}                       ignored
      {"type": "stream_event", "event": {"type": "content_block_delta",
          "index": 0, "delta": {"type": "text_delta", "text": "..."}}}  -> a text fragment
      {"type": "stream_event", "event": {... "thinking_delta" ...}}     ignored: private reasoning
      {"type": "assistant", "message": {"content": [{"type": "text", ...}]}}
                                                                        the whole message, kept as a fallback
      {"type": "result", "subtype": "success", "result": "...", "usage": {...}}
                                                                        the same object --output-format json prints

    Tolerant on purpose: an unknown line type, a line that is not JSON (a warning
    on stdout) or an event without the fields above is skipped, and a CLI that
    prints no partial deltas still yields its reply -- once, from the assistant
    message, at the end (`tail()`).
    """

    def __init__(self):
        self.result: dict | None = None
        self._parts: list[str] = []
        self._whole: list[str] = []
        self._streamed = False

    def feed(self, line: str) -> list[Delta]:
        line = line.strip()
        if not line.startswith("{"):
            return []
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return []
        kind = obj.get("type")
        if kind == "stream_event":
            ev = obj.get("event") or {}
            if ev.get("type") == "content_block_delta":
                d = ev.get("delta") or {}
                if d.get("type") == "text_delta" and d.get("text"):
                    self._streamed = True
                    self._parts.append(d["text"])
                    return [Delta("text", d["text"])]
            return []
        if kind == "assistant":
            for block in (obj.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                    self._whole.append(block["text"])
        elif kind == "result":
            self.result = obj
        return []

    def text(self) -> str:
        return "".join(self._parts) if self._streamed else "".join(self._whole)

    def tail(self) -> list[Delta]:
        if self._streamed or not self._whole:
            return []
        return [Delta("text", "".join(self._whole))]


def _feed_stdin(proc: subprocess.Popen, prompt: str) -> None:
    try:
        proc.stdin.write(prompt.encode("utf-8"))
        proc.stdin.close()
    except (BrokenPipeError, OSError, ValueError):
        pass                                               # it died (or was killed) before reading it all


def _kill(proc: subprocess.Popen) -> None:
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass
