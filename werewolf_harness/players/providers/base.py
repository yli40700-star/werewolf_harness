"""Provider-agnostic model interface.

One interface, two tool-calling implementations (native function calling and a
JSON-prompt fallback), so the ReAct loop never learns which gateway or model it
is talking to.
"""

from __future__ import annotations

import json
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field


class ProviderError(Exception):
    """Anything the gateway or the transport got wrong. Carries a hint that is
    safe to show a user (see `explain` in `openai_compat`)."""

    def __init__(self, message: str, *, hint: str | None = None, status: int | None = None):
        super().__init__(message)
        self.hint = hint
        self.status = status


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = "call_0"
    raw_arguments: str = ""
    malformed: bool = False


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    finish_reason: str = "stop"
    model: str = ""
    raw: dict | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Cancelled(Exception):
    """The caller no longer wants this answer: its deadline fired, its game stopped,
    or the turn it was asked for is gone. Not a ProviderError on purpose -- nothing
    failed, so nothing may retry it or fall back to another model.

    ``produced`` is how much of the reply had been generated (estimated tokens)
    when the upstream call was cut; the seat uses it to estimate what was saved.
    """

    def __init__(self, reason: str = "cancelled", produced: int = 0):
        super().__init__(reason)
        self.reason = reason
        self.produced = produced


class CancelToken:
    """A thread-safe "stop" flag that can also pull a plug.

    Checking a flag between steps is not enough: a model call blocks for seconds
    inside a socket read or a subprocess. So a provider registers what would stop
    it *now* -- kill the `claude -p` process, shut the HTTP socket -- with
    ``on_cancel`` while it waits, and ``cancel()`` runs it from whichever thread
    cancels. A child token (one per model call) is cancelled with its parent but
    can also be cancelled alone, which is how a per-call hard deadline kills one
    call without ending the whole turn.
    """

    def __init__(self, parent: CancelToken | None = None):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._hooks: list[Callable[[], None]] = []
        self.reason: str | None = None
        if parent is not None:
            parent.on_cancel(lambda: self.cancel(parent.reason or "cancelled"))

    def is_set(self) -> bool:
        return self._event.is_set()

    def wait(self, seconds: float) -> bool:
        """Sleep up to `seconds`; True if cancelled meanwhile (an interruptible sleep)."""
        return self._event.wait(seconds)

    def cancel(self, reason: str = "cancelled") -> None:
        with self._lock:
            if self._event.is_set():
                return
            self.reason = reason
            self._event.set()
            hooks, self._hooks = self._hooks, []
        for fn in hooks:
            try:
                fn()
            except Exception:                              # noqa: BLE001 -- a plug that is already out
                pass

    def on_cancel(self, fn: Callable[[], None]) -> Callable[[], None]:
        """Run `fn` on cancel (at once if already cancelled). Returns an unregister function."""
        with self._lock:
            if not self._event.is_set():
                self._hooks.append(fn)
                return lambda: self._discard(fn)
        fn()
        return lambda: None

    def _discard(self, fn) -> None:
        with self._lock:
            if fn in self._hooks:
                self._hooks.remove(fn)

    def raise_if(self, produced: int = 0) -> None:
        if self._event.is_set():
            raise Cancelled(self.reason or "cancelled", produced)

    def child(self) -> CancelToken:
        return CancelToken(self)


@dataclass
class Delta:
    """One fragment of a reply as it is generated.

    kind  "text"   free text (the JSON tool protocol's whole reply arrives this way)
          "tool"   a fragment of a native tool call's arguments JSON; `name` is the
                   tool's name once known (the first fragment of a call carries it)
          "reset"  everything streamed so far is void -- the gateway is starting
                   over on another model, or a transport retry is resending
    """

    kind: str
    text: str = ""
    name: str | None = None
    index: int = 0


class LLMClient(ABC):
    """Every client returns tool calls the same way, whatever the wire format."""

    name: str = "unknown"
    tool_mode: str = "native"

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 800,
        timeout: float = 30.0,
    ) -> LLMResponse:
        ...

    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 800,
        timeout: float = 30.0,
        cancel: CancelToken | None = None,
    ) -> Iterator[Delta | LLMResponse]:
        """The same call, yielding the reply as it is generated.

        Yields Delta fragments, then exactly one LLMResponse -- the same object
        `chat()` would have returned, so a caller that ignores the fragments gets
        identical behaviour. This default has nothing to stream: it calls `chat()`
        and yields the whole reply as one fragment. A client that can do better
        (SSE, stream-json) overrides it and honours `cancel` mid-call.
        """
        if cancel is not None:
            cancel.raise_if()
        resp = self.chat(messages, tools=tools, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
        if cancel is not None:
            cancel.raise_if(resp.completion_tokens)
        yield from deltas_of(resp)
        yield resp


def deltas_of(resp: LLMResponse) -> Iterator[Delta]:
    """A finished reply as the fragments a streaming client would have sent.

    On the JSON tool protocol the call was parsed out of the text, so it is sent
    once, as text; a native call is sent as its arguments.
    """
    if resp.text:
        yield Delta("text", resp.text)
    for i, call in enumerate(resp.tool_calls):
        raw = call.raw_arguments or json.dumps(call.arguments, ensure_ascii=False)
        if resp.text and raw in resp.text:
            continue
        yield Delta("tool", raw, call.name, i)


def collect(stream: Iterator[Delta | LLMResponse], on_delta: Callable[[Delta], None] | None = None) -> LLMResponse:
    """Drive a chat_stream to its end: fragments to `on_delta`, the response returned."""
    final: LLMResponse | None = None
    for item in stream:
        if isinstance(item, LLMResponse):
            final = item
        elif on_delta is not None:
            on_delta(item)
    if final is None:
        raise ProviderError("stream ended without a response", status=502)
    return final


# ------------------------------------------------------- JSON fallback

JSON_TOOL_INSTRUCTIONS = """\
TOOL PROTOCOL (strict)

This model is driven without native function calling, so every step must be a
single JSON object and nothing else -- no prose before or after:

{{"thought": "<one sentence of reasoning>", "action": "<tool name>", "args": {{...}}}}

"thought" is REQUIRED on every step, never empty. When the tool you are calling
also takes a "thought" argument, fill that one too: say who you suspect, on what
evidence, and what this move is meant to achieve.

Available tools:
{tools}

Example:
{{"thought": "I should check what player 5 said last round.", "action": "query_history", "args": {{"player_id": 5, "round": 2}}}}

Example with a terminal action:
{{"thought": "7 dodged the vote on 3 twice; I want him out.", "action": "vote", "args": {{"target_id": 7, "thought": "7 dodged the vote on 3 twice -- that is the hardest read I have."}}}}
"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
# Some models answer the JSON protocol in their native tool-call markup instead:
# <invoke name="vote"><parameter name="target_id">3</parameter>...</invoke>
_INVOKE = re.compile(r'<invoke name="([^"]+)">(.*?)</invoke>', re.DOTALL)
_PARAM = re.compile(r'<parameter name="([^"]+)">(.*?)</parameter>', re.DOTALL)


def invoke_call(text: str) -> ToolCall | None:
    """The native-markup form of a call, or None. Not part of parse_json_action: recorded games
    replay with the parser they were played with, so the loop uses it only when the game's
    settings say so (settings['invoke_calls'])."""
    m = _INVOKE.search(text)
    if not m:
        return None
    args: dict = {}
    for k, v in _PARAM.findall(m.group(2)):
        v = v.strip()
        args[k] = int(v) if re.fullmatch(r"-?\d+", v) else v
    return ToolCall(name=m.group(1), arguments=args, raw_arguments=m.group(0))


def parse_json_action(text: str) -> ToolCall:
    """Recover a tool call from free text.

    Tolerant on the outside (code fences, stray prose) and strict on the
    inside: a malformed call is flagged rather than guessed at, so the loop can
    retry with the parse error attached instead of acting on a hallucination.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z]*\n?|```$", "", candidate).strip()
    match = _JSON_BLOCK.search(candidate)
    if not match:
        return ToolCall(name="", arguments={}, raw_arguments=text, malformed=True)
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        try:  # a single trailing comma is the most common failure
            data = json.loads(re.sub(r",\s*([}\]])", r"\1", match.group(0)))
        except json.JSONDecodeError:
            return ToolCall(name="", arguments={}, raw_arguments=text, malformed=True)
    if not isinstance(data, dict):
        return ToolCall(name="", arguments={}, raw_arguments=text, malformed=True)
    action = data.get("action") or data.get("tool") or data.get("name")
    args = data.get("args") or data.get("arguments") or {}
    return ToolCall(
        name=action if isinstance(action, str) else "",
        arguments=args if isinstance(args, dict) else {},
        raw_arguments=match.group(0),
        malformed=not isinstance(action, str) or not action,
    )


def describe_tools_for_prompt(tool_schemas: list[dict]) -> str:
    lines = []
    for schema in tool_schemas:
        fn = schema["function"]
        params = fn.get("parameters", {}).get("properties", {})
        required = set(fn.get("parameters", {}).get("required", []))
        args = ", ".join(
            f"{p}: {spec.get('type')}" + ("" if p in required else " (optional)")
            for p, spec in params.items()
        )
        lines.append(f"- {fn['name']}({args}) -- {fn.get('description', '')}")
    return "\n".join(lines)
