"""Offline scripted client. Exercises the ReAct loop with no key and no network.

It reads the tool schemas it is handed (the same ones a real model gets) and
picks a legal call: sometimes an information tool, then a terminal. Terminals
are recognised by the "ENDS YOUR TURN" marker every environment terminal
carries in its description. Runs labelled ``model="mock"`` are never findings.

Streaming (for the draft protocol and its benchmark): with ``tokens_per_s`` set
it also behaves like a real model in *time* -- the first fragment after
``ttft_s``, then the reply at that rate in chunks of ``chunk_tokens``; `chat()`
sleeps the whole generation, `chat_stream()` yields it as it goes and stops the
moment it is cancelled. ``speech_tokens`` makes every speech that long (a real
speech is ~400 tokens; the default mock line is ten words). The decision itself
-- which tool, which target -- does not depend on these knobs.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time

from .base import Cancelled, CancelToken, Delta, LLMClient, LLMResponse, ProviderError, ToolCall

_SELF = re.compile(r"You are player (\d+), role: (\w+)")
_ROUND = re.compile(r"=== (?:ROUND|NIGHT) (\d+)")
_WOLVES = re.compile(r"Your fellow wolves: \[([\d, ]*)\]")
_TASK = re.compile(r"YOUR TASK \[(\w+)\]")

# Named mock seats for the online service. "mock-stream" times like a real model
# (1 s to the first token, 45 tokens/s, 400-token speeches); "mock-stream-fast" is
# the same shape at test speed.
STREAM_PROFILES: dict[str, dict] = {
    "mock-stream": {"ttft_s": 1.0, "tokens_per_s": 45.0, "chunk_tokens": 4, "speech_tokens": 400},
    "mock-stream-fast": {"ttft_s": 0.02, "tokens_per_s": 3000.0, "chunk_tokens": 6, "speech_tokens": 60},
}

_FILLER = ("我先说一下我的看法。", "昨晚的情况大家都看到了，", "这一轮的发言里有几处前后对不上，",
           "我会把票投给我最怀疑的那个位置，", "希望大家把逻辑讲清楚，", "不要只跟着别人的节奏走。",
           "我手里没有查验，只能从发言和票型里找线索，", "如果我是狼，我不会这样暴露自己。")


def _weight(ch: str) -> float:
    """Tokens one character costs, by the same rule as players.budget (CJK ~1.1 chars/token, else ~4)."""
    return 1 / 1.1 if "　" <= ch <= "鿿" or "＀" <= ch <= "￯" else 0.25


def tokens_of(text: str) -> float:
    return sum(_weight(c) for c in text)


def chunk_by_tokens(text: str, tokens: float) -> list[str]:
    """`text` cut into pieces of about `tokens` tokens each."""
    out, buf, w = [], [], 0.0
    for ch in text:
        buf.append(ch)
        w += _weight(ch)
        if w >= tokens:
            out.append("".join(buf))
            buf, w = [], 0.0
    if buf:
        out.append("".join(buf))
    return out


class MockClient(LLMClient):
    name = "mock"
    tool_mode = "native"

    def __init__(self, seed: int = 0, info_rate: float = 0.5, latency_ms: int = 2, sleep_ms: int = 0, *, ttft_s: float = 0.0,
                 tokens_per_s: float = 0.0, chunk_tokens: float = 4.0, speech_tokens: int = 0,
                 name: str | None = None, sleep=time.sleep):
        self.seed = seed
        self.info_rate = info_rate
        self.latency_ms = latency_ms
        self.sleep_ms = sleep_ms          # really wait this long per call: a stand-in for a model's latency
        self.ttft_s, self.tokens_per_s, self.chunk_tokens = ttft_s, tokens_per_s, chunk_tokens
        self.speech_tokens = speech_tokens
        if name:
            self.name = name
        self._sleep = sleep

    @classmethod
    def named(cls, model: str, seed: int) -> MockClient:
        """`mock` or one of STREAM_PROFILES."""
        return cls(seed=seed, name=model, **STREAM_PROFILES.get(model, {}))

    # ------------------------------------------------------------ timing
    def _pieces(self, resp: LLMResponse) -> list[Delta]:
        if resp.tool_calls:
            call = resp.tool_calls[0]
            return [Delta("tool", t, call.name if i == 0 else None)
                    for i, t in enumerate(chunk_by_tokens(call.raw_arguments, self.chunk_tokens))]
        return [Delta("text", t) for t in chunk_by_tokens(resp.text, self.chunk_tokens)]

    def _gen_s(self, text: str) -> float:
        return tokens_of(text) / self.tokens_per_s if self.tokens_per_s else 0.0

    def chat_stream(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0,
                    cancel: CancelToken | None = None):
        cancel = cancel or CancelToken()
        cancel.raise_if()
        if self.sleep_ms and cancel.wait(self.sleep_ms / 1000):
            raise Cancelled(cancel.reason or "cancelled", 0)
        resp = self._decide(messages, tools)
        started = time.monotonic()
        if self.ttft_s and cancel.wait(self.ttft_s):
            raise Cancelled(cancel.reason or "cancelled", 0)
        produced = 0.0
        for d in self._pieces(resp):
            gap = self._gen_s(d.text)
            if gap and cancel.wait(gap):
                raise Cancelled(cancel.reason or "cancelled", int(produced))
            cancel.raise_if(int(produced))
            produced += tokens_of(d.text)
            yield d
        if self.tokens_per_s:
            resp.latency_ms = int((time.monotonic() - started) * 1000)
        yield resp

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0) -> LLMResponse:
        if self.sleep_ms:
            time.sleep(self.sleep_ms / 1000)
        resp = self._decide(messages, tools)
        if self.tokens_per_s:
            body = resp.tool_calls[0].raw_arguments if resp.tool_calls else resp.text
            total = self.ttft_s + self._gen_s(body)
            self._sleep(total)
            resp.latency_ms = int(total * 1000)
        return resp

    # ------------------------------------------------------------ the decision
    def _decide(self, messages, tools) -> LLMResponse:
        system = messages[0]["content"] if messages else ""
        prompt = "\n".join(m.get("content") or "" for m in messages)
        me = int(_SELF.search(system).group(1)) if _SELF.search(system) else 0
        rnd = int(_ROUND.search(prompt).group(1)) if _ROUND.search(prompt) else 1
        step = sum(1 for m in messages if m.get("role") == "assistant")
        rng = random.Random(f"{self.seed}:{me}:{rnd}:{step}")
        tools = tools or []
        terminals = [t for t in tools if "ENDS YOUR TURN" in t["function"].get("description", "")]
        infos = [t for t in tools if t not in terminals]
        forced = "out of steps" in (messages[-1].get("content") or "")
        if infos and not forced and step < 2 and rng.random() < self.info_rate:
            pick = rng.choice(infos)
        elif terminals:
            pick = rng.choice(terminals)
        else:
            return LLMResponse(text="I have nothing to add.", latency_ms=self.latency_ms, model=self.name)
        fn = pick["function"]
        args = self._args(fn, rng, me, system, rnd)
        if self.speech_tokens and isinstance(args.get("content"), str):
            args["content"] = self._speech(f"{self.seed}:{me}:{rnd}:{step}:{fn['name']}", me)
        raw = json.dumps(args, ensure_ascii=False) if self.speech_tokens else json.dumps(args)
        call = ToolCall(name=fn["name"], arguments=args, id=f"call_{step}", raw_arguments=raw)
        out = 20 if not self.speech_tokens else max(20, int(tokens_of(raw)))
        return LLMResponse(text="", tool_calls=[call], prompt_tokens=len(prompt) // 4, completion_tokens=out,
                           latency_ms=self.latency_ms, model=self.name)

    def _speech(self, key: str, me: int) -> str:
        """A speech of about `speech_tokens` tokens, the same one for the same seat and step."""
        rng = random.Random(key)
        parts, w = [f"我是{me}号。"], 3.0
        while w < self.speech_tokens:
            line = rng.choice(_FILLER)
            parts.append(line)
            w += tokens_of(line)
        return "".join(parts)

    def _args(self, fn: dict, rng: random.Random, me: int, system: str, rnd: int) -> dict:
        props = fn.get("parameters", {}).get("properties", {})
        required = set(fn.get("parameters", {}).get("required", []))
        out: dict = {}
        for k, p in props.items():
            if k not in required and rng.random() < 0.3:
                continue
            if "enum" in p:
                if not p["enum"]:
                    continue
                out[k] = rng.choice(p["enum"])
            elif p["type"] == "integer":
                out[k] = max(1, rnd - 1) if k == "round" else rng.randint(1, 12)
            elif p["type"] == "number":
                out[k] = round(rng.uniform(p.get("minimum", 0.0), p.get("maximum", 1.0)), 2)
            elif p["type"] == "string":
                out[k] = f"player {me} says something about {fn['name']} in round {rnd}"
            elif p["type"] == "array":
                out[k] = []
            elif p["type"] == "object":
                m = _WOLVES.search(system)
                wolves = [me] + ([int(x) for x in m.group(1).split(",") if x.strip()] if m else [])
                out[k] = {str(w): "shenshui" for w in wolves}
        return out


class ChaosMockClient(MockClient):
    """A mock that behaves like a real model in the ways that make games hard to reproduce.

    Its choices come from an unseeded generator, so two runs of the same seed play
    different games (as a model at temperature > 0 does); it sleeps a random time,
    so seats finish in a different order every run; and it sometimes fails --
    a retryable 503, a malformed reply, an empty reply. Only a recording can
    replay a game played against it, which is what it exists to test.
    """

    name = "mock-chaos"

    def __init__(self, seed: int = 0, error_rate: float = 0.08, malformed_rate: float = 0.08,
                 max_sleep_ms: int = 8, rng: random.Random | None = None):
        super().__init__(seed=seed)
        self.name = "mock-chaos"
        self.error_rate = error_rate
        self.malformed_rate = malformed_rate
        self.max_sleep_ms = max_sleep_ms
        self._rng = rng or random.Random()             # deliberately not derived from the game seed
        self._lock = threading.Lock()

    # Streams by chunking the whole reply (the base default): the timed stream of MockClient
    # would bypass the dice in chat() -- no errors, no malformed replies, a seeded decision.
    chat_stream = LLMClient.chat_stream

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0) -> LLMResponse:
        with self._lock:
            roll, nap, salt = self._rng.random(), self._rng.random(), self._rng.randrange(1 << 30)
        time.sleep(nap * self.max_sleep_ms / 1000)
        if roll < self.error_rate:
            raise ProviderError("mock-chaos: upstream 503", status=503, hint="模拟的上游故障")
        if roll < self.error_rate + self.malformed_rate:
            return LLMResponse(text="嗯……我想想", latency_ms=int(nap * self.max_sleep_ms), model=self.name)
        # a fresh draw every call: the base mock keys its choices on its seed
        r = MockClient(seed=salt, info_rate=self.info_rate).chat(messages, tools, temperature, max_tokens, timeout)
        r.latency_ms = int(nap * self.max_sleep_ms)
        r.model = self.name
        if r.tool_calls and tools is None:
            r.tool_calls = []
        if not r.tool_calls and tools is None:
            r.text = f"记一下：第 {salt % 7} 号可疑。"
        return r


class FlakyMockClient(MockClient):
    """A seeded mock that is worse at the job: the "cheap model" of the cascade experiments.

    Unlike ChaosMockClient it is deterministic -- every choice is drawn from
    (seed, the request) -- so two arms of a paired experiment that reach the same
    prompt get the same cheap-model reply, and a game played twice is the same
    game. With probability ``bad_rate`` per call it makes one of the mistakes a
    small model makes, in equal shares:

      malformed   prose, no tool call
      unknown     a tool that does not exist
      illegal     a real terminal with an argument outside its enum (seat 99)
      vague       (speech only) a speech that names no seat
      cite        (speech only) "N号说过……" about a seat it never looked up

    The rest of the time it is the plain mock. Its ``name`` is what prices and
    recordings key on, so the two tiers of a cascade must not share one.
    """

    name = "mock-flaky"
    KINDS = ("malformed", "unknown", "illegal", "vague", "cite")

    def __init__(self, seed: int = 0, bad_rate: float = 0.25, name: str | None = None, **kw):
        super().__init__(seed=seed, name=name or "mock-flaky", **kw)
        self.bad_rate = bad_rate

    # Streams by chunking the whole reply (the base default), as ChaosMockClient does: the
    # timed stream of MockClient calls _decide() directly and would skip the mistakes in
    # chat(), so a streaming seat (every online speech) would never make one to escalate on.
    chat_stream = LLMClient.chat_stream

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0) -> LLMResponse:
        digest = hashlib.sha256(json.dumps([self.seed, messages, [t["function"]["name"] for t in tools or []]],
                                           ensure_ascii=False, default=str).encode()).hexdigest()
        rng = random.Random(digest)
        resp = super().chat(messages, tools, temperature, max_tokens, timeout)
        if not tools or rng.random() >= self.bad_rate:
            return resp
        call = resp.tool_calls[0] if resp.tool_calls else None
        speech = call is not None and isinstance(call.arguments.get("content"), str)
        kind = rng.choice([k for k in self.KINDS if speech or k not in ("vague", "cite")])
        if kind == "malformed" or call is None:
            return LLMResponse(text="嗯……我想想再说。", latency_ms=self.latency_ms, model=self.name,
                               prompt_tokens=resp.prompt_tokens, completion_tokens=8)
        args, name = dict(call.arguments), call.name
        if kind == "unknown":
            name = "reveal_all_roles"
        elif kind == "illegal":
            fn = next(t["function"] for t in tools if t["function"]["name"] == call.name)
            enum_keys = [k for k, p in fn.get("parameters", {}).get("properties", {}).items() if p.get("enum")]
            args = {**args, enum_keys[0]: 99} if enum_keys else {"bogus": True}
        elif kind == "vague":
            args["content"] = "我没有什么特别的看法，大家继续发言吧，我听听再说。"
        else:
            m = _SELF.search(messages[0]["content"] if messages else "")
            me = int(m.group(1)) if m else 0
            other = rng.choice([s for s in range(1, 9) if s != me])
            args["content"] = f"{other}号说过他是预言家，还说“我昨晚查验了三号是好人”，我觉得他在骗人。"
        raw = json.dumps(args, ensure_ascii=False)
        bad = ToolCall(name=name, arguments=args, id=call.id, raw_arguments=raw)
        return LLMResponse(text="", tool_calls=[bad], prompt_tokens=resp.prompt_tokens,
                           completion_tokens=resp.completion_tokens, latency_ms=self.latency_ms, model=self.name)


def cascade_mock(model: str, seed: int) -> MockClient | None:
    """The two offline tiers of a cascade, by name (None for any other model):

        mock-strong            the plain mock under its own name (reports 6 ms a call)
        mock-flaky[:bad_rate]  FlakyMockClient, default bad_rate 0.25 (reports 3 ms a call)

    The reported latencies only make the latency columns of a mock experiment
    distinguishable; nothing sleeps."""
    if model == "mock-strong":
        return MockClient(seed=seed, name="mock-strong", latency_ms=6)
    if model.startswith("mock-flaky"):
        rate = float(model.split(":", 1)[1]) if ":" in model else 0.25
        return FlakyMockClient(seed=seed, bad_rate=rate, name=model, latency_ms=3)
    return None
