"""A simulated provider, for load tests, fault drills and cache measurements.

It plays like the offline MockClient (same legal choices) and adds what a real
provider has and the mock does not:

  latency       log-normal around `median_ms` (model latency is long-tailed);
                injectable at run time through a LatencyControl (slow mode)
  faults        a fraction of calls time out, return 429, or return 5xx
  prefix cache  block-hash prefix caching, the rule OpenAI/DeepSeek/vLLM use:
                the prompt is cut into fixed blocks (128 tokens); a block counts as
                cached if the chain hash of it and every block before it was seen
                recently; nothing is cached under 1024 tokens

Numbers produced with it are labelled "simulated" wherever they are reported.
The cache rule is the providers' published one; the traffic is the real prompt
the harness builds, so the *relative* effect of a prompt layout is measurable
offline, before paying for it.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
from collections import OrderedDict

from ...players.providers.base import Cancelled, CancelToken, LLMClient, LLMResponse, ProviderError
from ...players.providers.mock import MockClient

CHARS_PER_TOKEN = 4
BLOCK_TOKENS = 128
MIN_CACHE_TOKENS = 1024


def serialise(messages: list[dict], tools: list[dict] | None, tools_first: bool = True) -> str:
    """The byte order a provider hashes: tools, then messages in order."""
    parts = []
    if tools and tools_first:
        parts.append(json.dumps(tools, sort_keys=True, ensure_ascii=False))
    for m in messages:
        parts.append(f"{m.get('role')}:{m.get('content') or ''}")
        for c in m.get("tool_calls") or []:
            parts.append(json.dumps(c, sort_keys=True, ensure_ascii=False))
    if tools and not tools_first:
        parts.append(json.dumps(tools, sort_keys=True, ensure_ascii=False))
    return "\n".join(parts)


class PrefixCache:
    def __init__(self, ttl_s: float = 300.0, capacity: int = 200_000):
        self.ttl_s = ttl_s
        self.capacity = capacity
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def lookup_and_fill(self, text: str, now: float | None = None) -> tuple[int, int]:
        """(prompt_tokens, cached_tokens) for this prompt; records its blocks."""
        now = now if now is not None else time.monotonic()
        block = BLOCK_TOKENS * CHARS_PER_TOKEN
        total_tokens = math.ceil(len(text) / CHARS_PER_TOKEN)
        chain = hashlib.sha256()
        cached_blocks, still_hitting = 0, True
        keys = []
        for i in range(0, len(text) - len(text) % block, block):
            chain.update(text[i:i + block].encode())
            keys.append(chain.hexdigest())
        with self._lock:
            for k in keys:
                t = self._seen.get(k)
                if still_hitting and t is not None and now - t <= self.ttl_s:
                    cached_blocks += 1
                else:
                    still_hitting = False
                self._seen[k] = now
                self._seen.move_to_end(k)
            while len(self._seen) > self.capacity:
                self._seen.popitem(last=False)
        cached = cached_blocks * BLOCK_TOKENS if total_tokens >= MIN_CACHE_TOKENS else 0
        return total_tokens, min(cached, total_tokens)


class LatencyControl:
    """Inject a provider slowdown at run time: every SimClient that shares this
    object scales its latency by `scale`, or uses `median_ms` instead of its own
    profile when that is set. The overload benchmark flips it mid-run
    (0.3 s -> 3 s per call) to play a provider that suddenly slows down.

    The process-wide default is CONTROL; SIM_LATENCY_SCALE / SIM_MEDIAN_MS in the
    environment set its starting values, so a worker process can be started slow.
    """

    def __init__(self, scale: float = 1.0, median_ms: float | None = None):
        self.scale = scale
        self.median_ms = median_ms

    def set(self, *, scale: float | None = None, median_ms: float | None = None) -> None:
        if scale is not None:
            self.scale = scale
        self.median_ms = median_ms

    def apply(self, own_median_ms: float) -> float:
        return (self.median_ms if self.median_ms is not None else own_median_ms) * self.scale

    @classmethod
    def from_env(cls) -> LatencyControl:
        import os
        m = os.environ.get("SIM_MEDIAN_MS")
        return cls(float(os.environ.get("SIM_LATENCY_SCALE") or 1.0), float(m) if m else None)


CONTROL = LatencyControl.from_env()


class SimClient(LLMClient):
    tool_mode = "native"

    def __init__(self, seed: int = 0, *, name: str = "sim", median_ms: float = 0.0, sigma: float = 0.6,
                 timeout_rate: float = 0.0, error_rate: float = 0.0, rate_limit_rate: float = 0.0,
                 cache: PrefixCache | None = None, tools_first: bool = True, sleep: bool = True,
                 control: LatencyControl | None = None, tokens_per_s: float = 0.0, speech_tokens: int = 0,
                 chunk_tokens: float = 4.0):
        self.name = name
        # The sampled latency is the time to the first token; with tokens_per_s the reply
        # then takes as long to generate as a real one would (and streams at that rate).
        self.tokens_per_s = tokens_per_s if sleep else 0.0
        self.inner = MockClient(seed=seed, tokens_per_s=self.tokens_per_s, speech_tokens=speech_tokens,
                                chunk_tokens=chunk_tokens)
        self.rng = random.Random(seed * 7919 + 1)
        self.median_ms, self.sigma = median_ms, sigma
        self.timeout_rate, self.error_rate, self.rate_limit_rate = timeout_rate, error_rate, rate_limit_rate
        self.cache = cache
        self.tools_first = tools_first
        self.sleep = sleep
        self.control = control if control is not None else CONTROL

    def _fault(self, timeout: float) -> float:
        """Roll the dice for this call: raise its fault, or return its latency in ms."""
        r = self.rng.random()
        if r < self.timeout_rate:
            if self.sleep:
                time.sleep(min(timeout, 0.05))
            raise ProviderError(f"{self.name}: timed out", status=None)
        if r < self.timeout_rate + self.rate_limit_rate:
            raise ProviderError(f"{self.name}: 429 rate limited", status=429)
        if r < self.timeout_rate + self.rate_limit_rate + self.error_rate:
            raise ProviderError(f"{self.name}: 503 upstream error", status=503)
        median = self.control.apply(self.median_ms)
        return median * math.exp(self.rng.gauss(0, self.sigma)) if median else 0.0

    def _account(self, resp: LLMResponse, messages, tools, ms: float) -> LLMResponse:
        prompt, cached = (self.cache.lookup_and_fill(serialise(messages, tools, self.tools_first))
                          if self.cache else (resp.prompt_tokens, 0))
        resp.prompt_tokens = prompt
        resp.latency_ms = int(ms) + (resp.latency_ms if self.tokens_per_s else 0)
        resp.model = self.name
        resp.raw = {"usage": {"prompt_tokens": prompt, "completion_tokens": resp.completion_tokens,
                              "prompt_tokens_details": {"cached_tokens": cached}}}
        return resp

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0) -> LLMResponse:
        ms = self._fault(timeout)
        if self.sleep and ms:
            time.sleep(ms / 1000)
        resp = self.inner.chat(messages, tools, temperature, max_tokens, timeout)
        return self._account(resp, messages, tools, ms)

    def chat_stream(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0,
                    cancel: CancelToken | None = None):
        """Same faults and latency as chat(); the sampled latency is the wait for the first token."""
        cancel = cancel or CancelToken()
        cancel.raise_if()
        ms = self._fault(timeout)
        if self.sleep and ms and cancel.wait(ms / 1000):
            raise Cancelled(cancel.reason or "cancelled", 0)
        for item in self.inner.chat_stream(messages, tools, temperature, max_tokens, timeout, cancel=cancel):
            yield self._account(item, messages, tools, ms) if isinstance(item, LLMResponse) else item
