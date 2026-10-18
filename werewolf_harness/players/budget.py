"""Local token accounting and a per-game spend cap.

The gateway's own numbers cannot be trusted: one relay billed 6921 prompt
tokens for a ten-word system message and 1273 for a 300-word one, and the same
request reported 5334 tokens twice and 289 once. So the harness counts what it
*sent*, from the strings it built, and treats the gateway's figure as a second
opinion worth logging and nothing more.

The estimate is deliberately crude and deliberately conservative -- it exists
to stop a runaway loop from spending a wallet, not to reconcile an invoice.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field

_CJK = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")

# Latin text runs ~4 characters per token; CJK runs closer to one token per
# character (often worse). Counting them apart is the whole trick -- a Chinese
# prompt measured at 4 chars/token reads as a quarter of its real size.
CHARS_PER_TOKEN_LATIN = 4.0
CHARS_PER_TOKEN_CJK = 1.1


def estimate_tokens(text: str) -> int:
    """A cheap upper-ish bound on the tokens in one string."""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return int(cjk / CHARS_PER_TOKEN_CJK + other / CHARS_PER_TOKEN_LATIN) + 1


def estimate_messages(messages: list[dict], tools: list[dict] | None = None) -> int:
    """Tokens for a whole request: every message part, plus the tool schemas.

    Tool schemas are counted because they are re-sent on every step and are the
    largest fixed cost in a tool-heavy turn -- 13 tools is ~4.5 KB of JSON per
    call, and a turn makes up to eight calls.
    """
    total = 0
    for m in messages or []:
        total += estimate_tokens(m.get("content") or "")
        for call in m.get("tool_calls") or []:
            fn = call.get("function", {})
            total += estimate_tokens(fn.get("name", "")) + estimate_tokens(fn.get("arguments", ""))
        total += 4                       # role and framing overhead
    if tools:
        total += estimate_tokens(json.dumps(tools, ensure_ascii=False))
    return total


class BudgetExhausted(RuntimeError):
    """The run has spent its allowance. Raised through the turn loop, never caught here.

    Same reasoning as GatewayExhausted: a run that quietly keeps going after its
    cap is a run whose cost is decided by a bug rather than by a number someone
    chose.
    """


@dataclass
class TokenBudget:
    """One game's allowance, shared by every seat.

    Shared rather than per-seat for the same reason the transport budget is:
    the thing being protected is the game (and the wallet), not the seat.
    """

    max_tokens: int = 2_000_000
    max_usd: float | None = None
    price_per_1m: tuple[float, float] = (0.0, 0.0)   # (prompt, completion) USD per 1M tokens

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    gateway_prompt_tokens: int = 0                   # what the relay claimed, for comparison only
    gateway_completion_tokens: int = 0
    # seats decide in parallel (votes, the review, independent night roles), so the
    # counters are shared across threads
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def usd(self) -> float:
        p, c = self.price_per_1m
        return self.prompt_tokens / 1e6 * p + self.completion_tokens / 1e6 * c

    def spend(self, *, prompt: int, completion: int,
              gateway_prompt: int = 0, gateway_completion: int = 0) -> None:
        with self._lock:
            self._spend(prompt, completion, gateway_prompt, gateway_completion)

    def _spend(self, prompt: int, completion: int, gateway_prompt: int, gateway_completion: int) -> None:
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.gateway_prompt_tokens += gateway_prompt
        self.gateway_completion_tokens += gateway_completion
        if self.total_tokens > self.max_tokens:
            raise BudgetExhausted(
                f"本局用量 {self.total_tokens} tokens（自算）超过上限 {self.max_tokens}，"
                f"共 {self.calls} 次调用。这一局停在这里，不入库。")
        if self.max_usd is not None and self.usd > self.max_usd:
            raise BudgetExhausted(
                f"本局花费约 ${self.usd:.2f} 超过上限 ${self.max_usd:.2f}，"
                f"共 {self.calls} 次调用。这一局停在这里，不入库。")

    def drift(self) -> float | None:
        """Gateway prompt tokens divided by ours -- 1.0 means they agree.

        Worth logging every run: the 40x figure that invalidated an earlier
        batch would have shown up here on the first call.
        """
        if not self.prompt_tokens or not self.gateway_prompt_tokens:
            return None
        return round(self.gateway_prompt_tokens / self.prompt_tokens, 2)

    def to_dict(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "total_tokens": self.total_tokens,
                "usd": round(self.usd, 4), "max_tokens": self.max_tokens, "max_usd": self.max_usd,
                "gateway_prompt_tokens": self.gateway_prompt_tokens,
                "gateway_completion_tokens": self.gateway_completion_tokens,
                "gateway_drift": self.drift()}
