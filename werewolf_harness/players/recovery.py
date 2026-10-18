"""Failure handling.

The point is that one agent failing must never take the game down: a crashed
game is a discarded sample, and discarded samples are how a batch quietly turns
into a biased batch. Every failure mode has a defined fallback, and every
fallback is counted, because "how often did the model need saving" is itself
one of the reported numbers (axis 4).

    timeout / transport error   -> retry with backoff, then abstain for the turn
    malformed output            -> retry with the parse error attached (<=3)
    unknown tool                -> retry with the whitelist attached
    semantically invalid action -> retry with the engine's reason attached
    ReAct loop not terminating  -> force a terminal action
    repeated identical calls    -> break the loop, force a terminal action
"""

from __future__ import annotations

import concurrent.futures
import os
import contextvars
import threading
import time
from dataclasses import dataclass, field

from .providers.base import Cancelled, CancelToken, Delta, ProviderError, collect
from .providers.recorded import ReplayDivergence

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

# A refusal that waiting cannot fix. The gateway returns 403 for two unrelated
# things -- a rate limit and an empty wallet -- and only the body tells them
# apart, so the body is what is read. One batch took ninety of these, filled
# ninety turns with the harness's own default, finished all five games and
# saved them as results.
TERMINAL_BODY = ("insufficient account balance", "insufficient balance",
                 "billing_error", "quota", "exceeded your current quota",
                 "invalid api key", "incorrect api key")


def is_terminal_refusal(exc: "ProviderError") -> bool:
    """Whether re-sending this request could ever succeed.

    401 and 402 mean the same thing everywhere -- no key, no money -- and are
    read from the status alone. 403 is NOT on that list on purpose: relays use
    it for "not right now" (a quota, a concurrency cap, a risk rule) as well as
    for "no money", and killing a game on the first ambiguous 403 is worse than
    spending one more call finding out. It qualifies only when the body says so.
    """
    if exc.status in (401, 402):
        return True
    return any(k in str(exc).lower() for k in TERMINAL_BODY)


class GatewayExhausted(RuntimeError):
    """The gateway will not serve this run any more -- money, key or quota.

    Raised through the turn loop rather than handled, because the alternative
    is what already happened: a game that keeps playing with the harness making
    every decision, and a log that is indistinguishable from a real one.
    """


@dataclass
class RecoveryPolicy:
    max_retries: int = 3
    max_transport_retries: int = 2
    timeout_s: float = 30.0
    backoff_s: float = 2.0
    max_react_steps: int = 8
    loop_repeat_threshold: int = 3
    # How many turns the model may fail to answer before the game is abandoned.
    # Not zero: a gateway drops the odd call and a game that dies on one of
    # them wastes the twenty minutes already spent. Not unbounded either -- at
    # 39% of turns the log stops being a record of what the models did. Ten is
    # under 10% of a typical game's turn count.
    max_transport_failures: int = 10
    # A refusal that would end the game is checked twice before it is believed.
    confirm_terminal: bool = True
    confirm_after_s: float = 2.0
    sleep = staticmethod(time.sleep)


@dataclass
class TransportBudget:
    """How many unanswered turns one game may accumulate before it is abandoned.

    Shared by every seat's loop, because the thing being protected is the game,
    not the seat. A per-seat counter would let twelve seats fail nine times
    each and call it healthy.
    """

    limit: int = 10
    failures: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def spend(self) -> None:
        with self._lock:
            self.failures += 1
            over = self.failures > self.limit
        if over:
            raise GatewayExhausted(
                f"模型连续答不上来：{self.failures} 个回合被兜底，超过上限 {self.limit}。"
                "这一局作废，不入库。"
            )


@dataclass
class RecoveryStats:
    retries: int = 0
    transport_retries: int = 0
    timeouts: bool = False
    terminal_refusal: str | None = None
    forced_terminal: bool = False
    loop_broken: bool = False
    fallback_used: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "retries": self.retries,
            "transport_retries": self.transport_retries,
            "timeout": self.timeouts,
            "forced_terminal": self.forced_terminal,
            "loop_broken": self.loop_broken,
            "fallback_used": self.fallback_used,
            "errors": self.errors[:10],
        }


def call_model(client, messages, tools, *, policy: RecoveryPolicy, stats: RecoveryStats,
               temperature: float, max_tokens: int, on_delta=None, cancel: CancelToken | None = None):
    """One model call, with bounded retries on transport-level failures.

    Returns None when every attempt failed; the caller then falls back to the
    conservative default for the turn rather than propagating an exception.

    With `on_delta` or `cancel` the call goes through `chat_stream`: fragments go
    to `on_delta` as they arrive (a ``reset`` fragment precedes a retry that
    follows a partly streamed attempt), and `cancel` stops the call mid-flight.
    Each attempt runs on a child of `cancel`, so the hard deadline below pulls
    the plug on that one call -- kills the `claude -p` process, shuts the socket
    -- instead of only walking away from its thread. Cancelled is raised, never
    retried: nobody wants the answer any more.
    """
    delay = policy.backoff_s
    streaming = on_delta is not None or cancel is not None
    for attempt in range(policy.max_transport_retries + 1):
        sent = {"any": False}

        def relay(d, _sent=sent):
            _sent["any"] = True
            on_delta(d)

        call_cancel = cancel.child() if cancel is not None else CancelToken()
        try:
            if streaming:
                fn = lambda: collect(client.chat_stream(messages, tools=tools, temperature=temperature,  # noqa: E731
                                                        max_tokens=max_tokens, timeout=policy.timeout_s,
                                                        cancel=call_cancel), relay if on_delta else None)
            else:
                fn = lambda: client.chat(messages, tools=tools, temperature=temperature,  # noqa: E731
                                         max_tokens=max_tokens, timeout=policy.timeout_s)
            return _with_deadline(fn, policy.timeout_s * 1.5 + 5, cancel=call_cancel)
        except Cancelled:
            if cancel is not None and cancel.is_set():
                raise
            # only this call's own deadline fired: a transport timeout like any other
            stats.errors.append("model call cancelled at its hard deadline")
            stats.timeouts = True
            if sent["any"] and on_delta is not None:
                on_delta(Delta("reset"))
            if attempt == policy.max_transport_retries:
                return None
            stats.transport_retries += 1
            policy.sleep(delay)
            delay *= 2
            continue
        except ProviderError as exc:
            if sent["any"] and on_delta is not None:
                on_delta(Delta("reset"))
            stats.errors.append(str(exc)[:200])
            # Money, key and quota are not weather. Retrying one of these buys
            # nothing and the fallback that follows it silently replaces the
            # model with the harness.
            if is_terminal_refusal(exc):
                # Ending a game is expensive -- a dozen model calls of play are
                # thrown away -- and gateways do report an empty wallet that is
                # not empty. So confirm it once before acting on it.
                if attempt == 0 and policy.confirm_terminal:
                    stats.transport_retries += 1
                    policy.sleep(policy.confirm_after_s)
                    continue
                stats.terminal_refusal = str(exc)[:200]
                # Raised in full: the reason often sits past the 200th character
                # ("... is suspended due to insufficient balance"), and whoever
                # reads this next has to classify it. Truncate when displaying,
                # never before deciding what it says.
                raise GatewayExhausted(str(exc)) from exc
            retryable = exc.status in RETRYABLE_STATUS or exc.status is None
            if "timed out" in str(exc).lower():
                stats.timeouts = True
            if not retryable or attempt == policy.max_transport_retries:
                return None
            stats.transport_retries += 1
            policy.sleep(delay)
            delay *= 2
        except ReplayDivergence:
            raise                 # a replay that no longer matches its recording is not a flaky model
        except Exception as exc:  # noqa: BLE001 -- a client bug must not kill a game
            stats.errors.append(f"{type(exc).__name__}: {exc}"[:200])
            return None
    return None


# Every model call in the process goes through this pool (for the hard deadline), so
# its size is a process-wide cap on concurrent model calls. The online service raises it
# (MODEL_CALL_THREADS); load testing found 32 capping throughput at ~100 calls/s.
_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get("MODEL_CALL_THREADS", "32")),
                                              thread_name_prefix="model-call")


def _with_deadline(fn, seconds: float, cancel: CancelToken | None = None):
    """Run one model call under a hard wall-clock deadline.

    A socket timeout is per read: a relay that trickles keep-alive bytes never
    trips it, and one game sat on a single call for two hours. The call runs in
    a worker thread; past the deadline `cancel` is pulled (a streaming client
    kills its process or shuts its socket) and the turn is treated as a
    transport timeout. A cancel from outside is noticed within 0.1 s even when
    the client cannot be interrupted: the thread is abandoned, the caller is not.
    """
    # carry the caller's contextvars (trace context, log fields) into the pool thread
    future = _POOL.submit(contextvars.copy_context().run, fn)
    end = time.monotonic() + seconds
    while True:
        try:
            return future.result(timeout=0.1 if cancel is not None else seconds)
        except concurrent.futures.TimeoutError as exc:
            if cancel is not None and cancel.is_set():
                future.cancel()
                raise Cancelled(cancel.reason or "cancelled") from exc
            if time.monotonic() < end and cancel is not None:
                continue
            future.cancel()
            if cancel is not None:
                cancel.cancel("deadline")
            raise ProviderError(f"model call timed out after {seconds:.0f}s (hard deadline)", status=None) from exc
