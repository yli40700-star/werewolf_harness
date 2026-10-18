"""Prometheus metrics. Scraped at /metrics; the Grafana board is deploy/grafana/.

Naming: wh_<what>_<unit>. Histograms in milliseconds with buckets chosen for
what is being measured (a journal commit is milliseconds, a model turn is seconds).
"""
from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

HTTP_MS = Histogram("wh_http_request_ms", "HTTP latency", ["route", "method", "code"],
                    buckets=(1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500))
COMMIT_MS = Histogram("wh_commit_ms", "journal-first commit (DB write + apply + publish)",
                      buckets=(0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256))
GAMES_CREATED = Counter("wh_games_created_total", "games created")
GAMES_FINISHED = Counter("wh_games_finished_total", "games ended", ["result"])
ADOPTIONS = Counter("wh_adoptions_total", "games adopted from a dead or draining node")
STALE_RESULTS = Counter("wh_stale_agent_results_total", "model answers that arrived after their turn was closed")
RATE_LIMITED = Counter("wh_rate_limited_total", "requests refused by a rate limit", ["scope"])
WS_CONNECTIONS = Gauge("wh_ws_connections", "open WebSocket connections")
WS_HEARTBEAT_TIMEOUTS = Counter("wh_ws_heartbeat_timeouts_total", "connections closed for missing heartbeats")

LLM_CALL_MS = Histogram("wh_llm_call_ms", "one model request", ["model", "status"],
                        buckets=(100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000))
LLM_TTFT_MS = Histogram("wh_llm_ttft_ms", "time to first token", ["model"],
                        buckets=(50, 100, 250, 500, 1000, 2000, 4000, 8000))
LLM_TOKENS = Counter("wh_llm_tokens_total", "tokens", ["model", "kind"])       # kind: input | cached | output
LLM_COST = Counter("wh_llm_cost_usd_total", "estimated spend", ["model"])
LLM_FALLBACKS = Counter("wh_llm_fallbacks_total", "a call served by a fallback", ["reason"])
BREAKER_STATE = Gauge("wh_breaker_open", "1 while a provider's breaker is open", ["provider"])
QUEUE_WAIT_MS = Histogram("wh_llm_queue_wait_ms", "time a model job waited for a worker", ["lane"],
                          buckets=(1, 5, 10, 50, 100, 500, 1000, 5000, 20000))

ACTIVE_GAMES = Gauge("wh_active_games", "games driven by this node")
MAILBOX_BACKLOG = Gauge("wh_mailbox_backlog", "messages waiting in all game mailboxes")
TIMERS_PENDING = Gauge("wh_timers_pending", "timeouts in the timing wheel")
AGENTS_INFLIGHT = Gauge("wh_agents_inflight", "model turns running")

# overload protection (runtime/load.py, runtime/admission.py)
LOAD_PRESSURE = Gauge("wh_load_pressure", "overload score of the model layer (1.0 = at the limit)")
OVERLOADED = Gauge("wh_overloaded", "1 while the node refuses to start new games")
QUEUE_DEPTH = Gauge("wh_llm_queue_depth", "model jobs waiting for a worker", ["lane"])
ADMISSION_QUEUE = Gauge("wh_admission_queue", "new games waiting to be started")
ADMISSIONS = Counter("wh_admissions_total", "what happened to a game creation", ["decision"])  # admit|queue|reject
ADMISSION_WAIT_S = Histogram("wh_admission_wait_s", "how long a queued game waited to start",
                             buckets=(1, 2, 5, 10, 20, 30, 60, 120, 300, 600))

_rt = None


def bind_runtime(rt) -> None:
    global _rt
    _rt = rt


def _refresh() -> None:
    if _rt is None:
        return
    ACTIVE_GAMES.set(len(_rt.actors))
    MAILBOX_BACKLOG.set(sum(a.mailbox.qsize() for a in _rt.actors.values()))
    TIMERS_PENDING.set(_rt.wheel.pending)
    AGENTS_INFLIGHT.set(getattr(_rt.executor, "inflight", 0))
    adm = getattr(_rt, "admission", None)
    if adm is not None:
        ADMISSION_QUEUE.set(len(adm))


def render() -> tuple[bytes, str]:
    _refresh()
    return generate_latest(), CONTENT_TYPE_LATEST
