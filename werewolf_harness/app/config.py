"""One place that reads the environment. Everything else takes a Settings."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/wh"
    redis_url: str = "redis://localhost:6379/0"
    jwt_secret: str = "dev-only-secret-change-me"
    jwt_ttl_s: int = 7 * 24 * 3600
    seat_token_ttl_s: int = 6 * 3600
    node_id: str = "node-1"

    # runtime
    lease_ttl_ms: int = 10_000          # a node that stops renewing loses the game after this
    lease_renew_s: float = 3.0
    recovery_scan_s: float = 5.0        # how often a node looks for games nobody holds
    recovery_batch: int = 50            # orphaned games adopted per round trip during a scan
    agent_timeout_s: float = 240.0      # a model turn that takes longer is replaced by the fallback
    claim_idle_ms: int = 15_000         # a job unacknowledged this long by a silent worker is run by another
    human_turn_s: float = 300.0

    # model layer
    agent_executor: str = "local"       # local | streams
    agent_workers: int = 16
    daily_budget_usd: float = 5.0
    # tokens per minute per model, as the provider grants them; {"gpt-x": 800000, ...}
    llm_tpm: dict[str, int] = {}
    llm_default_tpm: int = 2_000_000

    # overload protection (runtime/load.py, runtime/admission.py; docs/adr/0011)
    admission: bool = True              # off = every game starts at once, whatever the load
    max_queued_games: int = 50          # waiting queue; beyond it creation gets 503 + Retry-After
    max_active_games: int = 0           # hard cap on games driven by this node (0 = only the load signal)
    admission_probe_s: float = 5.0      # while busy (exit < pressure < enter): at most one start per this
    load_tick_s: float = 0.5            # how often the load signal is sampled and the queue drained
    overload_wait_s: float = 2.0        # observed or expected wait of a model job this long = pressure 1
    overload_demand_per_slot: float = 1.0  # smoothed model jobs wanting a slot, per slot = pressure 1
    overload_latency_ms: float = 20_000.0  # recent p50 per-call model latency = pressure 1
    overload_enter: float = 1.0
    overload_exit: float = 0.7
    overload_hold_s: float = 3.0
    overload_per_game_tau_s: float = 20.0  # smoothing of the jobs-per-game estimate (load.py)
    fair_queuing: bool = True           # LocalExecutor: per-game fair order instead of one FIFO
    per_game_asks: int = 4              # Streams executor: model jobs one game may have in the queue at once

    # api
    rate_limit_per_min: int = 240       # per IP, all routes
    create_limit_per_hour: int = 30     # per user, POST /api/games
    heartbeat_s: float = 25.0           # < Cloudflare's 100 s idle cut, with room for two misses


@lru_cache
def get_settings() -> Settings:
    return Settings()
