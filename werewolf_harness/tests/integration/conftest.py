"""Fixtures for tests that need real Postgres and Redis.

Point TEST_DATABASE_URL / TEST_REDIS_URL at throwaway instances (CI starts them as
services; `make dev` starts them in Docker). If they are not reachable the whole
directory is skipped, so the core suite still runs on a laptop with neither.
"""
from __future__ import annotations

import asyncio
import os
import socket
import urllib.parse

import pytest

DB_URL = os.environ.get("TEST_DATABASE_URL", "postgresql+asyncpg://postgres@127.0.0.1:5432/wh_test")
REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://127.0.0.1:6379/15")


def _reachable(url: str) -> bool:
    u = urllib.parse.urlparse(url.replace("+asyncpg", ""))
    try:
        with socket.create_connection((u.hostname, u.port or 5432), timeout=0.5):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    if _reachable(DB_URL) and _reachable(REDIS_URL):
        return
    skip = pytest.mark.skip(reason="Postgres/Redis not reachable (set TEST_DATABASE_URL / TEST_REDIS_URL)")
    for it in items:
        if "integration" in str(it.fspath):
            it.add_marker(skip)


@pytest.fixture(scope="session")
def _migrated():
    from werewolf_harness.app.storage import migrate
    migrate.downgrade(DB_URL)
    migrate.upgrade(DB_URL)
    return DB_URL


@pytest.fixture
async def store(_migrated):
    from sqlalchemy import text

    from werewolf_harness.app.storage import Store
    s = Store.connect(_migrated, pool_size=5)
    async with s.engine.begin() as c:
        await c.execute(text("TRUNCATE llm_calls, agent_state, journal, game_seats, games, users CASCADE"))
    yield s
    await s.close()


@pytest.fixture
async def redis():
    from redis.asyncio import Redis
    r = Redis.from_url(REDIS_URL, decode_responses=True)
    await r.flushdb()
    yield r
    await r.flushdb()
    await r.aclose()


@pytest.fixture
def settings():
    from werewolf_harness.app.config import Settings
    return Settings(database_url=DB_URL, redis_url=REDIS_URL, jwt_secret="test-secret-that-is-at-least-32-bytes-long",
                    lease_ttl_ms=1500, lease_renew_s=0.3, recovery_scan_s=0.3,
                    agent_timeout_s=5.0, human_turn_s=30.0)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)
