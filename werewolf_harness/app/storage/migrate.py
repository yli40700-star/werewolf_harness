"""Run migrations from code (tests, container start)."""
from __future__ import annotations

import pathlib

from alembic import command
from alembic.config import Config

ROOT = pathlib.Path(__file__).resolve().parents[2]


def upgrade(url: str, rev: str = "head") -> None:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, rev)


def downgrade(url: str, rev: str = "base") -> None:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.downgrade(cfg, rev)


async def aupgrade(url: str) -> None:
    """From inside a running event loop (app startup): alembic's env.py runs its own loop."""
    import asyncio
    await asyncio.to_thread(upgrade, url)


async def adowngrade(url: str) -> None:
    import asyncio
    await asyncio.to_thread(downgrade, url)
