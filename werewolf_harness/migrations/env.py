"""Alembic, async. Migrations are hand-written SQL: the schema is small and the
point of each index is easier to argue about in SQL than in ORM metadata."""
from __future__ import annotations

import asyncio

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

url = context.config.get_main_option("sqlalchemy.url")
if not url:
    from werewolf_harness.app.config import get_settings
    url = get_settings().database_url


def _run(connection) -> None:
    context.configure(connection=connection, target_metadata=None)
    with context.begin_transaction():
        context.run_migrations()


async def _main() -> None:
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        await conn.run_sync(_run)
        await conn.commit()
    await engine.dispose()


if context.is_offline_mode():
    context.configure(url=url, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(_main())
