"""games.engine_version: the rules a game was created under

A game is replayed (crash recovery, adoption, read-only loads) with the engine
version it recorded, not with whatever code is deployed. Every row that exists
before this migration was written by unversioned code, which played the baseline
rules, so the default is 1 -- and it stays 1, because a node still running the old
code during a rolling deploy inserts without the column and does play v1.

Revision ID: e1_engine_version
"""
from alembic import op

revision = "e1_engine_version"
down_revision = "0001"


def upgrade() -> None:
    # No index: "which live games are on version N?" (compat_check --db, retiring a
    # version) reads the few live rows through games_live_idx and filters.
    op.execute("ALTER TABLE games ADD COLUMN engine_version integer NOT NULL DEFAULT 1")


def downgrade() -> None:
    op.execute("ALTER TABLE games DROP COLUMN engine_version")
