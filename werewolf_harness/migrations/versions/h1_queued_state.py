"""games.state may be 'queued': created, waiting for admission (docs/adr/0011)

Revision ID: h1_queued_state
"""
from alembic import op

revision = "h1_queued_state"
down_revision = "e1_engine_version"


def upgrade() -> None:
    op.execute("ALTER TABLE games DROP CONSTRAINT IF EXISTS games_state_check")
    op.execute("ALTER TABLE games ADD CONSTRAINT games_state_check "
               "CHECK (state IN ('queued', 'waiting', 'running', 'over', 'aborted', 'crashed'))")


def downgrade() -> None:
    # a queued game never started; there is nothing to keep
    op.execute("UPDATE games SET state = 'aborted', error = 'queued game dropped by downgrade' WHERE state = 'queued'")
    op.execute("ALTER TABLE games DROP CONSTRAINT IF EXISTS games_state_check")
    op.execute("ALTER TABLE games ADD CONSTRAINT games_state_check "
               "CHECK (state IN ('waiting', 'running', 'over', 'aborted', 'crashed'))")
