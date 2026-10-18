"""the database at 100k games: lobby/user-history indexes, lean request index, game archive

Revision ID: j1_db_at_scale
Revises: 0001

What each piece is for (numbers in docs/db_scale.md, reasoning in docs/adr/0012):

  games_created_idx         lobby "newest first" + keyset pages: was a parallel seq scan + sort
  game_seats.game_created_at + game_seats_user_time_idx
                            "my games": the newest 20 seats straight off the index; was
                            join every game the player ever sat in, then sort
  journal_request_uidx      the request-id uniqueness only for rows that have one (a few %),
                            replacing UNIQUE (game_id, request_id) which indexed every step
  games.archived_at, game_archive, games_archivable_idx
                            finished games' journals packed into one jsonb each

Applying it to a live database: the index builds are CREATE/DROP INDEX CONCURRENTLY
inside alembic's autocommit block -- they take no lock that blocks reads or journal
writes, at the price of two table scans each and of not being transactional (a
failed build leaves an INVALID index; IF NOT EXISTS would then skip it, so a rerun
first drops invalid ones). The column adds are catalog-only (no default, no rewrite);
the backfill touches only claimed seats. Dropping the old UNIQUE constraint takes an
ACCESS EXCLUSIVE lock on journal for an instant (catalog change, no scan), and is
done after its replacement is valid so uniqueness is never unenforced.
"""
from alembic import op

revision = "j1_db_at_scale"
down_revision = "h1_queued_state"

TXN_UP = """
ALTER TABLE games ADD COLUMN IF NOT EXISTS archived_at timestamptz;
ALTER TABLE game_seats ADD COLUMN IF NOT EXISTS game_created_at timestamptz;
CREATE TABLE IF NOT EXISTS game_archive (
    game_id      text PRIMARY KEY REFERENCES games(id) ON DELETE CASCADE,
    steps        integer NOT NULL,
    journal      jsonb NOT NULL,   -- {"v": 1, "rows": [[seq, actor, task, action, forced, request_id], ...]}
    memory       jsonb,            -- {seat: [belief, note]}: each model seat's last entry
    archived_at  timestamptz NOT NULL DEFAULT now()
)
"""

# lz4 (PG14+, if the server was built with it) packs faster and a little smaller than pglz
LZ4 = """
DO $$ BEGIN
  ALTER TABLE game_archive ALTER COLUMN journal SET COMPRESSION lz4;
  ALTER TABLE game_archive ALTER COLUMN memory SET COMPRESSION lz4;
EXCEPTION WHEN feature_not_supported OR invalid_parameter_value THEN NULL;
END $$
"""

# outside the DDL transaction, so the backfill holds row locks, not the ADD COLUMN's table lock
BACKFILL = """
UPDATE game_seats s SET game_created_at = g.created_at FROM games g
 WHERE g.id = s.game_id AND s.user_id IS NOT NULL AND s.game_created_at IS NULL
"""

CONCURRENT_UP = [
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS games_created_idx ON games (created_at DESC, id DESC)",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS game_seats_user_time_idx ON game_seats "
    "(user_id, game_created_at DESC NULLS LAST, game_id DESC) INCLUDE (seat) WHERE user_id IS NOT NULL",
    "DROP INDEX CONCURRENTLY IF EXISTS game_seats_user_idx",
    "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS journal_request_uidx ON journal (game_id, request_id) "
    "WHERE request_id IS NOT NULL",
    "ALTER TABLE journal DROP CONSTRAINT IF EXISTS journal_game_id_request_id_key",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS games_archivable_idx ON games (ended_at) "
    "WHERE archived_at IS NULL AND state IN ('over', 'aborted', 'crashed')",
]

DROP_INVALID = """
DO $$ DECLARE r record; BEGIN
  FOR r IN SELECT c.relname FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
           WHERE NOT i.indisvalid AND c.relname IN ('games_created_idx', 'game_seats_user_time_idx',
                 'journal_request_uidx', 'games_archivable_idx') LOOP
    EXECUTE format('DROP INDEX %I', r.relname);
  END LOOP;
END $$
"""

# Downgrade puts archived games back into the hot tables first: dropping game_archive
# with journals in it would lose games.
UNARCHIVE = """
INSERT INTO journal (game_id, seq, actor, task, action, forced, request_id)
SELECT a.game_id, (r->>0)::int, (r->>1)::smallint, r->>2, r->3, (r->>4)::boolean, r->>5
  FROM game_archive a, jsonb_array_elements(a.journal->'rows') r
ON CONFLICT DO NOTHING;
INSERT INTO agent_state (game_id, seat, seq, belief, note)
SELECT a.game_id, m.key::smallint, a.steps - 1, m.value->0, coalesce(m.value->>1, '')
  FROM game_archive a, jsonb_each(coalesce(a.memory, '{}'::jsonb)) m
ON CONFLICT DO NOTHING;
UPDATE games SET archived_at = NULL WHERE archived_at IS NOT NULL
"""

CONCURRENT_DOWN = [
    "DROP INDEX CONCURRENTLY IF EXISTS games_archivable_idx",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS game_seats_user_idx ON game_seats (user_id, game_id) "
    "WHERE user_id IS NOT NULL",
    "DROP INDEX CONCURRENTLY IF EXISTS game_seats_user_time_idx",
    "DROP INDEX CONCURRENTLY IF EXISTS games_created_idx",
]

TXN_DOWN = """
ALTER TABLE journal ADD CONSTRAINT journal_game_id_request_id_key UNIQUE (game_id, request_id);
DROP INDEX IF EXISTS journal_request_uidx;
DROP TABLE IF EXISTS game_archive;
ALTER TABLE game_seats DROP COLUMN IF EXISTS game_created_at;
ALTER TABLE games DROP COLUMN IF EXISTS archived_at
"""


def _each(sql: str):
    for stmt in sql.split(";\n"):
        if stmt.strip():
            op.execute(stmt)


def upgrade() -> None:
    _each(TXN_UP)
    op.execute(LZ4)
    with op.get_context().autocommit_block():
        op.execute(BACKFILL)
        op.execute(DROP_INVALID)
        for stmt in CONCURRENT_UP:
            op.execute(stmt)


def downgrade() -> None:
    _each(UNARCHIVE)
    with op.get_context().autocommit_block():
        for stmt in CONCURRENT_DOWN:
            op.execute(stmt)
    _each(TXN_DOWN)
