"""initial schema: games, seats, journal, agent state, model calls

Revision ID: 0001
"""
from alembic import op

revision = "0001"
down_revision = None

UP = """
CREATE TABLE users (
    id          text PRIMARY KEY,
    kind        text NOT NULL CHECK (kind IN ('guest', 'oauth')),
    name        text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE games (
    id             text PRIMARY KEY,
    variant        text NOT NULL,
    seed           integer NOT NULL,
    config         jsonb NOT NULL,             -- models, guard, playbook, human seats, clock, budget
    state          text NOT NULL CHECK (state IN ('waiting', 'running', 'over', 'aborted', 'crashed')),
    owner_node     text,                       -- who drives it now (informational; the lease decides)
    fencing_token  bigint NOT NULL DEFAULT 0,  -- bumped on every lease acquisition; journal writes must match
    outcome        jsonb,
    error          text,
    created_by     text REFERENCES users(id),
    resumed_from   text,
    created_at     timestamptz NOT NULL DEFAULT now(),
    ended_at       timestamptz
);
-- startup and the recovery scan ask "which games are still live?"; the partial
-- index keeps that lookup small however many finished games pile up
CREATE INDEX games_live_idx ON games (created_at) WHERE state IN ('waiting', 'running');

CREATE TABLE game_seats (
    game_id  text NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    seat     smallint NOT NULL,
    kind     text NOT NULL CHECK (kind IN ('human', 'llm', 'script')),
    model    text,
    user_id  text REFERENCES users(id),        -- the person who claimed a human seat
    PRIMARY KEY (game_id, seat)
);
-- "all games this player sat in, newest first": leftmost column is the filter
CREATE INDEX game_seats_user_idx ON game_seats (user_id, game_id) WHERE user_id IS NOT NULL;

-- The source of truth. world == f(seed, journal): one row per applied step.
CREATE TABLE journal (
    game_id     text NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    seq         integer NOT NULL,              -- 0-based position; the primary key makes "apply once" a DB fact
    actor       smallint NOT NULL,
    task        text NOT NULL,
    action      jsonb NOT NULL,
    forced      boolean NOT NULL DEFAULT false,
    request_id  text,                          -- a client's retry key; the same click lands once
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (game_id, seq),
    UNIQUE (game_id, request_id)
);

-- What a model seat remembers between its turns (belief table + note to self).
-- It is not in the world, so replay cannot rebuild it: without this table a
-- recovered game continues with amnesiac AI.
CREATE TABLE agent_state (
    game_id  text NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    seat     smallint NOT NULL,
    seq      integer NOT NULL,                 -- the journal row this state was produced with
    belief   jsonb NOT NULL,
    note     text NOT NULL DEFAULT '',
    PRIMARY KEY (game_id, seat, seq)
);

-- One row per model request (attempt), for cost, latency and cache accounting.
CREATE TABLE llm_calls (
    id               bigserial PRIMARY KEY,
    game_id          text,
    seat             smallint,
    seq              integer,
    model            text NOT NULL,
    provider         text,
    status           text NOT NULL,           -- ok | error | timeout | rejected | fallback
    input_tokens     integer NOT NULL DEFAULT 0,
    cached_tokens    integer NOT NULL DEFAULT 0,
    output_tokens    integer NOT NULL DEFAULT 0,
    latency_ms       integer NOT NULL DEFAULT 0,
    cost_usd         numeric(12, 6) NOT NULL DEFAULT 0,
    fallback_reason  text,
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX llm_calls_game_idx ON llm_calls (game_id, seq);
CREATE INDEX llm_calls_time_idx ON llm_calls (created_at);
"""

DOWN = """
DROP TABLE IF EXISTS llm_calls, agent_state, journal, game_seats, games, users CASCADE;
"""


def _each(sql: str):
    # asyncpg prepares statements one at a time
    for stmt in sql.split(";\n"):
        if stmt.strip():
            op.execute(stmt)


def upgrade() -> None:
    _each(UP)


def downgrade() -> None:
    _each(DOWN)
