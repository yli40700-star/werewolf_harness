"""Postgres access. Plain SQL through SQLAlchemy Core (async, asyncpg).

The journal write is the one that matters, and it enforces two things in the
database rather than in Python:

  * apply once     -- PRIMARY KEY (game_id, seq): a second write of the same step fails.
  * one writer     -- the caller's fencing token must equal games.fencing_token, read
                      FOR SHARE in the same transaction. A node whose lease expired
                      (GC pause, network partition, SIGSTOP) and then wakes up still
                      holds its old token; its write is refused instead of forking the game.
"""
from __future__ import annotations

import json
import secrets
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


class Fenced(Exception):
    """This node no longer owns the game: its fencing token is stale."""


class DuplicateStep(Exception):
    """The step (or the client request) is already in the journal."""


def _j(v: Any) -> Any:
    return json.loads(v) if isinstance(v, str) else v


@dataclass
class JournalRow:
    seq: int
    actor: int
    task: str
    action: Any
    forced: bool
    request_id: str | None


class Store:
    def __init__(self, engine: AsyncEngine):
        self.engine = engine

    @classmethod
    def connect(cls, url: str, pool_size: int = 10) -> Store:
        return cls(create_async_engine(url, pool_size=pool_size, max_overflow=pool_size, pool_pre_ping=True))

    async def close(self) -> None:
        await self.engine.dispose()

    async def ping(self) -> bool:
        async with self.engine.connect() as c:
            return (await c.scalar(text("SELECT 1"))) == 1

    # ------------------------------------------------------------ users
    async def create_guest(self, name: str | None = None) -> dict:
        uid = "u_" + secrets.token_hex(6)
        name = name or f"游客{uid[-4:]}"
        async with self.engine.begin() as c:
            await c.execute(text("INSERT INTO users (id, kind, name) VALUES (:id, 'guest', :n)"), {"id": uid, "n": name})
        return {"id": uid, "name": name, "kind": "guest"}

    async def get_user(self, uid: str) -> dict | None:
        async with self.engine.connect() as c:
            r = (await c.execute(text("SELECT id, kind, name FROM users WHERE id = :id"), {"id": uid})).mappings().first()
        return dict(r) if r else None

    # ------------------------------------------------------------ games
    async def create_game(self, *, game_id: str, variant: str, seed: int, config: dict, state: str,
                          seats: Iterable[tuple[int, str, str | None]], created_by: str | None = None,
                          resumed_from: str | None = None, engine_version: int | None = None,
                          owner_node: str | None = None) -> None:
        """Every game records the engine version it is created with: ``engine_version``,
        else the one in ``config`` (Table pins it there), else the current one. The column
        is what recovery reads; the config copy (kept equal) is what a worker's Table sees.
        ``owner_node`` is set for a queued game: the node whose admission line holds it."""
        from ...envs import ENGINE_VERSION
        ev = engine_version if engine_version is not None else config.get("engine_version")
        ev = ENGINE_VERSION if ev is None else int(ev)
        config = {**config, "engine_version": ev}
        async with self.engine.begin() as c:
            await c.execute(text(
                "INSERT INTO games (id, variant, seed, config, state, created_by, resumed_from, engine_version, "
                "owner_node) VALUES (:id, :v, :s, CAST(:cfg AS jsonb), :st, :by, :rf, :ev, :own)"),
                {"id": game_id, "v": variant, "s": seed, "cfg": json.dumps(config, ensure_ascii=False),
                 "st": state, "by": created_by, "rf": resumed_from, "ev": ev, "own": owner_node})
            await c.execute(text("INSERT INTO game_seats (game_id, seat, kind, model) VALUES (:g, :seat, :k, :m)"),
                            [{"g": game_id, "seat": s, "k": k, "m": m} for s, k, m in seats])

    async def get_game(self, game_id: str) -> dict | None:
        async with self.engine.connect() as c:
            r = (await c.execute(text("SELECT * FROM games WHERE id = :id"), {"id": game_id})).mappings().first()
        if not r:
            return None
        d = dict(r)
        d["config"], d["outcome"] = _j(d["config"]), _j(d["outcome"])
        return d

    async def set_state(self, game_id: str, state: str, *, outcome: dict | None = None, error: str | None = None) -> None:
        ended = state in ("over", "aborted", "crashed")
        async with self.engine.begin() as c:
            await c.execute(text(
                "UPDATE games SET state = :st, outcome = COALESCE(CAST(:o AS jsonb), outcome), "
                "error = COALESCE(:e, error), ended_at = CASE WHEN :ended THEN now() ELSE ended_at END "
                "WHERE id = :id"),
                {"id": game_id, "st": state, "o": json.dumps(outcome, ensure_ascii=False, default=str) if outcome else None,
                 "e": error, "ended": ended})

    async def live_games(self) -> list[dict]:
        """Games somebody should be driving. Served by games_live_idx."""
        async with self.engine.connect() as c:
            rows = (await c.execute(text(
                "SELECT id, state, owner_node, fencing_token, engine_version FROM games "
                "WHERE state IN ('waiting', 'running') ORDER BY created_at"))).mappings().all()
        return [dict(r) for r in rows]

    async def queued_games(self, node: str) -> list[dict]:
        """Games this node put in its admission queue and has not started (oldest first)."""
        async with self.engine.connect() as c:
            rows = (await c.execute(text(
                "SELECT id, seed, config, created_at, engine_version FROM games "
                "WHERE state = 'queued' AND owner_node = :n ORDER BY created_at"), {"n": node})).mappings().all()
        return [{**dict(r), "config": _j(r["config"])} for r in rows]

    async def recent_games(self, limit: int = 50, before: tuple | None = None) -> list[dict]:
        """Newest first, one page. `before` = (created_at, id) of the last row of the
        previous page: keyset pagination on games_created_idx, so page 200 costs what
        page 1 costs (OFFSET would read and throw away every row before it)."""
        where, args = "", {"n": limit}
        if before is not None:
            where, args = "WHERE (created_at, id) < (:bt, :bid) ", {**args, "bt": before[0], "bid": before[1]}
        async with self.engine.connect() as c:
            rows = (await c.execute(text(
                "SELECT id, variant, seed, config, state, outcome, error, owner_node, created_at, ended_at, "
                "engine_version "
                f"FROM games {where}ORDER BY created_at DESC, id DESC LIMIT :n"), args)).mappings().all()
        return [{**dict(r), "config": _j(r["config"]), "outcome": _j(r["outcome"])} for r in rows]

    async def finished_games(self, limit: int = 500) -> list[dict]:
        """The newest `limit` games that ended normally (verify_replay's sample)."""
        async with self.engine.connect() as c:
            rows = (await c.execute(text(
                "SELECT id, seed, config, outcome, created_at, engine_version FROM games WHERE state = 'over' "
                "ORDER BY created_at DESC, id DESC LIMIT :n"), {"n": limit})).mappings().all()
        return [{**dict(r), "config": _j(r["config"]), "outcome": _j(r["outcome"])} for r in rows]

    async def seats(self, game_id: str) -> list[dict]:
        async with self.engine.connect() as c:
            rows = (await c.execute(text("SELECT seat, kind, model, user_id FROM game_seats WHERE game_id = :g ORDER BY seat"),
                                    {"g": game_id})).mappings().all()
        return [dict(r) for r in rows]

    async def claim_seat(self, game_id: str, seat: int, user_id: str) -> bool:
        """First person to claim a human seat owns it; the same person may claim it again."""
        async with self.engine.begin() as c:
            # game_created_at is the game's created_at copied onto the seat, so "my games,
            # newest first" is one index range (game_seats_user_time_idx), not a join + sort
            r = await c.execute(text(
                "UPDATE game_seats SET user_id = :u, "
                "game_created_at = (SELECT created_at FROM games WHERE id = :g) "
                "WHERE game_id = :g AND seat = :s AND kind = 'human' "
                "AND (user_id IS NULL OR user_id = :u) RETURNING seat"), {"g": game_id, "s": seat, "u": user_id})
            return r.first() is not None

    async def user_games(self, user_id: str, limit: int = 20, before: tuple | None = None) -> list[dict]:
        """A player's games, newest first; `before` = (created_at, id) of the previous page's last row.

        Served by game_seats_user_time_idx (user_id, game_created_at DESC, game_id DESC):
        the newest `limit` seats are read straight off the index, then `limit` games by
        primary key. Joining first and sorting after read every game the player ever sat in.
        """
        where, args = "", {"u": user_id, "n": limit}
        if before is not None:
            where = "AND (s.game_created_at, s.game_id) < (:bt, :bid) "
            args.update(bt=before[0], bid=before[1])
        async with self.engine.connect() as c:
            rows = (await c.execute(text(
                "SELECT g.id, g.variant, g.state, g.outcome, s.seat, g.created_at FROM game_seats s "
                f"JOIN games g ON g.id = s.game_id WHERE s.user_id = :u {where}"
                "ORDER BY s.game_created_at DESC NULLS LAST, s.game_id DESC LIMIT :n"), args)).mappings().all()
        return [{**dict(r), "outcome": _j(r["outcome"])} for r in rows]

    # ------------------------------------------------------------ ownership
    async def bump_fencing(self, game_id: str, node: str) -> int:
        """Called right after winning the lease: every earlier holder's token is now stale."""
        async with self.engine.begin() as c:
            tok = await c.scalar(text(
                "UPDATE games SET fencing_token = fencing_token + 1, owner_node = :n WHERE id = :id "
                "RETURNING fencing_token"), {"id": game_id, "n": node})
        if tok is None:
            raise KeyError(game_id)
        return int(tok)

    async def claim_live(self, game_ids: list[str], node: str) -> dict[str, tuple[int, dict]]:
        """bump_fencing() for a batch of games whose leases we just won, returning each
        new token with the game row -- only for games still live (the others are left
        untouched; the caller hands their leases back)."""
        if not game_ids:
            return {}
        async with self.engine.begin() as c:
            rows = (await c.execute(text(
                "UPDATE games SET fencing_token = fencing_token + 1, owner_node = :n "
                "WHERE id = ANY(:ids) AND state IN ('waiting', 'running') "
                "RETURNING id, seed, config, state, created_at, resumed_from, fencing_token, engine_version"),
                {"ids": list(game_ids), "n": node})).mappings().all()
        return {r["id"]: (int(r["fencing_token"]), {**dict(r), "config": _j(r["config"])}) for r in rows}

    # ------------------------------------------------------------ journal
    async def append(self, game_id: str, *, token: int, seq: int, actor: int, task: str, action: Any,
                     forced: bool, request_id: str | None = None,
                     agent_states: Iterable[tuple[int, dict, str]] = ()) -> None:
        """Write one step (and the model seat's memory it produced) atomically.

        Raises Fenced if `token` is stale, DuplicateStep if the step or the
        request id is already there. Nothing is written in either case.
        """
        try:
            async with self.engine.begin() as c:
                current = await c.scalar(text("SELECT fencing_token FROM games WHERE id = :id FOR SHARE"), {"id": game_id})
                if current != token:
                    raise Fenced(f"{game_id}: token {token} is stale (current {current})")
                await c.execute(text(
                    "INSERT INTO journal (game_id, seq, actor, task, action, forced, request_id) "
                    "VALUES (:g, :seq, :a, :t, CAST(:act AS jsonb), :f, :rid)"),
                    {"g": game_id, "seq": seq, "a": actor, "t": task,
                     "act": json.dumps(action, ensure_ascii=False, default=str), "f": forced, "rid": request_id})
                states = [{"g": game_id, "s": s, "seq": seq, "b": json.dumps(b, ensure_ascii=False), "n": n}
                          for s, b, n in agent_states]
                if states:
                    await c.execute(text(
                        "INSERT INTO agent_state (game_id, seat, seq, belief, note) "
                        "VALUES (:g, :s, :seq, CAST(:b AS jsonb), :n)"), states)
        except IntegrityError as e:
            raise DuplicateStep(f"{game_id}#{seq} (request {request_id})") from e

    async def journal(self, game_id: str, archived: bool = False) -> list[JournalRow]:
        """The game's steps in order -- from the hot table, or unpacked from game_archive.

        `archived` (the game row's archived_at is set) only says which to try first."""
        async with self.engine.connect() as c:
            if archived:
                packed = await c.scalar(text("SELECT journal FROM game_archive WHERE game_id = :g"), {"g": game_id})
                if packed is not None:
                    return unpack_journal(_j(packed))
            rows = (await c.execute(text(
                "SELECT seq, actor, task, action, forced, request_id FROM journal WHERE game_id = :g ORDER BY seq"),
                {"g": game_id})).all()
            if rows:
                return [JournalRow(r.seq, r.actor, r.task, _j(r.action), r.forced, r.request_id) for r in rows]
            packed = await c.scalar(text("SELECT journal FROM game_archive WHERE game_id = :g"), {"g": game_id})
        return unpack_journal(_j(packed)) if packed is not None else []

    async def journals(self, game_ids: list[str]) -> dict[str, list[JournalRow]]:
        """Many games' journals in two queries (hot, then archive for the rest) instead of one per game."""
        out: dict[str, list[JournalRow]] = {g: [] for g in game_ids}
        if not game_ids:
            return out
        async with self.engine.connect() as c:
            for r in (await c.execute(text(
                    "SELECT game_id, seq, actor, task, action, forced, request_id FROM journal "
                    "WHERE game_id = ANY(:ids) ORDER BY game_id, seq"), {"ids": list(game_ids)})).all():
                out[r.game_id].append(JournalRow(r.seq, r.actor, r.task, _j(r.action), r.forced, r.request_id))
            rest = [g for g, rows in out.items() if not rows]
            if rest:
                for r in (await c.execute(text("SELECT game_id, journal FROM game_archive WHERE game_id = ANY(:ids)"),
                                          {"ids": rest})).all():
                    out[r.game_id] = unpack_journal(_j(r.journal))
        return out

    async def step_for_request(self, game_id: str, request_id: str) -> JournalRow | None:
        async with self.engine.connect() as c:
            r = (await c.execute(text(
                "SELECT seq, actor, task, action, forced, request_id FROM journal "
                "WHERE game_id = :g AND request_id = :r"), {"g": game_id, "r": request_id})).first()
        return JournalRow(r.seq, r.actor, r.task, _j(r.action), r.forced, r.request_id) if r else None

    async def agent_memory(self, game_id: str) -> dict[int, tuple[dict, str]]:
        """The latest belief table and note of every model seat."""
        async with self.engine.connect() as c:
            rows = (await c.execute(text(
                "SELECT DISTINCT ON (seat) seat, belief, note FROM agent_state WHERE game_id = :g "
                "ORDER BY seat, seq DESC"), {"g": game_id})).all()
            if not rows:                                   # archived: each seat's last entry was kept
                mem = await c.scalar(text("SELECT memory FROM game_archive WHERE game_id = :g"), {"g": game_id})
                return {int(k): (v[0], v[1]) for k, v in (_j(mem) or {}).items()}
        return {int(r.seat): (_j(r.belief), r.note) for r in rows}

    async def agent_memories(self, game_ids: list[str]) -> dict[str, dict[int, tuple[dict, str]]]:
        """agent_memory() for live games in one query (archived games have none to restore)."""
        out: dict[str, dict[int, tuple[dict, str]]] = {g: {} for g in game_ids}
        if not game_ids:
            return out
        async with self.engine.connect() as c:
            for r in (await c.execute(text(
                    "SELECT DISTINCT ON (game_id, seat) game_id, seat, belief, note FROM agent_state "
                    "WHERE game_id = ANY(:ids) ORDER BY game_id, seat, seq DESC"), {"ids": list(game_ids)})).all():
                out[r.game_id][int(r.seat)] = (_j(r.belief), r.note)
        return out

    # ------------------------------------------------------------ model calls
    async def record_calls(self, rows: list[dict]) -> None:
        if not rows:
            return
        async with self.engine.begin() as c:
            await c.execute(text(
                "INSERT INTO llm_calls (game_id, seat, seq, model, provider, status, input_tokens, cached_tokens, "
                "output_tokens, latency_ms, cost_usd, fallback_reason) VALUES (:game_id, :seat, :seq, :model, :provider, "
                ":status, :input_tokens, :cached_tokens, :output_tokens, :latency_ms, :cost_usd, :fallback_reason)"),
                [{"game_id": None, "seat": None, "seq": None, "provider": None, "cached_tokens": 0,
                  "cost_usd": 0, "fallback_reason": None, **r} for r in rows])

    async def call_stats(self, game_id: str | None = None, hours: float | None = None) -> dict:
        """Usage of one game (llm_calls_game_idx), of the last `hours` (llm_calls_time_idx),
        or of all time. All time is a full scan of llm_calls -- seconds at millions of
        rows -- so the API asks for a window unless the caller insists (hours=0).
        """
        where = "WHERE game_id = :g" if game_id else ""
        if hours and not game_id:
            where = "WHERE created_at >= now() - make_interval(secs => :h)"
        async with self.engine.connect() as c:
            r = (await c.execute(text(
                "SELECT count(*) AS calls, coalesce(sum(input_tokens),0) AS input, coalesce(sum(cached_tokens),0) AS cached, "
                "coalesce(sum(output_tokens),0) AS output, coalesce(sum(cost_usd),0) AS cost, "
                "percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms, "
                "count(*) FILTER (WHERE status = 'ok' AND fallback_reason IS NOT NULL) AS fallbacks, "
                f"count(*) FILTER (WHERE status <> 'ok') AS failed FROM llm_calls {where}"),
                {"g": game_id, "h": float(hours or 0) * 3600})).mappings().first()
        d = dict(r)
        d["cache_hit_rate"] = round(float(d["cached"]) / float(d["input"]), 4) if d["input"] else 0.0
        return d

    # ------------------------------------------------------------ archive
    async def archive_batch(self, older_than_days: float, batch: int = 500) -> list[str]:
        """Pack up to `batch` games that ended more than `older_than_days` ago; returns their ids.

        One statement, one transaction: the journal becomes one jsonb array per game in
        game_archive (TOAST-compressed), model memory shrinks to each seat's last entry,
        and the hot rows are deleted. The games row stays where it is (lobby, "my games"
        and the seat FKs keep working) with archived_at set. FOR UPDATE SKIP LOCKED lets
        two archivers run at once without packing the same game twice.
        """
        async with self.engine.begin() as c:
            rows = (await c.execute(text(ARCHIVE_SQL), {"secs": float(older_than_days) * 86400, "n": batch})).all()
        return [r[0] for r in rows]


# Data-modifying CTEs all read the statement's snapshot, so the INSERT packs the rows
# the two DELETEs remove, and nothing in between can see a game half-archived.
ARCHIVE_SQL = """
WITH pick AS (
    SELECT id FROM games
    WHERE archived_at IS NULL AND state IN ('over', 'aborted', 'crashed')
      AND ended_at < now() - make_interval(secs => :secs)
    ORDER BY ended_at LIMIT :n
    FOR UPDATE SKIP LOCKED
), packed AS (
    INSERT INTO game_archive (game_id, steps, journal, memory)
    SELECT p.id,
           (SELECT count(*) FROM journal j WHERE j.game_id = p.id),
           jsonb_build_object('v', 1, 'rows', coalesce(
               (SELECT jsonb_agg(jsonb_build_array(seq, actor, task, action, forced, request_id) ORDER BY seq)
                FROM journal j WHERE j.game_id = p.id), '[]'::jsonb)),
           (SELECT jsonb_object_agg(seat, jsonb_build_array(belief, note)) FROM (
                SELECT DISTINCT ON (seat) seat, belief, note FROM agent_state a
                WHERE a.game_id = p.id ORDER BY seat, seq DESC) m)
    FROM pick p
    RETURNING game_id
), del_journal AS (
    DELETE FROM journal WHERE game_id IN (SELECT game_id FROM packed)
), del_memory AS (
    DELETE FROM agent_state WHERE game_id IN (SELECT game_id FROM packed)
)
UPDATE games SET archived_at = now() WHERE id IN (SELECT game_id FROM packed) RETURNING id
"""


def unpack_journal(packed: dict) -> list[JournalRow]:
    """game_archive.journal -> rows. v1: {"v": 1, "rows": [[seq, actor, task, action, forced, request_id], ...]}"""
    if packed.get("v") != 1:
        raise ValueError(f"unknown archive format {packed.get('v')!r}")
    return [JournalRow(int(r[0]), int(r[1]), r[2], r[3], bool(r[4]), r[5]) for r in packed["rows"]]
