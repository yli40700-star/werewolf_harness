"""J: the database at 100k games -- archive, keyset pages, batched adoption, migration j1."""
from __future__ import annotations

import asyncio
import datetime as dt

from sqlalchemy import text

from werewolf_harness.app.runtime.runtime import Runtime
from werewolf_harness.app.runtime.table import Table, TableConfig
from werewolf_harness.app.storage import Store, migrate
from werewolf_harness.arena import Record, Terminal
from werewolf_harness.scripts.archive_games import archive
from werewolf_harness.scripts.verify_replay import verify

from .conftest import DB_URL
from .helpers import CountingExecutor, GatedExecutor, until
from .test_runtime import _rt


async def _play(settings, store, redis, seeds=(51,), variant="8p"):
    rt = _rt(settings, store, redis)
    await rt.start(scan=False)
    actors = [await rt.create(TableConfig(variant=variant, models=["mock"]), seed=s) for s in seeds]
    await until(lambda: all(a.t.state == "over" for a in actors), 90)
    await asyncio.sleep(0.3)                       # the final set_state
    await rt.shutdown()
    return [a.id for a in actors]


async def _age(store, gids, days):
    async with store.engine.begin() as c:
        await c.execute(text("UPDATE games SET ended_at = now() - make_interval(days => :d), "
                             "created_at = now() - make_interval(days => :d + 1) WHERE id = ANY(:ids)"),
                        {"d": days, "ids": list(gids)})


async def _count(store, sql, **kw):
    async with store.engine.connect() as c:
        return await c.scalar(text(sql), kw)


# ------------------------------------------------------------------ archive
async def test_an_archived_game_reads_back_the_same_and_replays_to_its_outcome(settings, store, redis):
    old, recent = await _play(settings, store, redis, seeds=(51, 52))
    before = await store.journal(old)
    memory = await store.agent_memory(old)
    outcome = (await store.get_game(old))["outcome"]
    await _age(store, [old], 40)

    r = await archive(store, days=30, batch=10, vacuum=True, log=lambda *_: None)
    assert r["archived"] == 1 and r["vacuum_s"] is not None

    # the hot rows are gone, the packed copy is there, the game row stays (marked)
    assert await _count(store, "SELECT count(*) FROM journal WHERE game_id = :g", g=old) == 0
    assert await _count(store, "SELECT count(*) FROM agent_state WHERE game_id = :g", g=old) == 0
    assert await _count(store, "SELECT steps FROM game_archive WHERE game_id = :g", g=old) == len(before)
    assert (await store.get_game(old))["archived_at"] is not None
    assert (await store.get_game(recent))["archived_at"] is None          # ended just now: left alone

    # the read path: same rows, same memory, replays to the recorded outcome
    assert await store.journal(old) == before
    assert (await store.journals([old, recent]))[old] == before
    assert await store.agent_memory(old) == memory
    t = Table(old, 51, TableConfig(variant="8p", models=["mock"]))
    Runtime.replay(t, [Record(x.actor, x.task, x.action, x.forced) for x in await store.journal(old)])
    assert isinstance(t.turn, Terminal) and t.turn.outcome["winner"] == outcome["winner"]
    rt = _rt(settings, store, redis)
    ro = await rt.load_readonly(old)
    assert ro.state == "over" and len(ro.journal) == len(before)
    v = await verify(store)
    assert v["checked"] == 2 and v["consistent_rate"] == 1.0, v

    # idempotent: nothing left to do
    assert (await archive(store, days=30, log=lambda *_: None))["archived"] == 0


async def test_live_games_are_never_archived(settings, store, redis):
    await store.create_game(game_id="glive", variant="8p", seed=1, config={"models": ["mock"]}, state="running",
                            seats=[(s, "llm", "mock") for s in range(1, 9)])
    async with store.engine.begin() as c:
        await c.execute(text("UPDATE games SET created_at = now() - interval '90 days', "
                             "ended_at = now() - interval '89 days' WHERE id = 'glive'"))
    assert await store.archive_batch(1) == []


async def test_two_archivers_do_not_pack_a_game_twice(settings, store, redis):
    gids = await _play(settings, store, redis, seeds=(61, 62, 63, 64))
    await _age(store, gids, 40)
    a, b = await asyncio.gather(store.archive_batch(30, 2), store.archive_batch(30, 2))
    got = a + b + await store.archive_batch(30, 10)
    assert sorted(got) == sorted(gids)
    assert await _count(store, "SELECT count(*) FROM game_archive") == 4


# ------------------------------------------------------------------ keyset pages
async def _many_games(store, n=37):
    """n games, several sharing a created_at (ties are where keyset paging goes wrong)."""
    u = await store.create_guest("p")
    base = dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc)
    for i in range(n):
        gid = f"g{i:03d}"
        await store.create_game(game_id=gid, variant="8p", seed=i, config={"models": ["mock"]}, state="over",
                                seats=[(1, "human", None)] + [(s, "llm", "mock") for s in range(2, 9)])
        async with store.engine.begin() as c:
            await c.execute(text("UPDATE games SET created_at = :t WHERE id = :g"),
                            {"t": base + dt.timedelta(minutes=i // 3), "g": gid})
        if i % 4 != 1:
            assert await store.claim_seat(gid, 1, u["id"])
    return u["id"]


async def _pages(fetch, size):
    out, cursor = [], None
    while True:
        page = await fetch(size, cursor)
        out += page
        if len(page) < size:
            return out
        cursor = (page[-1]["created_at"], page[-1]["id"])


async def test_keyset_pages_cover_every_game_once_in_order(store):
    uid = await _many_games(store)
    async with store.engine.connect() as c:
        want = [r[0] for r in (await c.execute(text("SELECT id FROM games ORDER BY created_at DESC, id DESC"))).all()]
        mine = [r[0] for r in (await c.execute(text(
            "SELECT g.id FROM games g JOIN game_seats s ON s.game_id = g.id WHERE s.user_id = :u "
            "ORDER BY g.created_at DESC, g.id DESC"), {"u": uid})).all()]
    for size in (1, 5, 7, 37, 50):
        got = await _pages(lambda n, b: store.recent_games(n, before=b), size)
        assert [g["id"] for g in got] == want, size
        got = await _pages(lambda n, b: store.user_games(uid, n, before=b), size)
        assert [g["id"] for g in got] == mine, size
    assert len(mine) == 28


def test_the_api_cursor_round_trips():
    from werewolf_harness.app.api.main import _cursor_in, _cursor_out
    t = dt.datetime(2026, 5, 1, 12, 0, 0, 123456, tzinfo=dt.timezone(dt.timedelta(hours=8)))
    c = _cursor_out({"created_at": t, "id": "g~1"})
    assert "+" not in c and "/" not in c and "=" not in c
    assert _cursor_in(c) == (t, "g~1")


# ------------------------------------------------------------------ batched adoption
async def test_a_scan_adopts_a_batch_of_orphans_with_their_memory(settings, store, redis):
    s = settings.model_copy(update={"recovery_batch": 3})
    a = _rt(s, store, redis, node="node-a", executor=CountingExecutor(8, delay=0.1))   # slow: games stay live
    await a.start(scan=False)
    games = [await a.create(TableConfig(variant="8p", models=["mock"]), seed=70 + i) for i in range(7)]
    await until(lambda: all(len(g.t.journal) > 4 for g in games), 60)
    await a.shutdown(drain=False)                     # node-a dies holding its leases ...
    await redis.flushdb()                             # ... and they expire
    await asyncio.sleep(0.5)                          # a commit that was in flight lands (or not) first
    mem = {g.id: await store.agent_memory(g.id) for g in games}

    gate = GatedExecutor()                            # node-b answers nothing until we have looked
    b = _rt(s, store, redis, node="node-b", executor=gate)
    await b.start(scan=False)
    assert await b.scan() == 7
    adopted = {g.id: b.actors[g.id] for g in games}
    for gid, actor in adopted.items():
        assert actor.t.recoveries == 1 and actor.t.memory == mem[gid]
        assert await redis.get(f"lease:{gid}") == "node-b"
    assert await b.scan() == 0                       # all held now
    gate.release()
    await until(lambda: all(x.t.state == "over" for x in adopted.values()), 90)
    await b.shutdown()
    v = await verify(store)
    assert v["checked"] == 7 and v["consistent_rate"] == 1.0, v


async def test_claim_live_skips_games_that_ended(store):
    for gid, st in (("ga", "running"), ("gb", "over")):
        await store.create_game(game_id=gid, variant="8p", seed=1, config={"models": ["mock"]}, state=st,
                                seats=[(1, "llm", "mock")])
    got = await store.claim_live(["ga", "gb", "gzz"], "n1")
    assert list(got) == ["ga"] and got["ga"][0] == 1 and got["ga"][1]["seed"] == 1


# ------------------------------------------------------------------ migration
async def test_migration_j1_down_restores_archived_journals_and_up_again(settings, store, redis):
    (gid,) = await _play(settings, store, redis, seeds=(81,))
    before = await store.journal(gid)
    await _age(store, [gid], 40)
    assert await store.archive_batch(30) == [gid]
    await store.engine.dispose()                     # no pooled connection holds a plan on the old tables

    await asyncio.to_thread(migrate.downgrade, DB_URL, "0001")
    s = Store.connect(DB_URL, pool_size=2)
    try:
        async with s.engine.connect() as c:
            assert await c.scalar(text("SELECT to_regclass('game_archive')")) is None
            assert await c.scalar(text("SELECT to_regclass('game_seats_user_idx')")) is not None
            cons = await c.scalar(text("SELECT count(*) FROM pg_constraint WHERE conname = 'journal_game_id_request_id_key'"))
            assert cons == 1
            rows = (await c.execute(text("SELECT seq, actor, task, action, forced FROM journal WHERE game_id = :g "
                                         "ORDER BY seq"), {"g": gid})).all()
        assert [(r.seq, r.actor, r.task, r.action, r.forced) for r in rows] == \
            [(r.seq, r.actor, r.task, r.action, r.forced) for r in before]
    finally:
        await s.close()
        await asyncio.to_thread(migrate.upgrade, DB_URL)
    s = Store.connect(DB_URL, pool_size=2)
    try:
        async with s.engine.connect() as c:
            names = {r[0] for r in (await c.execute(text("SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"))).all()}
            invalid = await c.scalar(text("SELECT count(*) FROM pg_index WHERE NOT indisvalid"))
        assert {"games_created_idx", "game_seats_user_time_idx", "journal_request_uidx", "games_archivable_idx"} <= names
        assert "game_seats_user_idx" not in names and invalid == 0
        assert await s.journal(gid) == before         # back in the hot table, still readable
    finally:
        await s.close()
