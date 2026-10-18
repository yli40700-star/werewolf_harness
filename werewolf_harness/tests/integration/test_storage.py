"""P1: the database enforces what the runtime relies on."""
from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text

from werewolf_harness.app.storage import DuplicateStep, Fenced


async def _game(store, gid="g1"):
    await store.create_game(game_id=gid, variant="8p", seed=5, config={"models": ["mock"]}, state="running",
                            seats=[(s, "human" if s == 3 else "llm", None if s == 3 else "mock") for s in range(1, 9)])
    return await store.bump_fencing(gid, "node-a")


async def test_a_step_is_written_once(store):
    tok = await _game(store)
    await store.append("g1", token=tok, seq=0, actor=1, task="night", action={"name": "x"}, forced=False)
    with pytest.raises(DuplicateStep):
        await store.append("g1", token=tok, seq=0, actor=1, task="night", action={"name": "y"}, forced=False)
    rows = await store.journal("g1")
    assert [r.action for r in rows] == [{"name": "x"}]


async def test_concurrent_writers_of_the_same_step_one_wins(store):
    tok = await _game(store)
    results = await asyncio.gather(*[
        store.append("g1", token=tok, seq=0, actor=1, task="night", action={"name": f"v{i}"}, forced=False)
        for i in range(8)], return_exceptions=True)
    assert sum(r is None for r in results) == 1
    assert all(isinstance(r, DuplicateStep) for r in results if r is not None)


async def test_a_retried_request_lands_once(store):
    tok = await _game(store)
    await store.append("g1", token=tok, seq=0, actor=3, task="vote", action={"name": "vote"}, forced=False, request_id="r-1")
    with pytest.raises(DuplicateStep):   # same click, next seq
        await store.append("g1", token=tok, seq=1, actor=3, task="vote", action={"name": "vote"}, forced=False, request_id="r-1")
    hit = await store.step_for_request("g1", "r-1")
    assert hit is not None and hit.seq == 0


async def test_a_stale_fencing_token_cannot_write(store):
    old = await _game(store)
    new = await store.bump_fencing("g1", "node-b")          # someone else took the lease
    assert new == old + 1
    with pytest.raises(Fenced):
        await store.append("g1", token=old, seq=0, actor=1, task="night", action={}, forced=False)
    await store.append("g1", token=new, seq=0, actor=1, task="night", action={}, forced=False)
    assert len(await store.journal("g1")) == 1


async def test_agent_memory_is_written_with_its_step_and_read_back_latest(store):
    tok = await _game(store)
    await store.append("g1", token=tok, seq=0, actor=2, task="speak", action={}, forced=False,
                       agent_states=[(2, {"5": {"suspicion": 0.7}}, "watch 5")])
    await store.append("g1", token=tok, seq=1, actor=2, task="vote", action={}, forced=False,
                       agent_states=[(2, {"5": {"suspicion": 0.9}}, "vote 5")])
    mem = await store.agent_memory("g1")
    assert mem == {2: ({"5": {"suspicion": 0.9}}, "vote 5")}


async def test_a_fenced_write_leaves_no_agent_memory_behind(store):
    old = await _game(store)
    await store.bump_fencing("g1", "node-b")
    with pytest.raises(Fenced):
        await store.append("g1", token=old, seq=0, actor=2, task="speak", action={}, forced=False,
                           agent_states=[(2, {"5": {}}, "x")])
    assert await store.agent_memory("g1") == {}


async def test_seat_claim_is_first_come(store):
    await _game(store)
    a = await store.create_guest("a")
    b = await store.create_guest("b")
    assert await store.claim_seat("g1", 3, a["id"])
    assert await store.claim_seat("g1", 3, a["id"])          # same person again: fine
    assert not await store.claim_seat("g1", 3, b["id"])      # someone else: no
    assert not await store.claim_seat("g1", 4, a["id"])      # not a human seat
    games = await store.user_games(a["id"])
    assert [g["id"] for g in games] == ["g1"]


async def test_live_games_and_user_games_use_their_indexes(store):
    """The two hot lookups are index scans once the table is not tiny."""
    await _game(store)
    async with store.engine.begin() as c:
        await c.execute(text("SET LOCAL enable_seqscan = off"))
        await c.execute(text("SET LOCAL enable_bitmapscan = off"))
        plan = "\n".join(r[0] for r in (await c.execute(text(
            "EXPLAIN SELECT id FROM games WHERE state IN ('waiting','running') ORDER BY created_at"))).all())
        assert "games_live_idx" in plan
        # "my games" reads the newest seats straight off the index: no sort node (j1)
        plan = "\n".join(r[0] for r in (await c.execute(text(
            "EXPLAIN SELECT game_id FROM game_seats WHERE user_id = 'u_x' "
            "ORDER BY game_created_at DESC NULLS LAST, game_id DESC LIMIT 20"))).all())
        assert "game_seats_user_time_idx" in plan and "Sort" not in plan
        plan = "\n".join(r[0] for r in (await c.execute(text(
            "EXPLAIN SELECT id FROM games ORDER BY created_at DESC, id DESC LIMIT 50"))).all())
        assert "games_created_idx" in plan and "Sort" not in plan


async def test_call_stats(store):
    await store.record_calls([
        {"game_id": "g1", "seat": 1, "seq": 0, "model": "m", "status": "ok", "input_tokens": 1000,
         "cached_tokens": 800, "output_tokens": 50, "latency_ms": 900},
        {"game_id": "g1", "seat": 2, "seq": 1, "model": "m", "status": "ok", "input_tokens": 1000,
         "cached_tokens": 0, "output_tokens": 50, "latency_ms": 100, "fallback_reason": "timeout"},
    ])
    s = await store.call_stats("g1")
    assert s["calls"] == 2 and s["cache_hit_rate"] == 0.4 and s["fallbacks"] == 1
