"""Engine versions across a restart: a game keeps the rules it was created with.

The scenario the version exists for: a game is created on a node running engine v1,
the code is upgraded to v2 (a step-changing rule: 平票 PK), the service restarts and
recovers the game. It must come back as the same world -- replayed under v1, not v2 --
and keep playing v1 rules to the end. And the reverse, a rolling deploy: a node still
on v1 must not adopt a game a v2 node created.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from werewolf_harness import envs
from werewolf_harness.app.runtime import table as table_mod
from werewolf_harness.app.runtime.runtime import Runtime
from werewolf_harness.app.runtime.table import Table, TableConfig
from werewolf_harness.arena import Record, Turn
from werewolf_harness.envs import werewolf as ww
from werewolf_harness.scripts import compat_check as cc

from .helpers import GatedExecutor, until

CORPUS = pathlib.Path(__file__).resolve().parents[1] / "corpus"


def _rt(settings, store, redis, node, **kw):
    return Runtime(settings.model_copy(update={"node_id": node}), store, redis, **kw)


def _old_code(mp: pytest.MonkeyPatch) -> None:
    """Make this process the code from before v2 existed: current = v1, v2 unknown."""
    for mod in (ww, envs, table_mod):
        mp.setattr(mod, "ENGINE_VERSION", 1)


async def test_a_v1_game_recovers_under_v1_after_the_engine_moves_to_v2(settings, store, redis):
    # real-base-g1007: day one ends in a tie with the 18th row. v1: nobody is exiled and
    # night falls (4 号 / night). v2 would hold a revote instead (1 号 / vote).
    raw = json.loads((CORPUS / "real-base-g1007.json").read_text())
    rows = [Record(r["actor"], r["task"], r["action"], r["forced"]) for r in raw["journal"][:18]]
    save = {"game": {"variant": "8p", "seed": raw["seed"], "playbook": "static", "models": ["mock"]},
            "journal": [r.__dict__ for r in rows]}

    with pytest.MonkeyPatch.context() as mp:                    # ---- node A runs the old code
        _old_code(mp)
        gated = GatedExecutor()                                 # A never gets an answer back: frozen at row 18
        a_rt = _rt(settings, store, redis, "node-a", executor=gated)
        await a_rt.start(scan=False)
        a = await a_rt.import_log(save)
        gid = a.id
        assert a.t.engine_version == 1
        await a_rt.shutdown(drain=True)                         # the service goes down for the upgrade
    assert ww.ENGINE_VERSION == 2                               # ---- the code is now v2

    row = await store.get_game(gid)
    assert row["engine_version"] == 1 and row["config"]["engine_version"] == 1
    assert len(await store.journal(gid)) == 18

    # what the world must be: the same journal under v1 -- and under v2 it is not even replayable
    want = Table("ref", raw["seed"], TableConfig(variant="8p", models=["mock"], engine_version=1))
    Runtime.replay(want, rows)
    assert want.turn == Turn(4, "night")
    with pytest.raises(ValueError, match="diverged"):
        Runtime.replay(Table("v2", raw["seed"], TableConfig(variant="8p", models=["mock"], engine_version=2)),
                       rows + [Record(4, "night", {"name": "night_skip", "args": {}})])

    b_rt = _rt(settings, store, redis, "node-b")                # ---- the restarted service
    await b_rt.start(scan=False)
    b = await b_rt.adopt(gid)
    assert b is not None and b.t.engine_version == 1 and b.t.env.rules.tie_revote is False
    assert b.t.turn == Turn(4, "night") and b.t.fingerprint() == want.fingerprint()
    await until(lambda: b.t.state == "over", 90)

    # it played on under v1: the tie stood, no revote ever happened, and the whole game
    # (old rows + the rows written after the restart) replays under v1
    assert b.t.env._exiles[1] is None and b.t.env._tie_ballots == {}
    stored = await store.journal(gid)
    assert [(r.actor, r.task, r.action) for r in stored[:18]] == [(r.actor, r.task, r.action) for r in rows]
    assert len(stored) > 18
    final = await store.get_game(gid)
    game = cc.StoredGame(gid, "8p", "static", final["seed"], final["engine_version"],
                         [Record(r.actor, r.task, r.action, r.forced) for r in stored], final["outcome"])
    rep = cc.check_game(game, against=2, fingerprint=True)
    assert rep.recorded == "identical" and rep.current == "diverged" and rep.detail["current"]["step"] == 18

    # a game created now is a v2 game, in the column and in the config a worker reads
    fresh = await b_rt.create(TableConfig(variant="8p", models=["mock"]), seed=4)
    frow = await store.get_game(fresh.id)
    assert frow["engine_version"] == 2 and frow["config"]["engine_version"] == 2
    await b_rt.shutdown()


async def test_a_node_on_older_code_does_not_adopt_a_newer_game(settings, store, redis):
    new_rt = _rt(settings, store, redis, "node-new")
    await new_rt.start(scan=False)
    g = await new_rt.create(TableConfig(variant="8p", models=["mock"], human_seats=[1], turn_seconds=None), seed=2)
    await g.start()
    await new_rt.shutdown(drain=True)

    with pytest.MonkeyPatch.context() as mp:                    # a node not yet upgraded scans
        _old_code(mp)
        old_rt = _rt(settings, store, redis, "node-old")
        await old_rt.start(scan=True)
        assert g.id not in old_rt.actors and "v2" in old_rt.incompatible[g.id]
        assert await old_rt.leases.holder(g.id) is None        # left free for a node that can replay it
        await old_rt.shutdown()

    up_rt = _rt(settings, store, redis, "node-up")
    await up_rt.start(scan=True)
    assert g.id in up_rt.actors and up_rt.actors[g.id].t.engine_version == 2
    await up_rt.shutdown()


async def test_batched_adoption_rebuilds_each_game_under_its_own_version(settings, store, redis):
    """scan() adopts orphans `recovery_batch` at a time from one claim_live UPDATE: the row
    it returns must carry engine_version, or every game in the batch would come back as v1."""
    s = settings.model_copy(update={"recovery_batch": 4})
    a_rt = _rt(s, store, redis, "node-a", executor=GatedExecutor())       # frozen: games stay live
    await a_rt.start(scan=False)
    made = {}
    for i, v in enumerate((1, 2, 1, 2, 2)):
        g = await a_rt.create(TableConfig(variant="8p", models=["mock"], engine_version=v), seed=50 + i)
        made[g.id] = v
    await a_rt.shutdown(drain=True)

    b_rt = _rt(s, store, redis, "node-b", executor=GatedExecutor())
    await b_rt.start(scan=False)
    assert await b_rt.scan() == len(made)
    assert {gid: b_rt.actors[gid].t.engine_version for gid in made} == made
    assert all(b_rt.actors[gid].t.env.rules.tie_revote is (v == 2) for gid, v in made.items())
    await b_rt.shutdown()
