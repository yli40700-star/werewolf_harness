from werewolf_harness.app.runtime.table import TableConfig
from werewolf_harness.scripts.verify_replay import verify

from .helpers import until
from .test_runtime import _rt


async def test_finished_games_replay_to_the_same_outcome(settings, store, redis):
    rt = _rt(settings, store, redis)
    await rt.start(scan=False)
    games = [await rt.create(TableConfig(variant="8p", models=["mock"]), seed=s) for s in (31, 32, 33)]
    await until(lambda: all(g.t.state == "over" for g in games), 90)
    r = await verify(store)
    assert r["checked"] == 3 and r["consistent_rate"] == 1.0, r
    await rt.shutdown()


async def test_open_turns_are_asked_together_and_the_journal_still_replays(settings, store, redis):
    rt = _rt(settings, store, redis)
    await rt.start(scan=False)
    on = await rt.create(TableConfig(variant="12p", models=["mock"]), seed=41)
    off = await rt.create(TableConfig(variant="12p", models=["mock"], parallel=False), seed=42)
    await until(lambda: on.t.state == "over" and off.t.state == "over", 90)
    assert on.t.asked_ahead > 0 and off.t.asked_ahead == 0
    assert on.t.health()["asked_ahead"] == on.t.asked_ahead
    r = await verify(store)
    assert r["consistent_rate"] == 1.0, r
    await rt.shutdown()
