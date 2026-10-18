"""P4: the model layer -- fallbacks, breakers, token limits, budgets, accounting."""
from __future__ import annotations

import pytest

from werewolf_harness.app.llm.gateway import GatewayClient, Hub, classify
from werewolf_harness.app.runtime.table import TableConfig
from werewolf_harness.players.budget import BudgetExhausted
from werewolf_harness.players.providers.base import LLMClient, LLMResponse, ProviderError

from .conftest import REDIS_URL
from .helpers import until
from .test_runtime import _rt

MSG = [{"role": "system", "content": "x" * 400}, {"role": "user", "content": "hi"}]


class Fake(LLMClient):
    def __init__(self, name, fail=None):
        self.name, self.fail, self.calls = name, fail, 0

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0):
        self.calls += 1
        if self.fail:
            raise self.fail
        return LLMResponse(text="ok", prompt_tokens=100, completion_tokens=10, model=self.name,
                           raw={"usage": {"prompt_tokens_details": {"cached_tokens": 60}}})


@pytest.fixture
def hub(redis):
    h = Hub(REDIS_URL, breaker_kw={"min_calls": 2, "threshold": 0.5, "cooldown_s": 60}, limiter_wait_s=0.2)
    yield h


def _gw(hub, *clients, game="g", cap=None):
    return GatewayClient(hub, [(c.name, c) for c in clients], game_id=game, seat=1, seq=0, max_game_tokens=cap)


def test_failure_kinds_are_told_apart():
    assert classify(ProviderError("x", status=429)) == "rate_limited"
    assert classify(ProviderError("model call timed out after 90s", status=None)) == "timeout"
    assert classify(ProviderError("bad gateway", status=502)) == "upstream_error"
    assert classify(ProviderError("maximum context length exceeded", status=400)) == "context_length"
    assert classify(ProviderError("invalid parameter", status=400)) is None


async def test_a_failing_model_falls_back_and_the_record_says_why(hub):
    a, b = Fake("m-a", ProviderError("upstream 503", status=503)), Fake("m-b")
    r = _gw(hub, a, b).chat(MSG)
    assert r.model == "m-b"
    recs = hub.sink.drain()
    assert [(x.model, x.status, x.fallback_reason) for x in recs] == [
        ("m-a", "error", "upstream_error"), ("m-b", "ok", "upstream_error")]
    assert recs[1].cached_tokens == 60 and recs[1].cost_usd > 0


async def test_an_open_breaker_skips_the_model_without_calling_it(hub):
    a, b = Fake("m-a", ProviderError("upstream 503", status=503)), Fake("m-b")
    gw = _gw(hub, a, b)
    gw.chat(MSG)
    gw.chat(MSG)                                   # two failures -> open
    assert hub.breaker("m-a").state == "open"
    calls = a.calls
    gw.chat(MSG)
    assert a.calls == calls                        # not even tried
    assert hub.sink.drain()[-2].fallback_reason == "breaker_open"


async def test_an_open_breaker_on_the_only_model_still_lets_the_call_through(hub):
    # nothing to fall back to: refusing would hand every turn to the rule seat
    a = Fake("m-solo", ProviderError("upstream 503", status=503))
    gw = _gw(hub, a)
    for _ in range(2):
        with pytest.raises(ProviderError):
            gw.chat(MSG)
    assert hub.breaker("m-solo").state == "open"
    a.fail = None
    assert gw.chat(MSG).model == "m-solo" and a.calls == 3


async def test_wallet_and_key_errors_stop_immediately(hub):
    a, b = Fake("m-a", ProviderError("insufficient balance", status=402)), Fake("m-b")
    with pytest.raises(ProviderError):
        _gw(hub, a, b).chat(MSG)
    assert b.calls == 0


async def test_everything_down_is_a_503_for_the_seat_to_handle(hub):
    with pytest.raises(ProviderError) as e:
        _gw(hub, Fake("m-a", ProviderError("x", status=500)), Fake("m-b", ProviderError("y", status=429))).chat(MSG)
    assert e.value.status == 503


async def test_the_game_budget_is_shared_and_enforced(hub):
    gw = _gw(hub, Fake("m-a"), game="gb", cap=200)
    gw.chat(MSG)
    gw.chat(MSG)                                   # 2 x 110 = 220 >= 200
    with pytest.raises(BudgetExhausted):
        _gw(hub, Fake("m-a"), game="gb", cap=200).chat(MSG)     # a second client (another worker) sees it too


async def test_token_bucket_refuses_what_does_not_fit_and_refunds_the_unused_part(redis):
    h = Hub(REDIS_URL, tpm={"m-a": 6000}, limiter_wait_s=0.1)
    assert h.limiter.reserve("m-a", 5000, 0.1)
    assert not h.limiter.reserve("m-a", 5000, 0.1)          # 1000 left; would wait ~40 s for the rest
    h.limiter.settle("m-a", 5000, 1000)                     # the call used 1000 of its 5000
    assert h.limiter.reserve("m-a", 4500, 0.1)


async def test_a_game_on_a_dead_provider_falls_back_and_finishes(settings, store, redis):
    hub = Hub(REDIS_URL, sim_sleep=False)
    rt = _rt(settings, store, redis, hub=hub)
    await rt.start(scan=False)
    a = await rt.create(TableConfig(variant="8p", models=["sim-down"], fallbacks=["sim-fast"]), seed=41)
    await until(lambda: a.t.state == "over", 90)
    await rt.flush_calls()
    stats = await store.call_stats(a.id)
    assert stats["fallbacks"] > 0 and stats["calls"] > 0
    assert a.t.health()["forced_rate"] < 0.05                  # the fallback model played, not the harness
    await rt.shutdown()


async def test_with_every_model_down_the_rule_seat_keeps_the_game_moving(settings, store, redis):
    hub = Hub(REDIS_URL, sim_sleep=False)
    rt = _rt(settings, store, redis, hub=hub)
    await rt.start(scan=False)
    a = await rt.create(TableConfig(variant="8p", models=["sim-down"]), seed=42)
    await until(lambda: a.t.state in ("over", "aborted"), 120)
    assert a.t.state == "over"
    assert a.t.health()["forced_rate"] == 1.0
    await rt.shutdown()


async def test_routing_sends_decisions_to_the_strong_model(settings, store, redis):
    hub = Hub(REDIS_URL, sim_sleep=False)
    rt = _rt(settings, store, redis, hub=hub)
    await rt.start(scan=False)
    a = await rt.create(TableConfig(variant="8p", models=["x"], routing={"strong": "sim-strong", "cheap": "sim-cheap"}),
                        seed=43)
    await until(lambda: a.t.state == "over", 90)
    await rt.flush_calls()
    from sqlalchemy import text
    async with store.engine.connect() as c:
        rows = (await c.execute(text("SELECT j.task, l.model FROM llm_calls l JOIN journal j "
                                     "ON j.game_id = l.game_id AND j.seq = l.seq WHERE l.game_id = :g"),
                                {"g": a.id})).all()
    by_task = {}
    for task, model in rows:
        by_task.setdefault(task, set()).add(model)
    assert by_task["night"] == {"sim-strong"} and by_task["speak"] == {"sim-cheap"}
    await rt.shutdown()
