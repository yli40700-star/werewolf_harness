"""P5: a turn is one trace -- turn -> invoke_agent -> chat, and turn -> commit."""
from __future__ import annotations

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from werewolf_harness.app.llm.gateway import Hub
from werewolf_harness.app.obs import tracing
from werewolf_harness.app.runtime.table import TableConfig

from .conftest import REDIS_URL
from .helpers import until
from .test_runtime import _rt

EXP = InMemorySpanExporter()
tracing.setup("test", exporter=EXP)


async def test_every_model_call_hangs_under_its_turn(settings, store, redis):
    EXP.clear()
    rt = _rt(settings, store, redis, hub=Hub(REDIS_URL, sim_sleep=False))
    await rt.start(scan=False)
    a = await rt.create(TableConfig(variant="8p", models=["sim-down"], fallbacks=["sim-fast"]), seed=61)
    await until(lambda: a.t.state == "over", 90)
    await rt.shutdown()
    spans = EXP.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    turns = [s for s in spans if s.name == "turn"]
    chats = [s for s in spans if s.name.startswith("chat ")]
    assert len(turns) == len(a.t.journal)
    assert chats
    for c in chats:
        agent = by_id[c.parent.span_id]
        assert agent.name == "invoke_agent"
        assert by_id[agent.parent.span_id].name == "turn"
        assert c.context.trace_id == agent.context.trace_id
    failed = [c for c in chats if c.name == "chat sim-down"]
    ok = [c for c in chats if c.name == "chat sim-fast"]
    assert failed and all(c.attributes.get("error.type") == "upstream_error" or not c.status.is_ok for c in failed)
    assert ok and ok[0].attributes["gen_ai.usage.input_tokens"] > 0
    commits = [s for s in spans if s.name == "commit"]
    assert all(by_id[c.parent.span_id].name == "turn" for c in commits)
