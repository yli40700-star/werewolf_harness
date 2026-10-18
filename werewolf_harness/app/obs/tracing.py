"""OpenTelemetry traces, one per turn, following the GenAI semantic conventions.

    turn (game_id, seat, task, version)            -- opened when the turn is asked, closed at commit
    ├─ invoke_agent (seat, model)                  -- the whole ReAct turn, on a worker thread/process
    │   ├─ chat {model}   gen_ai.request.model, gen_ai.usage.input_tokens / output_tokens,
    │   │                 cached tokens, fallback reason, error
    │   └─ chat {fallback model} ...
    └─ commit (seq)                                -- the journal write

The turn span crosses three boundaries: the actor's mailbox (kept on the actor),
a thread (copied contextvars) and, with the queue executor, a process (W3C
traceparent in the job's fields). Exported over OTLP/HTTP when
OTEL_EXPORTER_OTLP_ENDPOINT is set (Jaeger / Tempo in deploy/), otherwise
tracing is a no-op.
"""
from __future__ import annotations

import os

from opentelemetry import context as otctx
from opentelemetry import trace
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

tracer = trace.get_tracer("werewolf_harness")
_configured = False


def setup(service: str = "werewolf-api", exporter=None) -> TracerProvider | None:
    """Install a provider. `exporter` is for tests (an in-memory exporter)."""
    global _configured
    if _configured and exporter is None:
        return None
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if exporter is None and not endpoint:
        return None
    provider = TracerProvider(resource=Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", service)}))
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _configured = True
    return provider


def get_tracer():
    return trace.get_tracer("werewolf_harness")


def to_carrier(ctx) -> dict:
    carrier: dict = {}
    inject(carrier, context=ctx)
    return carrier


def from_carrier(carrier: dict):
    return extract(carrier)


def attach(ctx):
    return otctx.attach(ctx) if ctx is not None else None


def detach(token) -> None:
    if token is not None:
        otctx.detach(token)
