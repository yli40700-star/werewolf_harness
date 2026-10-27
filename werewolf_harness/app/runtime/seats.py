"""Building one turn's player from plain settings -- shared by the actor's own
executor and the model workers.

`Table.seat_settings` (table.py) resolves everything a seat's player depends on --
which model answers, the cascade's strong tier and policy, budgets, guard, playbook,
language -- into a JSON dict. The actor puts it in the job; `build_player` turns it
back into an LLMPlayer. A worker (app/llm/queue.py) therefore needs no game row, no
journal and no engine to run a turn: the view and the action spec come in the job
too, computed by the actor.
"""
from __future__ import annotations

import os

from ...players import GuardStack, LLMPlayer, MockClient, RandomPlayer, RecoveryPolicy, TokenBudget, TransportBudget
from ...players.llm.cascade import CascadePolicy
from ...players.providers import profile
from ...players.providers.mock import cascade_mock
from ...players.providers.recorded import RecordingClient, recorder_for


def guard_for(spec: str) -> GuardStack:
    """The guard layers named in a table config ("L1+L2+L3"; "无" or "" = none)."""
    return GuardStack(tuple(x for x in spec.split("+") if x and x != "无"))


def direct_client(model: str, seed: int, env_keys: dict | None):
    """A model client without the gateway (no Hub in this process)."""
    if model == "mock":
        return MockClient(seed=seed)
    offline = cascade_mock(model, seed)                # the cascade's offline tiers
    if offline is not None:
        return offline
    from ...server.service import _client as _provider_client
    return _provider_client(model, seed, env_keys or {})


def cassette_writer(game_id: str, seed: int, record: bool, config: dict):
    """A game's cassette writer, or None when the game is not being recorded.

    A file per game, appended to by whichever process runs the turn (the actor's
    node, a Streams worker, the node that adopted the game after a crash): one line
    per call, so lines from several processes interleave but never tear. The writer
    is cached per process so the message blobs are written once per process.
    """
    if not (record or os.environ.get("LLM_RECORD") == "1"):
        return None
    path = os.path.join(os.environ.get("LLM_RECORD_DIR", "runs/cassettes"), f"{game_id}.cassette.jsonl")
    return recorder_for(path, {"game_id": game_id, "seed": seed, "config": config, "online": True})


def build_player(game_id: str, seat: int, version: int | None, st: dict, *, hub=None, env_keys: dict | None = None,
                 guard: GuardStack, transport: TransportBudget, tokens: TokenBudget, client_factory=None,
                 direct=None, on_step=None):
    """One turn's player from `Table.seat_settings`. The actor's own executor (with the
    table's shared guard and budgets) and a model worker (with fresh ones, from the
    settings in the job) both build it here, so the two cannot drift apart.

    Clients go through `hub` (the gateway) when there is one, else `direct(model, seat)`
    (default: direct_client); `client_factory(model, seat)` overrides both (tests).
    """
    model = st["model"]
    if model == "script":
        return RandomPlayer(st["seed"] * 100 + seat)

    def seat_client(m: str, tier: str | None = None):
        if client_factory is None and hub is not None:
            client = hub.client(m, st["seed"] * 100 + seat, game_id=game_id, seat=seat, seq=version,
                                max_game_tokens=st["max_tokens"], fallbacks=st["fallbacks"])
        else:
            make = client_factory or direct or (lambda mm, s: direct_client(mm, st["seed"] * 100 + s, env_keys))
            client = make(m, seat)
        writer = cassette_writer(game_id, st["seed"], st.get("record", False), st.get("config") or {})
        if writer is not None:
            # One player object per turn here, so its own turn counter is always 0: the
            # journal version it was asked at goes in front of it in the key. A cascade's
            # strong tier shares that numbering (recorded.RecordingClient, `tier`).
            client = RecordingClient(client, writer, seat=seat, turn_prefix="-" if version is None else version,
                                     tier=tier)
        return client

    client = seat_client(model)
    timeout = profile(model)["timeout_s"]
    tiers: dict = {}
    if st.get("cascade"):
        strong = st["strong"]
        tiers = {"strong_client": seat_client(strong, tier="strong"), "cascade": CascadePolicy.from_dict(st["cascade"])}
        timeout = max(timeout, profile(strong)["timeout_s"])   # one policy serves both tiers
    # The gateway tries each model of the chain once, with no pause between. So the
    # seat retries the whole chain, with backoff: without it one 503 from the only model
    # turned the turn into the default action (faults/report.md, model_faults).
    pol = RecoveryPolicy(timeout_s=timeout, backoff_s=1.0 if hub is not None else RecoveryPolicy.backoff_s)
    budget = transport if (st["abort_on_outage"] or hub is None) else None
    return LLMPlayer(client, guard=guard, policy=pol, transport_budget=budget, token_budget=tokens,
                     playbook=st["playbook"] != "none", on_step=on_step, lang=st["lang"], **tiers)
