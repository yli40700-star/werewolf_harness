"""One table's state and everything a client may read about it.

This is the part of the old `server.service.Game` that is not about threads:
dealing seats, building the model seats, turning steps into public news, and
gating what each reader sees. The actor (actor.py) is the only thing that
changes a Table; readers call the `*_view` methods.

What a reader may see is decided in exactly one place, `may_see_all`, and every
event and summary goes through `public_action` / `public_actor` before it leaves.
Two leaks the old server had are closed here and pinned by tests:

  * the event stream carried every raw action -- a night kill and a seat's
    private `thought` went to anyone holding the SSE URL;
  * the "who is acting" line named the night actor -- the first seat to act at
    night is the seer, so a spectator could read roles off the order.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

from ...arena import Record, StepLedger, Terminal, Turn
from ...envs import BASELINE_ENGINE_VERSION, ENGINE_VERSION, VARIANTS, Werewolf
from ...players import GuardStack, LLMPlayer, MockClient, RandomPlayer, RecoveryPolicy, TokenBudget, TransportBudget
from ...players.human import Pending
from ...players.llm.cascade import CascadePolicy
from ...players.providers import profile
from ...players.providers.mock import cascade_mock
from ...players.providers.recorded import RecordingClient, recorder_for
from ...server.service import _client as _provider_client

# Tasks whose actor and (thought-free) action the whole table sees.
PUBLIC_TASKS = frozenset({"speak", "vote", "campaign", "campaign_vote", "last_words", "hunter_shoot", "badge", "review"})


def public_action(action: Any) -> Any:
    """A seat's reasoning is its own: `thought` never travels to another reader."""
    if not isinstance(action, dict):
        return action
    args = {k: v for k, v in (action.get("args") or {}).items() if k != "thought"}
    return {**action, "args": args}


@dataclass
class TableConfig:
    variant: str = "12p"
    models: list[str] = field(default_factory=lambda: ["mock"])
    guard: str = "L1+L2+L3"
    playbook: str = "static"
    human_seats: list[int] = field(default_factory=list)
    turn_seconds: float | None = 300.0
    max_tokens: int = 6_000_000
    xray: bool = False
    # model layer (app/llm): try these after the seat's model fails, in order
    fallbacks: list[str] = field(default_factory=list)
    # route by task instead of by seat: {"strong": m1, "cheap": m2, "strong_tasks": [...]}
    routing: dict | None = None
    # Model cascade (players/llm/cascade.py, docs/adr/0014): every AI seat answers on the cheap
    # model and moves to the strong one inside a turn when the cheap reply fails validation
    # and/or on decisive turns. {"cheap": m1, "strong": m2, "mode": "on-fail"|"decisive"|"both",
    # "decisive": [...] (optional), "low_confidence": bool (optional)}. Supersedes `routing`
    # (which is `mode="decisive"` without the in-turn retry); a stored routing config still works.
    cascade: dict | None = None
    lane: str | None = None           # queue lane override (e.g. "offline" for batch experiments)
    lang: str = "zh"                  # what the AI seats speak: "zh" (简体中文) or "en"
    # ask every open model turn at once (votes, the review, independent night roles)
    parallel: bool = True
    # Experiments abandon a game whose models keep failing (a sample the harness played
    # is not a sample). A table with people at it keeps going on rule seats instead.
    abort_on_outage: bool = False
    # The rules the game is played under, fixed when it is created and never changed:
    # recovery and every later replay rebuild the world with this version, whatever
    # code is deployed (envs/werewolf.py, docs/adr/0010). None = the current version,
    # resolved by Table; a stored config without the key predates versioning.
    engine_version: int | None = None
    # Keep every model call of this game (request + reply/error) in a cassette under
    # LLM_RECORD_DIR (players/providers/recorded.py). LLM_RECORD=1 turns it on for all games.
    record: bool = False

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> TableConfig:
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        known.setdefault("engine_version", BASELINE_ENGINE_VERSION)
        return cls(**known)


@dataclass
class Event:
    seq: int
    kind: str
    data: dict
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"seq": self.seq, "kind": self.kind, "ts": round(self.ts, 3), **self.data}


# Turns whose answer is spoken text and may be streamed as a draft, and who may
# watch it: "public" (everyone, like the turn itself), "wolves" (the pack channel).
DRAFT_AUDIENCE = {"speak": "public", "last_words": "public", "campaign": "public", "review": "public",
                  "pack_chat": "wolves"}
DRAFT_LOG_MAX = 4000


@dataclass
class Draft:
    """A speech still being generated: shown, not yet said. Never persisted."""
    id: str
    seat: int
    seq: int                         # the journal version the turn was asked at
    task: str
    audience: str
    run: str
    attempt: int
    asked_at: float
    text: str = ""
    first_at: float | None = None
    last_at: float | None = None


@dataclass
class DraftEvent:
    """One draft message. Its own sequence (dseq), outside the journal-derived `events`:
    drafts are not replayed, so they must not shift the seq a reconnecting client
    resumes from. `at` = len(events) when emitted, which orders it against them."""
    dseq: int
    kind: str                        # draft | draft_commit | draft_retract | draft_reset
    audience: str
    at: int
    data: dict
    ts: float = field(default_factory=time.time)

    def to_dict(self, epoch: int) -> dict:
        return {"dseq": self.dseq, "kind": self.kind, "epoch": epoch, "ts": round(self.ts, 3), **self.data}


class Table:
    def __init__(self, game_id: str, seed: int, cfg: TableConfig, env_keys: dict | None = None,
                 created: float | None = None, hub=None):
        self.id = game_id
        self.seed = seed
        self.cfg = cfg
        self.env_keys = env_keys or {}
        self.hub = hub                                   # app.llm.gateway.Hub, or None for direct clients
        self.created = created or time.time()
        self.n = VARIANTS[cfg.variant].num_players
        self.human_seats = [s for s in cfg.human_seats if 1 <= s <= self.n]
        self.models = list(cfg.models) or ["mock"]
        self.model_label = self.models[0] if len(self.models) == 1 else "混合"
        if cfg.engine_version is None:
            cfg.engine_version = ENGINE_VERSION          # pinned here, so the stored config carries it
        self.env = Werewolf(cfg.variant, playbook_mode=cfg.playbook, engine_version=cfg.engine_version)
        self.engine_version = self.env.engine_version
        self.ledger = StepLedger(game=game_id)
        self.journal: list[Record] = []
        self.request_ids: dict[str, int] = {}       # client retry key -> seq
        self.events: list[Event] = []
        self.traces: dict[int, list[dict]] = {}
        self.turn: Turn | Terminal | None = None
        self.acting: dict | None = None
        self.pending: Pending | None = None          # a human seat's open turn
        self.pending_seat: int | None = None
        self.memory: dict[int, tuple[dict, str]] = {}   # model seat -> (belief snapshot, note)
        self.state = "waiting" if self.human_seats else "running"
        self.outcome: dict | None = None
        self.error: str | None = None
        self.resumed_from: str | None = None
        self.seat_stats: dict[int, dict] = {s: {"turns": 0, "forced": 0} for s in range(1, self.n + 1)}
        self.stale_results = 0
        self.recoveries = 0
        self.asked_ahead = 0                  # model turns asked before they were due
        self.misses = 0                       # ... whose answer could not be used and was asked again
        # Streaming drafts (docs/adr/0009). Live only: an incarnation (fencing token = epoch)
        # starts with none, and says so with a draft_reset before anything else.
        self.draft_epoch = 0
        self.drafts: dict[str, Draft] = {}
        self.draft_log: list[DraftEvent] = []
        self.draft_base = 0                   # dseq of draft_log[0]
        self.draft_closed: dict[str, dict] = {}       # recently closed: id -> its commit/retract event
        self.draft_timings: list[dict] = []   # per committed/retracted draft: ms to first char, to the end
        self.draft_stats = {"opened": 0, "committed": 0, "retracted": 0, "deltas": 0,
                            "cancelled_turns": 0, "tokens_saved": 0, "tokens_saved_cut": 0,
                            "tokens_saved_skipped": 0}

        # Which model sits where: round-robin over a seeded shuffle (as before), so a
        # mixed table does not always hand seat 1 the same model.
        ai = [s for s in range(1, self.n + 1) if s not in self.human_seats]
        rota = list(self.models)
        random.Random(seed).shuffle(rota)
        self.seat_models: dict[int, str] = {s: rota[i % len(rota)] for i, s in enumerate(ai)}
        if cfg.routing:
            self.seat_models = {s: "route" for s in ai}
            self.models = [cfg.routing["strong"], cfg.routing["cheap"]]
            self.model_label = f"路由 {cfg.routing['strong']}/{cfg.routing['cheap']}"
        self.cascade_policy = None
        if cfg.cascade:
            self.cascade_policy = CascadePolicy.from_mode(cfg.cascade.get("mode", "both"),
                                                          decisive=cfg.cascade.get("decisive"),
                                                          low_confidence=bool(cfg.cascade.get("low_confidence")))
        if self.cascade_policy is not None:
            self.seat_models = {s: "cascade" for s in ai}
            self.models = [cfg.cascade["cheap"], cfg.cascade["strong"]]
            self.model_label = f"级联 {cfg.cascade['cheap']}→{cfg.cascade['strong']}（{self.cascade_policy.mode}）"
        # per game: turns each tier answered, escalations by reason (from AgentResult.meta)
        self.cascade_stats: dict = {"turns": 0, "strong_turns": 0, "calls": 0, "strong_calls": 0, "escalations": {}}

        guards = tuple(x for x in cfg.guard.split("+") if x and x != "无")
        self.guard = GuardStack(guards)
        self.transport = TransportBudget()
        self.tokens = TokenBudget(max_tokens=cfg.max_tokens)

    # ------------------------------------------------------------ seats
    def seat_kind(self, seat: int) -> str:
        if seat in self.human_seats:
            return "human"
        return "script" if self.seat_models.get(seat) == "script" else "llm"

    def seat_rows(self) -> list[tuple[int, str, str | None]]:
        return [(s, self.seat_kind(s), self.seat_models.get(s)) for s in range(1, self.n + 1)]

    STRONG_TASKS = ("vote", "night", "pack_plan", "hunter_shoot", "badge", "campaign_vote")

    def model_for(self, seat: int, task: str | None) -> str:
        m = self.seat_models[seat]
        if m == "route" and self.cfg.routing:
            strong = self.cfg.routing.get("strong_tasks") or self.STRONG_TASKS
            return self.cfg.routing["strong"] if task in strong else self.cfg.routing["cheap"]
        if m == "cascade" and self.cfg.cascade:
            return self.cfg.cascade["cheap"]           # the tier a turn starts on; the loop may escalate
        return m

    def make_player(self, seat: int, client_factory=None, on_step=None, task: str | None = None,
                    version: int | None = None):
        """A fresh player object for one turn. Model seats keep nothing between
        turns: their memory travels in and out with the job (see agents.py)."""
        model = self.model_for(seat, task)
        if model == "script":
            return RandomPlayer(self.seed * 100 + seat)
        client = self._seat_client(model, seat, version, client_factory)
        timeout = profile(model)["timeout_s"]
        tiers: dict = {}
        if self.seat_models.get(seat) == "cascade" and self.cascade_policy is not None:
            strong = self.cfg.cascade["strong"]
            tiers = {"strong_client": self._seat_client(strong, seat, version, client_factory, tier="strong"),
                     "cascade": self.cascade_policy}
            timeout = max(timeout, profile(strong)["timeout_s"])   # one policy serves both tiers
        # The gateway tries each model of the chain once, with no pause between. So the
        # seat retries the whole chain, with backoff: without it one 503 from the only model
        # turned the turn into the default action (faults/report.md, model_faults).
        pol = RecoveryPolicy(timeout_s=timeout, backoff_s=1.0 if self.hub is not None else RecoveryPolicy.backoff_s)
        budget = self.transport if (self.cfg.abort_on_outage or self.hub is None) else None
        return LLMPlayer(client, guard=self.guard, policy=pol, transport_budget=budget,
                         token_budget=self.tokens, playbook=self.cfg.playbook != "none",
                         on_step=on_step, lang=self.cfg.lang, **tiers)

    def _seat_client(self, model: str, seat: int, version: int | None, client_factory=None, tier: str | None = None):
        if client_factory is None and self.hub is not None:
            client = self.hub.client(model, self.seed * 100 + seat, game_id=self.id, seat=seat, seq=version,
                                     max_game_tokens=self.cfg.max_tokens, fallbacks=self.cfg.fallbacks)
        else:
            client = (client_factory or self.client_for)(model, seat)
        writer = self.cassette()
        if writer is not None:
            # One player object per turn here, so its own turn counter is always 0: the
            # journal version it was asked at goes in front of it in the key. A cascade's
            # strong tier shares that numbering (recorded.RecordingClient, `tier`).
            client = RecordingClient(client, writer, seat=seat, turn_prefix="-" if version is None else version,
                                     tier=tier)
        return client

    def note_cascade(self, meta: dict) -> None:
        """Count one committed model turn's cascade record (AgentResult.meta["cascade"])."""
        c = meta.get("cascade")
        if not c:
            return
        st = self.cascade_stats
        st["turns"] += 1
        st["strong_turns"] += int(c.get("tier") == "strong")
        st["calls"] += len(c.get("calls") or [])
        st["strong_calls"] += sum(1 for x in c.get("calls") or [] if x.get("tier") == "strong")
        for e in c.get("escalations") or []:
            st["escalations"][e["reason"]] = st["escalations"].get(e["reason"], 0) + 1

    def cassette(self):
        """This game's cassette writer, or None when the game is not being recorded.

        A file per game, appended to by whichever process runs the turn (the actor's
        node, a Streams worker, the node that adopted the game after a crash): one line
        per call, so lines from several processes interleave but never tear. The writer
        is cached per process so the message blobs are written once per process.
        """
        if not (self.cfg.record or os.environ.get("LLM_RECORD") == "1"):
            return None
        path = os.path.join(os.environ.get("LLM_RECORD_DIR", "runs/cassettes"), f"{self.id}.cassette.jsonl")
        return recorder_for(path, {"game_id": self.id, "seed": self.seed, "config": self.cfg.to_dict(),
                                   "online": True})

    def client_for(self, model: str, seat: int):
        if model == "mock":
            return MockClient(seed=self.seed * 100 + seat)
        offline = cascade_mock(model, self.seed * 100 + seat)     # the cascade's offline tiers
        if offline is not None:
            return offline
        return _provider_client(model, self.seed * 100 + seat, self.env_keys)

    # ------------------------------------------------------------ world
    def reset(self) -> None:
        self.turn = self.env.reset(self.seed)

    def apply(self, action: Any, forced: bool) -> tuple[Record, list[str]]:
        """Advance the world by one step. Only the actor calls this, after the journal write."""
        assert isinstance(self.turn, Turn)
        before = self._snap()
        record, self.turn = self.ledger.apply(self.env, self.turn, action, forced=forced)
        self.journal.append(record)
        st = self.seat_stats.setdefault(record.actor, {"turns": 0, "forced": 0})
        st["turns"] += 1
        st["forced"] += int(forced)
        return record, self._news(before)

    def fingerprint(self) -> str:
        """A digest of the world as every seat sees it. Two tables with the same
        seed and journal must agree; used by the determinism check."""
        import hashlib
        import json
        views = {s: self.env.observe(s) for s in range(1, self.n + 1)} if isinstance(self.turn, Turn) else {}
        body = json.dumps({"turn": repr(self.turn), "views": views}, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    # ------------------------------------------------------------ events
    def emit(self, kind: str, **data) -> Event:
        ev = Event(len(self.events), kind, data)
        self.events.append(ev)
        return ev

    def public_actor(self, turn: Turn) -> int | None:
        return turn.actor if turn.task in PUBLIC_TASKS else None

    def phase_event(self) -> Event:
        t = self.turn
        assert isinstance(t, Turn)
        return self.emit("phase", phase=self.env._phase, round=self.env._round,
                         actor=self.public_actor(t), task=t.task if t.task in PUBLIC_TASKS else "hidden",
                         human=(t.actor in self.human_seats) if t.task in PUBLIC_TASKS else None)

    def step_events(self, record: Record, news: list[str]) -> None:
        public = record.task in PUBLIC_TASKS
        self.emit("turn", actor=record.actor if public else None, task=record.task if public else "hidden",
                  action=public_action(record.action) if public else None, forced=record.forced if public else None,
                  n=len(self.journal), alive=sorted(self.env._alive), round=self.env._round, phase=self.env._phase)
        for line in news:
            self.emit("news", text=line)

    # ------------------------------------------------------------ drafts (streamed speech)
    # Only the actor calls the draft_* writers, like every other writer here. The
    # invariant they keep: every draft id that appears in a `draft` event is closed by
    # exactly one `draft_commit` (whose text is the journal's text for that turn) or
    # `draft_retract` -- or, for an epoch that died, by the next epoch's `draft_reset`.
    def _draft_emit(self, kind: str, audience: str, **data) -> DraftEvent:
        ev = DraftEvent(self.draft_base + len(self.draft_log), kind, audience, len(self.events), data)
        self.draft_log.append(ev)
        if len(self.draft_log) > DRAFT_LOG_MAX:
            cut = len(self.draft_log) - DRAFT_LOG_MAX
            del self.draft_log[:cut]
            self.draft_base += cut
        return ev

    def draft_open(self, seat: int, seq: int, task: str, run: str, attempt: int, asked_at: float) -> Draft:
        d = Draft(f"d{self.draft_epoch}.{seq}.{seat}.{run}.{attempt}", seat, seq, task,
                  DRAFT_AUDIENCE.get(task, "private"), run, attempt, asked_at)
        self.drafts[d.id] = d
        self.draft_stats["opened"] += 1
        return d

    def draft_delta(self, d: Draft, offset: int, text: str) -> bool:
        """Append `text` at `offset`. A repeat is dropped, an overlap trimmed; a gap is refused."""
        have = len(d.text)
        if offset > have or not text:
            return False
        if offset < have:
            text = text[have - offset:]
            if not text:
                return False
            offset = have
        now = time.time()
        d.first_at = d.first_at or now
        d.last_at = now
        d.text += text
        self.draft_stats["deltas"] += 1
        self._draft_emit("draft", d.audience, draft_id=d.id, seat=d.seat, seq=d.seq, task=d.task,
                         offset=offset, delta=text)
        return True

    def draft_commit(self, d: Draft, text: str, n: int) -> None:
        """The turn was journalled: what is shown becomes exactly `text`, the journal's."""
        self.drafts.pop(d.id, None)
        self.draft_stats["committed"] += 1
        ev = self._draft_emit("draft_commit", d.audience, draft_id=d.id, seat=d.seat, seq=d.seq, task=d.task,
                              n=n, text=text)
        self._draft_close(d, ev, "committed")

    def draft_retract(self, d: Draft, reason: str) -> None:
        self.drafts.pop(d.id, None)
        self.draft_stats["retracted"] += 1
        ev = self._draft_emit("draft_retract", d.audience, draft_id=d.id, seat=d.seat, seq=d.seq, task=d.task,
                              reason=reason)
        self._draft_close(d, ev, "retracted")

    def draft_reset(self, reason: str) -> None:
        """A new incarnation: every draft an earlier epoch showed is void."""
        self._draft_emit("draft_reset", "public", reason=reason)

    def _draft_close(self, d: Draft, ev: DraftEvent, outcome: str) -> None:
        self.draft_closed[d.id] = {"ev": ev}
        while len(self.draft_closed) > 400:
            self.draft_closed.pop(next(iter(self.draft_closed)))
        now = time.time()
        self.draft_timings.append({"seat": d.seat, "task": d.task, "outcome": outcome, "chars": len(d.text),
                                   "first_ms": round((d.first_at - d.asked_at) * 1000, 1) if d.first_at else None,
                                   "last_ms": round((d.last_at - d.asked_at) * 1000, 1) if d.last_at else None,
                                   "close_ms": round((now - d.asked_at) * 1000, 1)})
        del self.draft_timings[:-2000]

    def note_cancelled(self, saved: dict | None) -> None:
        self.draft_stats["cancelled_turns"] += 1
        saved = saved or {}
        self.draft_stats["tokens_saved"] += int(saved.get("total", 0))
        self.draft_stats["tokens_saved_cut"] += int(saved.get("cut", 0))
        self.draft_stats["tokens_saved_skipped"] += int(saved.get("skipped", 0))

    def draft_visible(self, audience: str, seat: int | None) -> bool:
        """Who may watch a draft: the same rule as the turn it belongs to (adr/0007)."""
        if audience == "public" or self.may_see_all(seat):
            return True
        if audience == "wolves" and seat:
            return getattr(self.env, "_roles", {}).get(seat) == "werewolf"
        return False

    @property
    def dseq(self) -> int:
        """The dseq the next draft event will get."""
        return self.draft_base + len(self.draft_log)

    def draft_events(self, after: int, seat: int | None) -> list[DraftEvent] | None:
        """Draft events with dseq > `after` this reader may see; None if they were trimmed away."""
        if after + 1 < self.draft_base:
            return None
        return [e for e in self.draft_log[max(0, after + 1 - self.draft_base):] if self.draft_visible(e.audience, seat)]

    def drafts_view(self, seat: int | None) -> list[dict]:
        return [{"draft_id": d.id, "seat": d.seat, "seq": d.seq, "task": d.task, "text": d.text}
                for d in self.drafts.values() if d.text and self.draft_visible(d.audience, seat)]

    def draft_sync(self, seat: int | None, known: dict[str, int] | None = None) -> list[dict]:
        """Bring a client that lost the thread up to date: each open draft from the offset it
        already has, and the outcome of each draft it knows that is no longer open."""
        known = dict(known or {})
        out: list[dict] = []
        for d in self.drafts.values():
            if not d.text or not self.draft_visible(d.audience, seat):
                continue
            off = max(0, min(int(known.pop(d.id, 0) or 0), len(d.text)))
            out.append({"kind": "draft", "epoch": self.draft_epoch, "draft_id": d.id, "seat": d.seat, "seq": d.seq,
                        "task": d.task, "offset": off, "delta": d.text[off:], "sync": True})
        for did in known:
            closed = self.draft_closed.get(did)
            if closed and self.draft_visible(closed["ev"].audience, seat):
                out.append({**closed["ev"].to_dict(self.draft_epoch), "sync": True})
            else:                               # an id from another epoch, or one long gone
                out.append({"kind": "draft_retract", "epoch": self.draft_epoch, "draft_id": did,
                            "reason": "lost", "sync": True})
        return out

    # ------------------------------------------------------------ news (public facts only)
    def _snap(self) -> dict:
        e = self.env
        return {"dead": {d.player_id for d in e._deaths}, "round": e._round, "phase": e._phase,
                "exiles": dict(e._exiles), "sheriff": e._sheriff, "revote": list(getattr(e, "_revote", []))}

    def _news(self, before: dict) -> list[str]:
        e = self.env
        out: list[str] = []
        fresh = [d for d in e._deaths if d.player_id not in before["dead"]]
        for d in fresh:
            how = {"vote": "被放逐", "hunter": "被猎人带走", "detonate": "自爆"}.get(d.cause, "夜里出局")
            out.append(f"第 {d.round} 轮：{d.player_id} 号{how}。")
        for r, ex in e._exiles.items():
            if r not in before["exiles"]:
                counts: dict = {}
                for _v, t in (e._votes.get(r) or {}).items():
                    counts[t] = counts.get(t, 0) + 1
                tally = "、".join(f"{t} 号 {n} 票" for t, n in sorted(counts.items(), key=lambda x: -x[1])) or "无人投票"
                out.append(f"第 {r} 轮票型：{tally} → " + (f"{ex} 号出局。" if ex is not None else "平票，无人出局。"))
        revote = list(getattr(e, "_revote", []))
        if revote and revote != before["revote"]:
            out.append(f"第 {e._round} 轮平票：" + "、".join(f"{t} 号" for t in revote) + " PK，其余玩家再投一次。")
        if before["phase"] != e._phase and e._phase == "day_speech" and not fresh:
            out.append(f"第 {e._round} 天天亮：平安夜，没有人死。")
        if before["phase"] != e._phase and e._phase == "review":
            side = "狼人阵营" if e._winner == "werewolf" else "好人阵营"
            out.append(f"本局结束：{side}获胜。全场底牌公开，开始复盘。")
        if before["sheriff"] != e._sheriff and e._sheriff:
            out.append(f"{e._sheriff} 号当选警长。")
        return out

    # ------------------------------------------------------------ reads
    def reviewing(self) -> bool:
        return getattr(self.env, "_phase", None) == "review"

    def may_see_all(self, seat: int | None) -> bool:
        """The only door to roles and thoughts: game over / in review, this seat is
        out of the game, or the table was opened in 测试模式 (xray)."""
        if self.cfg.xray or self.state in ("over", "aborted", "crashed") or self.reviewing():
            return True
        if seat and hasattr(self.env, "_alive") and self.env._alive:
            return seat not in self.env._alive
        return False

    def acting_view(self, seat: int | None) -> dict | None:
        a = self.acting
        if not a:
            return None
        if a["task"] not in PUBLIC_TASKS and not self.may_see_all(seat) and a["actor"] != seat:
            return {"actor": None, "task": "hidden", "human": None, "model": None,
                    "seconds": round(time.time() - a["since"], 1), "steps": []}
        return {"actor": a["actor"], "task": a["task"], "human": a["human"], "model": a["model"],
                "seconds": round(time.time() - a["since"], 1), "steps": list(a.get("steps") or [])}

    def health(self) -> dict:
        turns = len(self.journal)
        forced = sum(r.forced for r in self.journal)
        by_model: dict[str, dict] = {}
        for seat, m in self.seat_models.items():
            row = by_model.setdefault(m, {"seats": [], "turns": 0, "forced": 0})
            row["seats"].append(seat)
            row["turns"] += self.seat_stats[seat]["turns"]
            row["forced"] += self.seat_stats[seat]["forced"]
        return {"by_model": by_model, "turns": turns, "forced": forced,
                "forced_rate": round(forced / turns, 4) if turns else 0.0,
                "usable": (forced / turns if turns else 0) <= 0.05,
                "replays": self.ledger.replays, "stale_results": self.stale_results,
                "asked_ahead": self.asked_ahead, "misses": self.misses,
                "recoveries": self.recoveries, "transport_failures": self.transport.failures,
                "tokens": self.tokens.total_tokens, "max_tokens": self.tokens.max_tokens,
                "gateway_drift": self.tokens.drift(), "drafts": dict(self.draft_stats)}

    def summary(self, seat: int | None = None) -> dict:
        alive = sorted(self.env._alive) if hasattr(self.env, "_alive") else []
        rnd, phase = getattr(self.env, "_round", 0), getattr(self.env, "_phase", "-")
        return {"id": self.id, "variant": self.cfg.variant, "model": self.model_label, "models": self.models,
                "seat_models": self.seat_models, "guard": self.cfg.guard, "playbook": self.cfg.playbook,
                "state": self.state, "seed": self.seed, "seats": self.n, "human_seats": self.human_seats,
                "alive": alive, "round": rnd, "phase": phase, "reviewing": phase == "review",
                "outcome": self.outcome, "error": self.error, "xray": self.cfg.xray,
                "acting": self.acting_view(seat), "resumed_from": self.resumed_from,
                "health": self.health(), "created": round(self.created, 3), "version": len(self.journal),
                "engine_version": self.engine_version,
                # whole-game counts only: which seat escalated on which turn is not public
                **({"cascade": dict(self.cascade_stats, mode=self.cascade_policy.mode)}
                   if self.cascade_policy is not None else {})}

    def spectator_view(self) -> dict:
        v = dict(self.env.observe(1))
        v.pop("private", None)
        v.pop("night", None)
        v["role"] = None
        v["you"] = None
        return v

    def me(self, seat: int) -> dict:
        p = self.pending if self.pending_seat == seat else None
        st = self.seat_stats.get(seat, {"turns": 0, "forced": 0})
        return {"seat": seat, "waiting": p is not None, "turn": p.to_dict() if p else None,
                "turn_seconds": self.cfg.turn_seconds, "forced": st["forced"], "turns": st["turns"]}

    def seat_state(self, seat: int | None) -> dict:
        """What one client may know. `seat` must already be authenticated by the caller."""
        out: dict[str, Any] = {"game": self.summary(seat), "seat": seat, "acting": self.acting_view(seat)}
        if self.may_see_all(seat):
            out["revealed"] = dict(self.env._roles) if hasattr(self.env, "_roles") else {}
            out["traces"] = {str(k): v[-12:] for k, v in self.traces.items()}
        out["review"] = self.env.review_log() if hasattr(self.env, "review_log") else []
        if hasattr(self.env, "_alive") and self.env._alive:
            out["view"] = self.env.observe(seat) if seat and 1 <= seat <= self.n else self.spectator_view()
        if seat in self.human_seats:
            out["me"] = self.me(seat)
            # the seat's whole own record (the view keeps only the last 14 for the prompt)
            log = getattr(self.env, "_action_log", None)
            if log is not None and seat in log:
                out["my_record"] = list(log[seat])
        out["log"] = [e.to_dict() for e in self.events[-60:]]
        # what is being said right now (a polling client's copy of the draft stream)
        out["drafts"] = self.drafts_view(seat)
        out["draft_epoch"], out["dseq"] = self.draft_epoch, self.dseq
        return out

    def full_log(self, seat: int | None = None) -> dict:
        wide = self.may_see_all(seat)
        pub = self.spectator_view()["public"] if hasattr(self.env, "_alive") and self.env._alive else {}
        journal = [{"n": i + 1, "actor": r.actor, "task": r.task,
                    "action": r.action if wide else public_action(r.action),
                    "forced": r.forced, "model": self.seat_models.get(r.actor, "真人")}
                   for i, r in enumerate(self.journal)
                   if wide or r.task in PUBLIC_TASKS or r.actor == seat]
        over = self.state in ("over", "aborted", "crashed") or self.reviewing()
        return {"game": self.summary(seat), "public": pub, "journal": journal,
                "traces": ({str(k): v for k, v in self.traces.items()} if wide else {}),
                "news": [e.to_dict() for e in self.events if e.kind == "news"],
                "review": self.env.review_log() if hasattr(self.env, "review_log") else [],
                "revealed": dict(self.env._roles) if over else None}
