"""Record every model call of a game, and play it back.

A bug seen in a real game is a fact about a sequence of model replies. Without
those replies it cannot be reproduced -- the next run asks the model again and
gets a different game -- and every later change to the harness (the loop, the
guard, the engine) would have to be checked with fresh paid runs. So the replies
are kept: ``RecordingClient`` wraps any LLMClient and appends each call's request
and response (or error) to a cassette; ``ReplayClient`` answers from it.

    key = (seat, the seat's turn number, task, call number within that turn, kind)
    kind = step | forced | note

The key never depends on thread timing. Seats think in parallel and notes are
written in the background, so calls from different seats interleave in any
order -- but within one seat they are strictly serial (a seat has at most one
open ask, and its next act() waits for the previous note), so a per-seat
counter is deterministic. The seat, turn and kind come from the LLMPlayer
through a context variable (``call_scope``); the model-call pool copies
context variables into its worker thread, so the scope survives the hop.

Next to each response the cassette stores a hash of the canonical request
(messages, tool schemas, model, sampling parameters). On replay the incoming
request is hashed again: a different hash means the prompt changed, and the
error names the seat/turn/call and shows the first message that differs.

Format: one JSONL file per game, written as the game goes (a crash keeps every
call made so far). Messages and tool schemas are content-addressed: each
distinct message is written once as a blob line and calls refer to it by hash,
because step N+1 of a turn repeats steps 1..N and the system prompt repeats on
every turn. Lines:

    {"cassette": 1, "meta": {...}}                     header (game settings)
    {"client": 3, "name": "...", "tool_mode": "..."}  one per seat
    {"b": "<hash>", "v": <message or tool list>}      blob, first time seen
    {"k": "3/12/vote/0/step", "h": "<req hash>", "req": {...}, "resp": {...}}
    {"k": ..., "h": ..., "req": {...}, "err": {"type": "ProviderError", ...}}
"""
from __future__ import annotations

import contextlib
import contextvars
import difflib
import gzip
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field

from .base import Cancelled, CancelToken, LLMClient, LLMResponse, ProviderError, ToolCall

FORMAT = 1


class ReplayDivergence(RuntimeError):
    """The game asked the replay for something the recording does not hold.

    Not a ProviderError and never retried or turned into a fallback (recovery.call_model
    re-raises it): a replay that quietly filled a turn with the default action would
    report a different game as a reproduction of the recorded one.
    """


# ------------------------------------------------------------------ the call's identity
@dataclass(frozen=True)
class Scope:
    seat: int | str
    turn: int | str
    task: str
    kind: str                     # step | forced | note


_SCOPE: contextvars.ContextVar[Scope | None] = contextvars.ContextVar("llm_call_scope", default=None)


@contextlib.contextmanager
def call_scope(seat, turn, task: str, kind: str):
    """Say which seat/turn/kind the model calls made inside this block belong to."""
    token = _SCOPE.set(Scope(seat, turn, task, kind))
    try:
        yield
    finally:
        _SCOPE.reset(token)


def current_scope() -> Scope | None:
    return _SCOPE.get()


def make_key(seat, turn, task, call: int, kind: str) -> str:
    return f"{seat}/{turn}/{task}/{call}/{kind}"


def client_id(seat, tier: str | None = None):
    """How a seat's client is registered in a cassette: the seat, or "<seat>:<tier>" for a
    seat's second model (the strong tier of a cascade)."""
    return seat if tier is None else f"{seat}:{tier}"


# ------------------------------------------------------------------ canonical request
def _canon(v) -> str:
    """Key order is kept, not sorted. The wire JSON keeps it too, so two requests whose
    keys come in a different order are two different prompts; and a reply's arguments
    must come back in the order the model wrote them -- the engine renders a wolf's
    plan in dict order, and a sorted copy replayed a different review prompt."""
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"), default=str)


def blob_hash(v) -> str:
    return hashlib.sha256(_canon(v).encode()).hexdigest()[:20]


def canonical_request(messages, tools, temperature, max_tokens, model: str, tool_mode: str) -> dict:
    """What determines the reply. The timeout is deliberately not part of it: it changes
    how long we wait, not what the model is asked."""
    return {"messages": messages, "tools": tools, "temperature": temperature, "max_tokens": max_tokens,
            "model": model, "tool_mode": tool_mode}


def request_hash(req: dict) -> str:
    return hashlib.sha256(_canon(req).encode()).hexdigest()[:24]


# ------------------------------------------------------------------ responses <-> JSON
def response_to_dict(r: LLMResponse) -> dict:
    raw = r.raw or {}
    # The raw body is mostly the reply again; usage is the part nobody else keeps
    # (cache hits live there), so that is what is stored.
    raw_keep = {k: raw[k] for k in ("usage", "id", "system_fingerprint") if k in raw} if isinstance(raw, dict) else {}
    return {"text": r.text, "tool_calls": [{"name": c.name, "arguments": c.arguments, "id": c.id,
                                            "raw_arguments": c.raw_arguments, "malformed": c.malformed}
                                           for c in r.tool_calls],
            "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens, "latency_ms": r.latency_ms,
            "finish_reason": r.finish_reason, "model": r.model, "raw": raw_keep or None}


def response_from_dict(d: dict) -> LLMResponse:
    """A fresh object every time: the loop edits tool-call arguments in place."""
    d = json.loads(json.dumps(d))
    calls = [ToolCall(name=c["name"], arguments=c["arguments"], id=c.get("id", "call_0"),
                      raw_arguments=c.get("raw_arguments", ""), malformed=c.get("malformed", False))
             for c in d.get("tool_calls") or []]
    return LLMResponse(text=d.get("text", ""), tool_calls=calls, prompt_tokens=d.get("prompt_tokens", 0),
                       completion_tokens=d.get("completion_tokens", 0), latency_ms=d.get("latency_ms", 0),
                       finish_reason=d.get("finish_reason", "stop"), model=d.get("model", ""), raw=d.get("raw"))


def error_to_dict(exc: BaseException) -> dict:
    return {"type": type(exc).__name__, "message": str(exc), "status": getattr(exc, "status", None),
            "hint": getattr(exc, "hint", None), "provider": isinstance(exc, ProviderError)}


# Errors that must come back as themselves, not as a look-alike class: Cancelled is
# not a ProviderError, and recovery.call_model tells it apart by type.
_ERROR_TYPES: dict[str, type] = {"Cancelled": Cancelled}


def error_from_dict(d: dict) -> BaseException:
    """The same error the recorded game saw, so the same recovery path runs.

    A ProviderError comes back as one (with status and hint, which decide retry vs
    give up). Anything else -- a client bug -- comes back as an exception of the
    same class name and message, which is all the loop ever reads of it.
    """
    if d.get("provider"):
        return ProviderError(d["message"], hint=d.get("hint"), status=d.get("status"))
    cls = _ERROR_TYPES.setdefault(d["type"], type(d["type"], (Exception,), {}))
    return cls(d["message"])


# ------------------------------------------------------------------ the file
class CassetteWriter:
    """Append-only, shared by every seat of one game. Thread-safe; one line per write,
    flushed, so a game that dies keeps every call it made."""

    def __init__(self, path: str, meta: dict | None = None, *, append: bool = False):
        """append=False starts the file over (a CLI game recorded to a path used before);
        append=True continues it (an online game whose turns run in several processes,
        or one recovered on another node -- see app/runtime/table.py)."""
        self.path = path
        self._lock = threading.Lock()
        self._blobs: set[str] = set()
        self._calls: dict[tuple, int] = {}
        self._clients: set = set()
        self.calls = 0
        self.append = append
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fresh = not append or not os.path.exists(path) or os.path.getsize(path) == 0
        # Appending processes open the file per write: no handle to leak per game, and
        # each call goes down in one O_APPEND write, so two processes' lines never tear.
        self._f = None if append else open(path, "w", encoding="utf-8")
        if fresh:
            self._write([{"cassette": FORMAT, "meta": meta or {}}])

    def _write(self, objs: list[dict]) -> None:
        data = "".join(_canon(o) + "\n" for o in objs)
        if self._f is None:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(data)
        else:
            self._f.write(data)
            self._f.flush()

    def client(self, seat, name: str, tool_mode: str) -> None:
        with self._lock:
            if seat not in self._clients:
                self._clients.add(seat)
                self._write([{"client": seat, "name": name, "tool_mode": tool_mode}])

    def next_call(self, seat, turn) -> int:
        with self._lock:
            n = self._calls.get((seat, turn), 0)
            self._calls[(seat, turn)] = n + 1
            return n

    def _blob(self, v, out: list) -> str:
        h = blob_hash(v)
        if h not in self._blobs:
            self._blobs.add(h)
            out.append({"b": h, "v": v})
        return h

    def put(self, key: str, req: dict, *, resp: dict | None = None, err: dict | None = None, wall_ms: int = 0) -> None:
        with self._lock:
            lines: list[dict] = []
            stored = {"msgs": [self._blob(m, lines) for m in req["messages"]],
                      "tools": self._blob(req["tools"], lines) if req["tools"] is not None else None,
                      **{k: req[k] for k in ("temperature", "max_tokens", "model", "tool_mode")}}
            line = {"k": key, "h": request_hash(req), "req": stored, "wall_ms": wall_ms}
            line["resp" if err is None else "err"] = resp if err is None else err
            self._write(lines + [line])
            self.calls += 1

    def close(self) -> None:
        with self._lock:
            if self._f is not None:
                self._f.close()


_RECORDERS: dict[str, CassetteWriter] = {}
_RECORDERS_LOCK = threading.Lock()
_RECORDERS_MAX = 256


def recorder_for(path: str, meta: dict | None = None) -> CassetteWriter:
    """The process's appending writer for one cassette path (the online service).

    Bounded: a writer forgotten here only forgets which blobs it wrote, so the next
    one writes them again -- a larger file, never a wrong one."""
    with _RECORDERS_LOCK:
        w = _RECORDERS.get(path)
        if w is None:
            if len(_RECORDERS) >= _RECORDERS_MAX:
                _RECORDERS.pop(next(iter(_RECORDERS)))
            w = _RECORDERS[path] = CassetteWriter(path, meta, append=True)
        return w


@dataclass
class Cassette:
    """A cassette read back: header, the clients, and every call by key."""
    path: str
    meta: dict = field(default_factory=dict)
    clients: dict = field(default_factory=dict)        # seat (as str) -> {"name", "tool_mode"}
    calls: dict[str, dict] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    blobs: dict[str, object] = field(default_factory=dict)
    duplicates: int = 0                                # same key twice (an online turn asked again after recovery)

    @classmethod
    def load(cls, path: str) -> Cassette:
        c = cls(path)
        with _open(path, "r") as f:
            for n, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    if n > 0:
                        break                          # a half-written last line: the game died mid-write
                    raise
                if "cassette" in obj:
                    if obj["cassette"] != FORMAT:
                        raise ValueError(f"{path}: cassette format {obj['cassette']}, this code reads {FORMAT}")
                    c.meta = obj.get("meta") or {}
                elif "client" in obj:
                    c.clients[str(obj["client"])] = {"name": obj["name"], "tool_mode": obj["tool_mode"]}
                elif "b" in obj:
                    c.blobs[obj["b"]] = obj["v"]
                elif "k" in obj:
                    if obj["k"] in c.calls:
                        c.duplicates += 1
                        continue
                    c.calls[obj["k"]] = obj
                    c.order.append(obj["k"])
        return c

    def request(self, key: str) -> dict:
        """The recorded request in full, blobs resolved."""
        r = self.calls[key]["req"]
        return canonical_request([self.blobs[h] for h in r["msgs"]],
                                 self.blobs[r["tools"]] if r["tools"] is not None else None,
                                 r["temperature"], r["max_tokens"], r["model"], r["tool_mode"])

    def write_subset(self, path: str, keys: set[str], meta: dict | None = None) -> None:
        """A smaller cassette holding only these calls (and the blobs they use)."""
        with _open(path, "w") as f:
            f.write(_canon({"cassette": FORMAT, "meta": meta if meta is not None else self.meta}) + "\n")
            for seat, cl in self.clients.items():
                f.write(_canon({"client": int(seat) if seat.isdigit() else seat, **cl}) + "\n")
            written: set[str] = set()
            for k in self.order:
                if k not in keys:
                    continue
                line = self.calls[k]
                for h in line["req"]["msgs"] + ([line["req"]["tools"]] if line["req"]["tools"] else []):
                    if h not in written:
                        written.add(h)
                        f.write(_canon({"b": h, "v": self.blobs[h]}) + "\n")
                f.write(_canon(line) + "\n")


def _open(path: str, mode: str):
    """A stored case is gzipped (a cassette compresses ~14x: it is mostly the same
    Chinese prose over and over); a live recording is plain, so every line is on disk
    the moment it is written."""
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def cassette_path_for(log_path: str) -> str:
    """Where `play --out g.json --record` puts the cassette: next to the log."""
    base = log_path[:-5] if log_path.endswith(".json") else log_path
    return base + ".cassette.jsonl"


# ------------------------------------------------------------------ recording
class RecordingClient(LLMClient):
    """Passes every call through to ``inner`` and writes it to the cassette.

    ``seat`` is used when a call arrives with no scope (a client used outside the
    LLMPlayer); ``turn_prefix`` namespaces the turn number where a player object
    lives for one turn only (the online service builds one per job, so its own turn
    counter is always 0: the journal version goes in front).

    The recorder enforces the model-call deadline itself, one second before
    recovery's hard deadline would. Otherwise a call that overran would be recorded
    with the reply that eventually arrived while the game had already moved on with
    a timeout -- and the replay would play a different game.

    ``tier`` names a second client of the same seat (the strong model of a cascade,
    players/llm/cascade.py). Its calls share the seat's per-turn call counter -- an
    escalated retry is just the next call of the turn, so the key format is unchanged --
    and its model name is registered as client "<seat>:<tier>", which is what the
    ReplayClient for that tier reads back. The model is part of the request hash, so
    replaying a call on the wrong tier is a divergence, never a silent substitution.
    """

    def __init__(self, inner: LLMClient, cassette: CassetteWriter, *, seat=None, turn_prefix=None,
                 tier: str | None = None):
        self.inner = inner
        self.cassette = cassette
        self.seat = seat
        self.turn_prefix = turn_prefix
        self.tier = tier
        self.name = getattr(inner, "name", "unknown")
        self.tool_mode = getattr(inner, "tool_mode", "native")
        if seat is not None:
            cassette.client(client_id(seat, tier), self.name, self.tool_mode)

    def _key(self) -> tuple[str, Scope]:
        sc = current_scope() or Scope(self.seat if self.seat is not None else "-", "-", "-", "step")
        seat = sc.seat if sc.seat is not None else self.seat
        turn = sc.turn if self.turn_prefix is None else f"{self.turn_prefix}.{sc.turn}"
        if seat is not None:
            self.cassette.client(client_id(seat, self.tier), self.name, self.tool_mode)
        n = self.cassette.next_call(seat, turn)
        return make_key(seat, turn, sc.task, n, sc.kind), sc

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0) -> LLMResponse:
        key, _ = self._key()
        req = canonical_request(messages, tools, temperature, max_tokens, self.name, self.tool_mode)
        t0 = time.monotonic()
        try:
            resp = _deadline(lambda: self.inner.chat(messages, tools=tools, temperature=temperature,
                                                     max_tokens=max_tokens, timeout=timeout), timeout)
        except BaseException as exc:                       # noqa: BLE001 -- recorded, then re-raised as is
            self.cassette.put(key, req, err=error_to_dict(exc), wall_ms=int((time.monotonic() - t0) * 1000))
            raise
        self.cassette.put(key, req, resp=response_to_dict(resp), wall_ms=int((time.monotonic() - t0) * 1000))
        return resp

    def chat_stream(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0,
                    cancel: CancelToken | None = None):
        """Stream from ``inner`` and record what chat() would have recorded.

        Streaming is how a reply is delivered, not what it is: the line written is the
        final response (or error) under the same key and request hash as a chat() call,
        so a cassette recorded with streaming on replays through ReplayClient (whose
        chat_stream is the base default: the recorded reply sent as fragments) and one
        recorded without streaming replays through the streaming path.

        The deadline is the one chat() enforces, kept the streaming way: at
        hard deadline - slack this call's own cancel token is pulled and the call is
        recorded as the same timeout ProviderError. A client that cannot be interrupted
        finishes late, and is still recorded as that timeout -- the game moved on without
        its reply. A cancel from outside (the turn is gone) is recorded as an error too,
        so the per-turn call numbering stays the same as the run that made it.
        """
        key, _ = self._key()
        req = canonical_request(messages, tools, temperature, max_tokens, self.name, self.tool_mode)
        t0 = time.monotonic()
        hard = _hard_deadline(timeout)
        mine = cancel.child() if cancel is not None else CancelToken()
        fired = threading.Event()

        def expire():
            fired.set()
            mine.cancel("deadline")
        timer = threading.Timer(max(0.05, hard - RECORD_SLACK_S), expire)
        timer.daemon = True
        timer.start()
        late = ProviderError(f"model call timed out after {hard:.0f}s (hard deadline)", status=None)
        final: LLMResponse | None = None
        try:
            for item in self.inner.chat_stream(messages, tools=tools, temperature=temperature,
                                               max_tokens=max_tokens, timeout=timeout, cancel=mine):
                if isinstance(item, LLMResponse):
                    final = item
                    break
                if fired.is_set():
                    raise late
                yield item
            if final is None:
                raise ProviderError("stream ended without a response", status=502)
            if fired.is_set():
                raise late
        except Cancelled as exc:
            timer.cancel()
            err: BaseException = late if fired.is_set() and not (cancel is not None and cancel.is_set()) else exc
            self.cassette.put(key, req, err=error_to_dict(err), wall_ms=int((time.monotonic() - t0) * 1000))
            if err is exc:
                raise
            raise late from exc
        except BaseException as exc:                       # noqa: BLE001 -- recorded, then re-raised as is
            timer.cancel()
            if not isinstance(exc, GeneratorExit):
                self.cassette.put(key, req, err=error_to_dict(exc), wall_ms=int((time.monotonic() - t0) * 1000))
            raise
        timer.cancel()
        self.cassette.put(key, req, resp=response_to_dict(final), wall_ms=int((time.monotonic() - t0) * 1000))
        yield final


RECORD_SLACK_S = 1.0


def _hard_deadline(timeout: float) -> float:
    """recovery._with_deadline's deadline for a call with this timeout."""
    return timeout * 1.5 + 5


def _deadline(fn, timeout: float):
    """Run ``fn`` and give up just before recovery's hard deadline (timeout*1.5+5) would.

    The message is the one recovery would have produced, so a recorded game and an
    unrecorded one tell the same story in their error lists.
    """
    hard = _hard_deadline(timeout)
    box: dict = {}

    def work():
        try:
            box["v"] = fn()
        except BaseException as exc:                       # noqa: BLE001
            box["e"] = exc
    th = threading.Thread(target=work, daemon=True, name="recorded-call")
    th.start()
    th.join(max(0.05, hard - RECORD_SLACK_S))
    if th.is_alive():
        raise ProviderError(f"model call timed out after {hard:.0f}s (hard deadline)", status=None)
    if "e" in box:
        raise box["e"]
    return box["v"]


# ------------------------------------------------------------------ replay
@dataclass
class Drift:
    key: str
    diff: str


class ReplayClient(LLMClient):
    """Answers from a cassette, in the recorded order of each seat's own calls.

    strict (default): a request whose hash differs from the recorded one raises
    ReplayDivergence naming the call and showing the first differing message.
    allow_drift: the difference is logged in ``drifts`` and the recorded reply is
    served anyway -- for checking that a deliberate prompt change leaves the game
    intact. A call the recording never made is an error in both modes.
    """

    def __init__(self, cassette: Cassette, seat, *, allow_drift: bool = False, turn_prefix=None,
                 used: set | None = None, drifts: list | None = None, counters: dict | None = None,
                 tier: str | None = None):
        """`tier` replays a seat's second client (see RecordingClient); the two clients of
        one seat must share `counters`, as their recorded calls shared one numbering."""
        self.cassette = cassette
        self.seat = seat
        self.allow_drift = allow_drift
        self.turn_prefix = turn_prefix
        info = cassette.clients.get(str(client_id(seat, tier))) or {}
        self.name = info.get("name", "replay")
        self.tool_mode = info.get("tool_mode", "native")
        self.used = used if used is not None else set()
        self.drifts = drifts if drifts is not None else []
        self._counters = counters if counters is not None else {}
        self._lock = threading.Lock()
        self.calls = 0

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0) -> LLMResponse:
        sc = current_scope() or Scope(self.seat, "-", "-", "step")
        seat = sc.seat if sc.seat is not None else self.seat
        turn = sc.turn if self.turn_prefix is None else f"{self.turn_prefix}.{sc.turn}"
        with self._lock:
            n = self._counters.get((seat, turn), 0)
            self._counters[(seat, turn)] = n + 1
            self.calls += 1
        key = make_key(seat, turn, sc.task, n, sc.kind)
        line = self.cassette.calls.get(key)
        if line is None:
            raise ReplayDivergence(_missing(self.cassette, seat, turn, sc, n))
        self.used.add(key)
        req = canonical_request(messages, tools, temperature, max_tokens, self.name, self.tool_mode)
        if request_hash(req) != line["h"]:
            diff = describe_diff(self.cassette.request(key), req)
            if not self.allow_drift:
                raise ReplayDivergence(f"请求和录制时不一样：{_where(seat, turn, sc, n)}\n{diff}")
            self.drifts.append(Drift(key, diff))
        if "err" in line:
            raise error_from_dict(line["err"])
        return response_from_dict(line["resp"])


def _where(seat, turn, sc: Scope, n: int) -> str:
    return f"座位 {seat} · 第 {turn} 次行动（{sc.task}）· 本回合第 {n} 次调用 [{sc.kind}]"


def _missing(c: Cassette, seat, turn, sc: Scope, n: int) -> str:
    near = [k for k in c.order if k.startswith(f"{seat}/{turn}/")]
    return (f"录像里没有这一次调用：{_where(seat, turn, sc, n)}。"
            f"这一步录制时没有发生，游戏已经和录像分叉。"
            + (f" 录像里这一回合的调用：{near}" if near else f" 录像里没有座位 {seat} 第 {turn} 次行动的任何调用。"))


def describe_diff(recorded: dict, now: dict, width: int = 6) -> str:
    """The first thing that differs, short enough to read in an error message."""
    a, b = recorded["messages"], now["messages"]
    for i in range(max(len(a), len(b))):
        if i >= len(a):
            return f"消息 #{i}（{b[i].get('role')}）是新加的：{_short(b[i].get('content'))}"
        if i >= len(b):
            return f"消息 #{i}（{a[i].get('role')}）录制时有，现在没有：{_short(a[i].get('content'))}"
        if a[i] != b[i]:
            ra, rb = a[i].get("role"), b[i].get("role")
            if ra != rb:
                return f"消息 #{i} 的角色变了：{ra} -> {rb}"
            ta = a[i].get("content") if isinstance(a[i].get("content"), str) else _canon(a[i])
            tb = b[i].get("content") if isinstance(b[i].get("content"), str) else _canon(b[i])
            lines = list(difflib.unified_diff(ta.splitlines(), tb.splitlines(), "录制时", "现在", n=1, lineterm=""))
            body = "\n".join(lines[: 2 + width * 2]) or f"{_short(ta)}\n-> {_short(tb)}"
            return f"消息 #{i}（{ra}）不同：\n{body}"
    for k in ("tools", "temperature", "max_tokens", "model", "tool_mode"):
        if recorded.get(k) != now.get(k):
            if k == "tools":
                na = [t["function"]["name"] for t in recorded.get("tools") or []]
                nb = [t["function"]["name"] for t in now.get("tools") or []]
                return f"工具定义变了：{na} -> {nb}" if na != nb else "工具定义变了（名字相同，参数或描述不同）"
            return f"{k} 变了：{recorded.get(k)!r} -> {now.get(k)!r}"
    return "（找不到差异：哈希算法变了？）"


def _short(s, n: int = 160) -> str:
    s = s if isinstance(s, str) else _canon(s)
    return s if len(s) <= n else s[:n] + "…"
