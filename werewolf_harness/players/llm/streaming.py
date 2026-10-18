"""Speech text out of a reply that is still being generated.

A model does not stream a speech, it streams a tool call: native function
calling sends the arguments JSON a few characters at a time
(``{"thought": "...", "content": "我觉得 5 号..."}``), and the JSON tool protocol
sends the whole envelope as text (``{"thought": ..., "action": "speak", "args":
{"content": ...}}``). What a table may watch is the value of ``content`` and
nothing else -- the ``thought`` beside it is the seat's private reasoning, and it
often arrives *first*.

So this is a small incremental JSON reader that follows the key path of every
string it is inside and hands out the decoded characters of one target string
as they arrive: escapes (``\\n``, ``\\"``, ``\\uXXXX`` and surrogate pairs) are
decoded even when a chunk boundary cuts through them, and nothing outside the
target string is ever emitted. It is not a validator; the finished reply is
still parsed and checked as a whole by the loop -- this only decides what is
safe to show early.
"""
from __future__ import annotations

import threading
from collections.abc import Callable

from ..providers.base import Delta

# Terminal actions whose `content` is something said out loud (or, for pack_say,
# said to the pack). No information tool has a `content` argument, which is what
# makes it safe to start streaming a JSON-protocol reply before its "action" key.
SPEECH_TOOLS = frozenset({"speak", "last_words", "campaign_run", "pack_say", "review_say"})

_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
_HEX = set("0123456789abcdefABCDEF")


class ContentExtractor:
    """feed(chunk) -> the newly decoded characters of the target string.

    `paths` are key paths whose string value is captured (the first one that
    appears wins; later ones are ignored). `watch` paths are read whole and put
    in `values` when they close -- used for the JSON protocol's "action" key.
    Anything before the first ``{`` (a code fence, stray prose) is skipped.
    """

    def __init__(self, paths: tuple[tuple[str, ...], ...], watch: tuple[tuple[str, ...], ...] = ()):
        self.paths = set(paths)
        self.watch = set(watch)
        self.values: dict[tuple[str, ...], str] = {}
        self.text = ""                  # everything captured so far
        self.done = False               # the target string has closed
        self.closed = False             # the top-level object has closed
        self._started = False
        self._stack: list[list] = []    # [kind, key, expecting_key] per open container
        self._in_str = False
        self._role: str | None = None   # "key" | "value" for the open string
        self._capture = False
        self._watching: tuple[str, ...] | None = None
        self._buf: list[str] = []
        self._esc = False
        self._hex: str | None = None    # digits of a \\u escape still arriving
        self._high: int | None = None   # a high surrogate waiting for its pair

    @property
    def active(self) -> bool:
        return self._capture or bool(self.text)

    # ------------------------------------------------------------ input
    def feed(self, chunk: str) -> str:
        out: list[str] = []
        for c in chunk:
            if self.closed:
                break
            if not self._started:
                if c == "{":
                    self._started = True
                    self._stack.append(["obj", None, True])
                continue
            if self._in_str:
                self._string_char(c, out)
            else:
                self._structural(c)
        got = "".join(out)
        self.text += got
        return got

    def _path(self) -> tuple[str, ...]:
        return tuple((f[1] or "") if f[0] == "obj" else "[]" for f in self._stack)

    def _structural(self, c: str) -> None:
        top = self._stack[-1] if self._stack else None
        if c == '"':
            self._in_str, self._buf = True, []
            if top is not None and top[0] == "obj" and top[2]:
                self._role = "key"
            else:
                self._role = "value"
                path = self._path()
                self._capture = path in self.paths and not self.done
                self._watching = path if path in self.watch else None
        elif c in "{[":
            self._stack.append(["obj", None, True] if c == "{" else ["arr", None, False])
        elif c in "}]":
            if self._stack:
                self._stack.pop()
            if not self._stack:
                self.closed = True
        elif c == ":" and top is not None and top[0] == "obj":
            top[2] = False
        elif c == "," and top is not None and top[0] == "obj":
            top[1], top[2] = None, True
        # numbers, true/false/null and whitespace carry nothing we show

    def _string_char(self, c: str, out: list[str]) -> None:
        if self._hex is not None:
            if c in _HEX:
                self._hex += c
                if len(self._hex) == 4:
                    cp, self._hex = int(self._hex, 16), None
                    self._codepoint(cp, out)
                return
            # a broken \\u escape: keep what was written rather than guess
            bad, self._hex = "\\u" + self._hex, None
            for ch in bad:
                self._emit(ch, out)
            self._string_char(c, out)
            return
        if self._esc:
            self._esc = False
            if c == "u":
                self._hex = ""
            else:
                self._emit(_ESCAPES.get(c, c), out)
            return
        if c == "\\":
            self._esc = True
        elif c == '"':
            self._close_string(out)
        else:
            self._emit(c, out)

    def _codepoint(self, cp: int, out: list[str]) -> None:
        if 0xD800 <= cp < 0xDC00:
            if self._high is not None:
                self._emit("�", out)
            self._high = cp
            return
        if 0xDC00 <= cp < 0xE000 and self._high is not None:
            hi, self._high = self._high, None
            self._put(chr(0x10000 + ((hi - 0xD800) << 10) + (cp - 0xDC00)), out)
            return
        self._emit("�" if 0xDC00 <= cp < 0xE000 else chr(cp), out)

    def _emit(self, ch: str, out: list[str]) -> None:
        if self._high is not None:          # a high surrogate that never got its pair
            self._high = None
            self._put("�", out)
        self._put(ch, out)

    def _put(self, ch: str, out: list[str]) -> None:
        if self._role == "key" or self._watching is not None:
            self._buf.append(ch)
        if self._capture:
            out.append(ch)

    def _close_string(self, out: list[str]) -> None:
        if self._high is not None:
            self._high = None
            self._put("�", out)
        top = self._stack[-1] if self._stack else None
        if self._role == "key" and top is not None:
            top[1] = "".join(self._buf)
        elif self._watching is not None:
            self.values[self._watching] = "".join(self._buf)
        if self._capture:
            self.done = True
        self._in_str, self._role, self._capture, self._watching, self._buf = False, None, False, None, []


class SpeechStream:
    """One model call's fragments -> draft text, and a word when it stops counting.

    on_text(chars)   more of the speech, in order
    on_void(reason)  what was streamed is not going to be this call's speech after
                     all: the JSON protocol named a non-speech action, or the
                     gateway reset the stream to try another model

    Native tool calls are followed by name: fragments of a call whose name is not
    a speech tool are ignored, fragments that arrive before the name are held.
    Whichever channel (tool arguments or text) yields speech first is the one
    followed, so a reply that carries both is never shown twice.
    """

    def __init__(self, on_text: Callable[[str], None], on_void: Callable[[str], None],
                 speech_tools: frozenset[str] = SPEECH_TOOLS):
        self.on_text, self.on_void = on_text, on_void
        self.speech_tools = speech_tools
        self.text = ""
        self._fresh()

    def _fresh(self) -> None:
        self._mode: str | None = None
        self._void = False
        self._name: str | None = None
        self._held: list[str] = []
        self._native = ContentExtractor((("content",),))
        self._json = ContentExtractor((("args", "content"), ("arguments", "content")),
                                      watch=(("action",), ("tool",), ("name",)))

    def feed(self, d: Delta) -> None:
        if d.kind == "reset":
            self._drop("reset")
            self._fresh()
            return
        if self._void:
            return
        if d.kind == "tool":
            if d.index != 0:
                return
            if d.name and self._name is None:
                self._name = d.name
            if self._name is None:
                self._held.append(d.text)
                return
            text, self._held = "".join(self._held) + d.text, []
            if self._name in self.speech_tools:
                self._out("tool", self._native.feed(text))
        elif d.kind == "text":
            got = self._json.feed(d.text)
            v = self._json.values
            act = v.get(("action",)) or v.get(("tool",)) or v.get(("name",))
            if act is not None and act not in self.speech_tools:
                self._drop(f"not a speech: {act}")
                self._void = True
                return
            self._out("text", got)

    def _out(self, mode: str, got: str) -> None:
        if not got:
            return
        if self._mode is None:
            self._mode = mode
        if self._mode != mode:
            return
        self.text += got
        self.on_text(got)

    def _drop(self, reason: str) -> None:
        if self.text:
            self.text = ""
            self.on_void(reason)


# ---------------------------------------------------------------- what a cancel saves
#
# A cancelled call is billed for its prompt and for what it generated before the
# plug was pulled; what it saves is the rest of its reply, plus the calls the turn
# would still have made. Neither is observable -- the reply was never written --
# so both are estimated from what finished calls of the same kind cost:
#
#   cut      max(0, E[completion | model, kind] - produced)
#   skipped  the note-to-self call that ends every turn: its prompt (about the
#            size of the cancelled call's) + E[completion | model, "note"]
#
# E[...] is a running mean over this process's finished calls (speech, step, note
# kept apart -- a speech is ~10x a vote), seeded with a modest default so the first
# cancels are not counted as zero. Later ReAct steps the turn might have taken are
# NOT counted, so the figure errs low.
_DEFAULT_EXPECT = {"speech": 400, "step": 60, "note": 100}
_expect: dict[tuple[str, str], tuple[int, float]] = {}
_expect_lock = threading.Lock()


def expect_record(model: str, kind: str, completion_tokens: int) -> None:
    if completion_tokens <= 0:
        return
    with _expect_lock:
        n, mean = _expect.get((model, kind), (0, 0.0))
        n = min(n + 1, 200)                       # a moving mean: old calls fade out
        _expect[(model, kind)] = (n, mean + (completion_tokens - mean) / n)


def expected(model: str, kind: str, cap: int | None = None) -> int:
    with _expect_lock:
        n, mean = _expect.get((model, kind), (0, 0.0))
    e = int(mean) if n else _DEFAULT_EXPECT.get(kind, 60)
    return min(e, cap) if cap else e


def savings(model: str, kind: str, produced: int, prompt: int, *, note_pending: bool,
            max_tokens: int | None = None) -> dict:
    cut = max(0, expected(model, kind, max_tokens) - int(produced))
    skipped = prompt + expected(model, "note", max_tokens) if note_pending else 0
    return {"cut": cut, "skipped": skipped, "total": cut + skipped, "produced": int(produced)}
