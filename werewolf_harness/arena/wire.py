"""Plain JSON for what crosses a process boundary, without losing a type.

A seat's view and its legal actions are computed by the game's actor and sent to
a model worker (app/llm/queue.py). json.dumps would quietly turn ``{3: "x"}``
into ``{"3": "x"}`` and a tuple into a list -- and a view does have int-keyed
maps (tallies and assignments by seat) that the prompt and the info tools read
by number. So the two shapes JSON has no word for are tagged:

    dict with a non-str key   {"__map__": [[key, value], ...]}     (order kept)
    tuple                     {"__tuple__": [...]}

``decode(encode(x)) == x``, types included. Anything that is not JSON already
(a set, an object) is refused rather than stringified.
"""
from __future__ import annotations

from typing import Any

MAP = "__map__"
TUPLE = "__tuple__"


def encode(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, list):
        return [encode(x) for x in obj]
    if isinstance(obj, tuple):
        return {TUPLE: [encode(x) for x in obj]}
    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj) and MAP not in obj and TUPLE not in obj:
            return {k: encode(v) for k, v in obj.items()}
        return {MAP: [[encode(k), encode(v)] for k, v in obj.items()]}
    raise TypeError(f"cannot send a {type(obj).__name__} over the wire: {obj!r:.80}")


def decode(obj: Any) -> Any:
    if isinstance(obj, list):
        return [decode(x) for x in obj]
    if isinstance(obj, dict):
        if len(obj) == 1 and MAP in obj:
            return {decode(k): decode(v) for k, v in obj[MAP]}
        if len(obj) == 1 and TUPLE in obj:
            return tuple(decode(x) for x in obj[TUPLE])
        return {k: decode(v) for k, v in obj.items()}
    return obj
