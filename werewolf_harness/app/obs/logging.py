"""Structured logs: one JSON object per line, with the game context attached.

`bind(game_id=..., seat=...)` sets context for the current task (contextvars),
so a log line written deep inside the runtime still says which game it is about.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time

_ctx: contextvars.ContextVar[dict | None] = contextvars.ContextVar("log_ctx", default=None)


def bind(**kw) -> contextvars.Token:
    return _ctx.set({**(_ctx.get() or {}), **{k: v for k, v in kw.items() if v is not None}})


def unbind(token: contextvars.Token) -> None:
    _ctx.reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        row = {"ts": round(time.time(), 3), "level": record.levelname.lower(), "logger": record.name,
               "msg": record.getMessage(), **(_ctx.get() or {})}
        extra = getattr(record, "fields", None)
        if extra:
            row.update(extra)
        if record.exc_info:
            row["exc"] = self.formatException(record.exc_info)
        return json.dumps(row, ensure_ascii=False, default=str)


def setup(level: str = "INFO") -> None:
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root.addHandler(h)
    root.setLevel(level)


def get(name: str) -> logging.Logger:
    return logging.getLogger(name)


def kv(logger: logging.Logger, level: int, msg: str, **fields) -> None:
    logger.log(level, msg, extra={"fields": fields})
