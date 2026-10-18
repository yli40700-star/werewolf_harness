"""Layer A: the whole contract between a framework and an environment.

An environment implements five methods. Nothing here knows what a game is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ── what step() hands back ─────────────────────────────────────────────
@dataclass(frozen=True)
class Turn:
    actor: str
    task: str


@dataclass(frozen=True)
class Terminal:
    outcome: dict[str, Any]


# ── what an actor is allowed to submit right now ───────────────────────
@dataclass(frozen=True)
class Param:
    """One argument of a ToolSpec.

    type:     "int" | "str" | "float" | "list[str]" | "dict[int,str]"
    choices:  when given, the whitelist for this argument (rendered and checked)
    labels:   what to call each choice when a person reads it; ids stay the value
    """
    type: str
    description: str = ""
    required: bool = True
    choices: tuple | None = None
    labels: dict | None = None
    max_len: int | None = None
    non_empty: bool = False
    range: tuple[float, float] | None = None
    default: Any = None


@dataclass(frozen=True)
class ToolSpec:
    """One terminal action the actor may take right now, with its legal arguments.

    Legality that depends on state (which seats are alive, who may be shot) is
    expressed as `choices` on the relevant Param, computed by the environment.
    The same object renders into the prompt and gates step().
    """
    name: str
    description: str
    params: dict[str, Param] = field(default_factory=dict)
    validate: Any = None          # optional callable(args) -> reason | None, for cross-argument rules

    def check(self, args: dict) -> str | None:
        if not isinstance(args, dict):
            return f"{self.name}: arguments must be an object"
        reason = self._check_params(args)
        if reason:
            return reason
        if self.validate is not None:
            return self.validate(self.normalise(args))
        return None

    def _check_params(self, args: dict) -> str | None:
        extra = set(args) - set(self.params)
        if extra:
            return f"{self.name} does not take argument(s): {sorted(extra)}"
        for k, p in self.params.items():
            if k not in args or args[k] is None:
                if p.required:
                    return f"{self.name}: missing argument {k!r}"
                continue
            v = args[k]
            if p.type == "int":
                if isinstance(v, bool) or not isinstance(v, int):
                    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
                        v = int(v)
                    else:
                        return f"{self.name}.{k} must be an integer"
            elif p.type == "float":
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    return f"{self.name}.{k} must be a number"
                if p.range and not (p.range[0] <= v <= p.range[1]):
                    return f"{self.name}.{k}={v} outside [{p.range[0]}, {p.range[1]}]"
            elif p.type == "str":
                if not isinstance(v, str):
                    return f"{self.name}.{k} must be a string"
                if p.non_empty and not v.strip():
                    return f"{self.name}.{k} must not be empty"
                if p.max_len and len(v) > p.max_len:
                    return f"{self.name}.{k} longer than {p.max_len} chars"
            elif p.type == "list[str]":
                if isinstance(v, str):
                    v = [v]
                if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                    return f"{self.name}.{k} must be a list of strings"
            elif p.type == "dict[int,str]":
                if not isinstance(v, dict):
                    return f"{self.name}.{k} must be an object"
            if p.choices is not None:
                # For a collection the whitelist is about the ELEMENTS: an
                # assignment of one allowed value per key is legal when each is.
                picked = (list(v.values()) if p.type == "dict[int,str]"
                          else list(v) if p.type == "list[str]" else [v])
                bad = [x for x in picked if x not in p.choices]
                if bad:
                    return f"{self.name}.{k}={bad[0]!r} is not one of {list(p.choices)}"
        return None

    def normalise(self, args: dict) -> dict:
        """Coerce digit strings to ints, fill defaults. Call after check()."""
        out = {}
        for k, p in self.params.items():
            v = args.get(k, p.default)
            if v is not None and p.type == "int" and isinstance(v, str):
                v = int(v)
            if v is not None and p.type == "list[str]" and isinstance(v, str):
                v = [v]
            if v is not None and p.type == "dict[int,str]":
                v = {int(a): b for a, b in v.items()}
            out[k] = v
        return out

    def describe(self) -> str:
        bits = []
        for k, p in self.params.items():
            d = f"{k}: {p.type}"
            if p.choices is not None:
                d += " in " + str(list(p.choices))
            if not p.required:
                d += " (optional)"
            bits.append(d)
        return f"{self.name}({', '.join(bits)}) -- {self.description}"

    def openai_schema(self) -> dict:
        tmap = {"int": "integer", "float": "number", "str": "string", "list[str]": "array", "dict[int,str]": "object"}
        props, req = {}, []
        for k, p in self.params.items():
            prop: dict[str, Any] = {"type": tmap[p.type], "description": p.description}
            if p.type == "list[str]":
                prop["items"] = {"type": "string"}
            if p.type == "dict[int,str]":
                prop["additionalProperties"] = {"type": "string"}
            if p.choices is not None:
                if p.type in ("int", "str"):
                    prop["enum"] = list(p.choices)
                elif p.type == "list[str]":
                    prop["items"] = {"type": "string", "enum": list(p.choices)}
                elif p.type == "dict[int,str]":
                    prop["additionalProperties"] = {"type": "string", "enum": list(p.choices)}
            if p.range:
                prop["minimum"], prop["maximum"] = p.range
            props[k] = prop
            if p.required:
                req.append(k)
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": props, "required": req}}}


@dataclass(frozen=True)
class ActionSpec:
    """kind='choice'  → options is the whitelist
       kind='text'    → any string up to max_len
       kind='form'    → dict whose keys/ranges are given by fields
       kind='tool'    → {"name": <one of tools>, "args": {...}}; each ToolSpec gates its own args
    """
    kind: str
    options: tuple[str, ...] = ()
    max_len: int = 400
    fields: dict[str, tuple[float, float]] = field(default_factory=dict)
    tools: tuple[ToolSpec, ...] = ()
    fallback: Any = None          # used when an actor fails every retry

    def tool(self, name: str) -> ToolSpec | None:
        return next((t for t in self.tools if t.name == name), None)

    def check(self, action: Any) -> str | None:
        """Return a reason string if illegal, None if fine."""
        if self.kind == "choice":
            if action not in self.options:
                return f"{action!r} is not one of {list(self.options)}"
        elif self.kind == "text":
            if not isinstance(action, str) or not action.strip():
                return "expected non-empty text"
            if len(action) > self.max_len:
                return f"text longer than {self.max_len} chars"
        elif self.kind == "form":
            if not isinstance(action, dict):
                return "expected a dict"
            for k, (lo, hi) in self.fields.items():
                if k not in action:
                    return f"missing field {k!r}"
                if not (lo <= action[k] <= hi):
                    return f"{k}={action[k]} outside [{lo}, {hi}]"
        elif self.kind == "tool":
            if not isinstance(action, dict) or "name" not in action:
                return "expected {'name': <tool>, 'args': {...}}"
            t = self.tool(action["name"])
            if t is None:
                return f"{action['name']!r} is not available now; you may call: {[x.name for x in self.tools]}"
            return t.check(action.get("args") or {})
        return None

    def describe(self) -> str:
        """Rendered into the prompt. Same object that gates step()."""
        if self.kind == "choice":
            return "choose one of: " + ", ".join(self.options)
        if self.kind == "text":
            return f"free text, at most {self.max_len} characters"
        if self.kind == "tool":
            return "end your turn with exactly one of:\n" + "\n".join("- " + t.describe() for t in self.tools)
        return "fill in: " + ", ".join(f"{k} in [{lo}, {hi}]" for k, (lo, hi) in self.fields.items())


class Illegal(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ── the environment contract ───────────────────────────────────────────
@runtime_checkable
class Environment(Protocol):
    def reset(self, seed: int) -> Turn | Terminal: ...
    def observe(self, actor: str) -> dict[str, Any]: ...
    def action_space(self, actor: str, task: str) -> ActionSpec: ...
    def step(self, actor: str, task: str, action: Any) -> Turn | Terminal: ...
    def rules_text(self) -> str: ...


# ── the player contract (LLM / human / script all look like this) ──────
class Player(Protocol):
    def act(self, view: dict[str, Any], task: str, spec: ActionSpec, feedback: str | None) -> Any: ...
