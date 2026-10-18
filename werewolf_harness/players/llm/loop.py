"""The LLM seat: a bounded ReAct loop behind the one-method Player contract.

    act(view, task, spec, feedback) -> {"name": <terminal>, "args": {...}}

Inside: system message + situation, up to ``max_react_steps`` model calls,
each either one information tool (executed here, on the view) or one terminal
action (checked against the ActionSpec and the guard's output-side checks,
then returned to the driver). Three gates on every call: schema, whitelist,
semantics. Failures are retried with the reason attached; a turn that cannot
be salvaged returns the environment's fallback and says so in ``last_turn``.
"""
from __future__ import annotations

import copy
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from ...arena import ActionSpec
from ..guard import GuardStack, evidence_mod
from ..budget import TokenBudget, estimate_messages, estimate_tokens
from ..providers.base import Cancelled, CancelToken, LLMClient, ToolCall, invoke_call
from ..providers.recorded import call_scope
from ..recovery import RecoveryPolicy, RecoveryStats, TransportBudget, call_model
from .belief import BeliefState
from .cascade import CascadePolicy
from .context import ContextBuilder
from .streaming import SPEECH_TOOLS, SpeechStream, expect_record, savings
from .tools import InfoRegistry, SemanticError, ToolContext, build_registry

_RECORD_TOOLS = {"query_history", "query_votes", "query_deaths"}
NOTE_MAX_CHARS = 400
# Repeated at the end of the turn's message: models follow the last instruction they read,
# and the English examples in the tool protocol otherwise pull `thought` back into English.
LANG_REMINDER = {"zh": "\n\n（用简体中文：发言内容和每一步的 thought 都写中文。）", "en": ""}
NOTE_PROMPT = ("[harness] Before this turn ends, write a note to yourself for your next turn: what you decided and why, "
               f"what to watch for. At most {NOTE_MAX_CHARS} characters. Plain text only.")
_WANTED = {"speak": "speak", "vote": "vote", "pack_plan": "pack_plan", "campaign": "campaign_run or campaign_pass",
           "campaign_vote": "campaign_vote", "last_words": "last_words", "hunter_shoot": "hunter_shoot or hunter_hold",
           "badge": "badge_transfer or badge_tear", "pack_chat": "pack_say or pack_pass",
           "review": "review_say or review_pass"}


@dataclass
class ReActStep:
    step: int
    thought: str = ""
    action: str = ""
    args: dict = field(default_factory=dict)
    observation: str = ""
    tokens: int = 0
    latency_ms: int = 0
    guard_blocked: bool = False
    block_reason: str | None = None
    guard_detections: list[dict] = field(default_factory=list)
    # Cascade only (players/llm/cascade.py): which model made this step's call, and why the
    # turn moved to the strong model after it. Left out of the dict when unset, so a log
    # written without a cascade (every log from before it) stays byte-identical.
    model: str | None = None
    escalated: str | None = None

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        for k in ("model", "escalated"):
            if d[k] is None:
                del d[k]
        return d


@dataclass
class TurnResult:
    player_id: int
    round: int
    task: str
    steps: list[ReActStep] = field(default_factory=list)
    action: dict | None = None
    belief_before: dict = field(default_factory=dict)
    belief_after: dict = field(default_factory=dict)
    note_before: str = ""
    note_after: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    guard_blocks: list[dict] = field(default_factory=list)
    guard_detections: list[dict] = field(default_factory=list)
    hallucinated_citations: list[dict] = field(default_factory=list)
    recovery: dict = field(default_factory=dict)
    model: str = ""
    note_ms: int = 0                 # the part of latency_ms spent writing the note
    draft_attempt: int | None = None  # which streamed draft became this turn's action (see on_draft)
    # Cascade only: {"start": tier, "tier": the tier that answered, "escalations": [{step, reason,
    # from, to}], "calls": [{kind, tier, model, prompt, completion, ms}]}. None, and absent from
    # the log, otherwise.
    cascade: dict | None = None

    @property
    def fallback_used(self) -> str | None:
        return self.recovery.get("fallback_used")

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "steps"}
        if d.get("draft_attempt") is None:
            # only a streamed turn has one; a log written without streaming (every log from
            # before it, and every CLI game) stays byte-identical, so replay-llm compares equal
            d.pop("draft_attempt", None)
        if d.get("cascade") is None:
            d.pop("cascade", None)
        d["react_trace"] = [s.to_dict() for s in self.steps]
        return d


class LLMPlayer:
    def __init__(self, client: LLMClient, *, guard: GuardStack | None = None, policy: RecoveryPolicy | None = None,
                 temperature: float = 0.7, max_tokens: int = 700, playbook: bool = True, forensics_gate: bool = False,
                 transport_budget: TransportBudget | None = None, anonymise_speakers: bool = False,
                 registry: InfoRegistry | None = None, token_budget: TokenBudget | None = None,
                 on_step=None, lang: str = "zh", async_note: bool = False, on_draft=None,
                 cancel: CancelToken | None = None, strong_client: LLMClient | None = None,
                 cascade: CascadePolicy | None = None, memory: bool = True, invoke_calls: bool = False):
        self.client = client
        # memory=False is an ablation: the belief table and the note to the next turn
        # start empty every turn, so the seat keeps no judgment of its own between turns
        # (what the environment shows and the history tools return are unchanged).
        self.memory = memory
        # read <invoke name=...> replies as calls (games played since 2026-10-18; older ones replay without)
        self.invoke_calls = invoke_calls
        # Model cascade (players/llm/cascade.py): `client` is the cheap tier, `strong_client`
        # the strong one. A turn's transcript is built once and may be continued by either,
        # so both must speak the same tool protocol.
        if cascade is not None:
            if strong_client is None:
                raise ValueError("a cascade needs strong_client")
            modes = (getattr(client, "tool_mode", "native"), getattr(strong_client, "tool_mode", "native"))
            if modes[0] != modes[1]:
                raise ValueError(f"cascade tiers speak different tool protocols: {client.name} {modes[0]} / "
                                 f"{strong_client.name} {modes[1]}")
        self.strong_client = strong_client
        self.cascade = cascade
        self._tier = "cheap"
        # Streaming speech (docs/adr/0009). on_draft(kind, attempt, text) is told, while a
        # speech-like terminal is still being generated, what of its `content` has arrived
        # ("delta"), and when an attempt will not be the turn's answer after all ("abandon":
        # the guard refused it, the stream was reset, the turn fell back). Every attempt it
        # hears about ends in exactly one of: an abandon, or `last_turn.draft_attempt`.
        # Like on_step it is a watcher: nothing the seat decides depends on it.
        self.on_draft = on_draft
        self.cancel = cancel             # set from outside to stop the turn, and the call in flight
        self._attempts = 0
        self._open_attempt: int | None = None
        self._speechy = False
        self.lang = lang
        self.guard = guard or GuardStack()
        self.policy = policy or RecoveryPolicy()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.playbook = playbook
        self.forensics_gate = forensics_gate
        self.transport_budget = transport_budget
        self.token_budget = token_budget
        self.on_step = on_step or (lambda seat, step: None)   # a watcher, never a decider
        self.last_forced = False        # the driver reads this: a fallback is not a decision
        self.context = ContextBuilder(self.guard, max_steps=self.policy.max_react_steps, anonymise_speakers=anonymise_speakers,
                                      lang=lang)
        self.registry = registry
        self.belief: BeliefState | None = None
        # A note to the next turn: one bounded string the model writes as the turn
        # ends and reads as the next one starts. Not a transcript -- it is
        # overwritten every turn, and like the belief table it never reaches the
        # environment.
        self.note: str = ""
        self.history: list[TurnResult] = []
        self.last_turn: TurnResult | None = None
        # The note is for this seat's NEXT turn, so nobody at the table has to wait for
        # it: with async_note the action goes back to the driver at once and the note is
        # written in the background, and the seat's next act() waits for it first.
        self.async_note = async_note
        self._note_thread: threading.Thread | None = None
        self._note_error: BaseException | None = None
        self._gen = 0                    # bumped by discard_last(): a note for a discarded turn is dropped
        self._undo: tuple | None = None
        # Every act() gets a number, discarded ones included: (seat, act number, kind) names
        # each model call independently of thread timing, which is what a recording is keyed
        # by (providers/recorded.py). Calls of one seat never overlap, so this is enough.
        self.turns_started = 0
        self._scope: tuple = (0, 0, "")

    # ------------------------------------------------------------ after the turn
    def flush(self) -> None:
        """Wait for the note still being written, and surface what went wrong in it."""
        th = self._note_thread
        if th is not None:
            th.join()
            self._note_thread = None
        err, self._note_error = self._note_error, None
        if err is not None:
            raise err

    def discard_last(self) -> None:
        """Forget the last act(): the driver asked too early and threw the answer away.

        Belief, note and history go back to what they were before it, so asking again
        sees the seat exactly as the one-at-a-time loop would have.
        """
        if self._undo is None:
            return
        self._gen += 1
        th = self._note_thread
        if th is not None:
            th.join()
            self._note_thread = None
        self._note_error = None
        self.belief, self.note, n = self._undo
        del self.history[n:]
        self.last_turn = self.history[-1] if self.history else None
        self._undo = None

    # ------------------------------------------------------------ the contract
    def act(self, view: dict, task: str, spec: ActionSpec, feedback: str | None) -> Any:
        self.flush()
        self._scope = (view["you"], self.turns_started, task)
        self.turns_started += 1
        if self.belief is None or not self.memory:
            self.belief = BeliefState(view["you"], list(range(1, view["num_players"] + 1)))
        if not self.memory:
            self.note = ""
        if self.registry is None:
            self.registry = build_registry(playbook=self.playbook, num_players=view["num_players"])
        result = TurnResult(view["you"], view["round"], task, belief_before=self.belief.snapshot(), note_before=self.note,
                            model=self.client.name)
        stats = RecoveryStats()
        if self.cascade is not None:
            self._tier, why = self.cascade.start(task, [t.name for t in spec.tools])
            result.cascade = {"start": self._tier, "tier": self._tier, "escalations": [], "calls": []}
            if why:
                result.cascade["escalations"].append({"step": 0, "reason": why, "from": None,
                                                      "to": self.strong_client.name})
            result.model = self._active().name
        self.last_forced = False
        self._undo = (copy.deepcopy(self.belief), self.note, len(self.history))
        self._speechy = self.on_draft is not None and any(t.name in SPEECH_TOOLS for t in spec.tools)
        try:
            action = self._run(view, task, spec, feedback, result, stats)
        finally:
            if self._open_attempt is not None:     # the turn ended on something other than that speech
                self._abandon(stats.fallback_used or "turn ended")
            # A fallback is a legal action, which is exactly why it has to be
            # flagged: without this the journal reads a turn the harness filled
            # in as a move the model chose, and the 兜底率 stays 0% while the
            # circuit breaker is aborting the game for too many fallbacks.
            self.last_forced = bool(stats.fallback_used)
            result.belief_after = self.belief.snapshot()
            if self._note_thread is None:
                result.note_after = self.note        # otherwise the note fills it in when it lands
            result.recovery = stats.to_dict()
            if result.cascade is not None:
                result.cascade["tier"] = self._tier
                result.model = self._active().name     # the model whose answer the turn ended on
            self.history.append(result)
            self.last_turn = result
        result.action = action
        return action

    # ------------------------------------------------------------ the loop
    def _run(self, view: dict, task: str, spec: ActionSpec, feedback: str | None, result: TurnResult, stats: RecoveryStats) -> dict:
        role = view["role"]
        info_tools = self.registry.visible(role)
        terminal_names = [t.name for t in spec.tools]
        situation, directives, detections = self.context.situation_message(view, self.belief, task, spec, note=self.note)
        result.guard_detections.extend(detections)
        messages = [
            {"role": "system", "content": self.context.system_message(view, terminal_names, [t.name for t in info_tools])},
            {"role": "user", "content": situation + LANG_REMINDER.get(self.lang, "")},
        ]
        if feedback:
            messages.append({"role": "user", "content": f"[guard] Your previous answer was refused: {feedback}. Reply with exactly one legal terminal action."})
        schemas = [t.spec.openai_schema() for t in info_tools] + [t.openai_schema() for t in spec.tools]
        ctx = ToolContext(view, self.belief, task)
        queried_players: set[int] = set()
        queried_vote_rounds: set[int] = set()
        consulted_record = False
        lookups: dict[int, str] = {}
        recent: list[tuple] = []
        schema_failures = 0
        step_no = 0

        while step_no < self.policy.max_react_steps:
            step_no += 1
            step = ReActStep(step_no, model=self._active().name if self.cascade is not None else None)
            response = self._call(messages, schemas, stats, kind="step", result=result)
            if response is None:
                if self._escalate(result, step, "transport"):
                    # the cheap model did not answer at all: the same request, to the strong one
                    step.action, step.observation = "<no reply>", "the model did not answer"
                    step.guard_blocked, step.block_reason = True, "transport"
                    self._file(result, step)
                    continue
                stats.fallback_used = "transport_failure"
                if self.transport_budget is not None:
                    self.transport_budget.spend()
                break
            result.prompt_tokens += response.prompt_tokens
            result.completion_tokens += response.completion_tokens
            result.latency_ms += response.latency_ms
            step.tokens, step.latency_ms, step.thought = response.total_tokens, response.latency_ms, _thought_of(response)
            call = response.tool_calls[0] if response.tool_calls else None
            if self.invoke_calls and (call is None or call.malformed) and response.text:
                call = invoke_call(response.text) or call
            # With tool_choice=required a model answers with a call and no text, so
            # `response.text` is empty and the reasoning would be lost. Every terminal
            # carries a `thought` argument for exactly this: prefer what the model put
            # there over the (usually empty) free text.
            if call is not None and isinstance(call.arguments, dict):
                said = call.arguments.get("thought")
                if isinstance(said, str) and said.strip():
                    step.thought = said.strip()[:400]

            # gate 1: a usable call at all
            if call is None or call.malformed or not call.name:
                schema_failures += 1
                stats.retries += 1
                step.action, step.observation, step.guard_blocked, step.block_reason = "<malformed>", "no valid tool call in the reply", True, "schema"
                escalated = self._escalate(result, step, "schema")
                self._file(result, step)
                if schema_failures > self.policy.max_retries and not escalated:
                    break
                messages.append({"role": "assistant", "content": response.text or ""})
                messages.append({"role": "user", "content": "[guard] That reply contained no usable tool call. Reply with exactly one tool call and nothing else."})
                continue
            step.action, step.args = call.name, dict(call.arguments)
            signature = (call.name, tuple(sorted((k, str(v)) for k, v in call.arguments.items())))
            recent.append(signature)
            if len(recent) >= self.policy.loop_repeat_threshold and len(set(recent[-self.policy.loop_repeat_threshold:])) == 1:
                stats.loop_broken = True
                step.observation = "repeated identical call; loop broken"
                self._file(result, step)
                break

            # gate 2: whitelist -- is it one of this turn's tools?
            info = next((t for t in info_tools if t.name == call.name), None)
            term = spec.tool(call.name)
            if info is None and term is None:
                schema_failures += 1
                stats.retries += 1
                allowed = sorted([t.name for t in info_tools] + terminal_names)
                step.observation = f"unknown tool {call.name!r}; you may only call one of: {allowed}"
                step.guard_blocked, step.block_reason = True, "schema"
                escalated = self._escalate(result, step, "unknown_tool")
                self._file(result, step)
                if schema_failures > self.policy.max_retries and not escalated:
                    break
                self._append_turn(messages, call, f"[guard] {step.observation}")
                continue

            if term is not None:
                # On the JSON path the reasoning arrives in the envelope's own
                # `thought`, one level above `args`, and the model rarely repeats
                # it inside. Move it in when the tool takes one, so a claude seat
                # is as readable afterwards as a native-tool-calling seat. Info
                # tools take no `thought`, which is why this is keyed on the spec.
                if "thought" in term.params and isinstance(call.arguments, dict) \
                        and not (call.arguments.get("thought") or "").strip() and step.thought:
                    call.arguments["thought"] = step.thought[:400]
                # gate 2 (args) + gate 3 (state-derived choices, cross-argument rules) -- all in the ToolSpec
                reason = term.check(call.arguments)
                blocked = None if reason else self._guard_terminal(call, view, task, directives, consulted_record, queried_players, queried_vote_rounds, lookups)
                if reason or blocked:
                    stats.retries += 1
                    why = reason or blocked["reason"]
                    step.guard_blocked, step.block_reason, step.observation = True, ("semantics" if reason else blocked["check"]), f"[guard] {why}"
                    if blocked:
                        result.guard_blocks.append({**blocked, "step": step_no})
                    self._escalate(result, step, "illegal_args" if reason else f"guard:{blocked['check']}")
                    self._file(result, step)
                    self._append_turn(messages, call, f"[guard] {why}")
                    continue
                args = term.normalise(call.arguments)
                unsupported = (evidence_mod.unsupported_citations(args["content"], speaker=view["you"], lookups=lookups)
                               if call.name in ("speak", "campaign_run", "last_words") else [])
                if unsupported and self._escalate(result, step, "unsupported_citation"):
                    # Without a cascade an unsupported citation is only counted; with one, the
                    # cheap model's speech is refused and the strong model sees why.
                    who = sorted({u["player_id"] for u in unsupported})
                    why = (f"your speech attributes statements to player(s) {who} that you did not look up this turn; "
                           "call query_history first, or do not quote them")
                    step.guard_blocked, step.block_reason, step.observation = True, "unsupported_citation", f"[guard] {why}"
                    self._file(result, step)
                    self._append_turn(messages, call, f"[guard] {why}")
                    continue
                if self.cascade is not None and self._tier == "cheap" and \
                        self.cascade.low_confidence_reply(call.name, args) and self._escalate(result, step, "low_confidence"):
                    # Not refused, asked again: the strong model gets the transcript as it was
                    # before this reply, not the reply itself.
                    step.observation = "escalated: low confidence (the speech names no seat)"
                    self._file(result, step)
                    self._abandon("escalated")
                    continue
                self._file(result, step)
                if call.name in ("speak", "campaign_run", "last_words"):
                    result.hallucinated_citations = unsupported
                result.draft_attempt = self._take_attempt(call.name)
                self._note(messages, result, stats)
                return {"name": call.name, "args": args}

            # information tool
            reason = info.spec.check(call.arguments)
            if reason:
                schema_failures += 1
                stats.retries += 1
                step.guard_blocked, step.block_reason, step.observation = True, "schema", reason
                escalated = self._escalate(result, step, "schema")
                self._file(result, step)
                if schema_failures > self.policy.max_retries and not escalated:
                    break
                self._append_turn(messages, call, f"[guard] {reason}")
                continue
            if call.name in _RECORD_TOOLS:
                consulted_record = True
            try:
                out = self.registry.execute(call.name, call.arguments, ctx)
            except SemanticError as exc:
                stats.retries += 1
                step.guard_blocked, step.block_reason, step.observation = True, "semantics", str(exc)
                self._escalate(result, step, "illegal_args")
                self._file(result, step)
                self._append_turn(messages, call, f"[guard] {exc}")
                continue
            args = info.spec.normalise(call.arguments)
            if call.name == "query_history":
                queried_players.add(args["player_id"])
                if out.data:
                    lookups[args["player_id"]] = lookups.get(args["player_id"], "") + "\n" + out.observation
            if call.name == "query_votes":
                queried_vote_rounds.add(args["round"])
            source = f"query_history:player_{args['player_id']}" if call.name == "query_history" else None
            observation, dirs, dets, _ = self.context.observation_message(call.name, out.observation, out.untrusted or info.untrusted_output, source=source)
            directives |= dirs
            result.guard_detections.extend(dets)
            step.guard_detections, step.observation = dets, observation
            self._file(result, step)
            self._append_turn(messages, call, observation)
            messages[2:] = self.context.trim_steps(messages[2:])

        # out of steps, or the model stopped answering
        return self._force_terminal(messages, spec, task, view, result, stats, step_no)

    # ------------------------------------------------------------ output-side guard
    def _guard_terminal(self, call: ToolCall, view: dict, task: str, directives: set[int], consulted_record: bool,
                        queried_players: set[int], queried_vote_rounds: set[int], lookups: dict) -> dict | None:
        name, args = call.name, call.arguments
        # chat-like turns carry no ballot and no claim to check against: the pack channel,
        # and the post-game review, which cannot move a result that is already frozen.
        if name.startswith("night_") or name in ("pack_say", "pack_pass", "pack_plan", "review_say", "review_pass"):
            return None
        if self.forensics_gate and name == "speak" and view["role"] == "werewolf" and not consulted_record:
            return {"check": "forensics", "action": name,
                    "reason": "开口之前先对账：用 query_history 或 query_votes 至少查一次记录，再发言。这张桌子上大多数狼是被票型和前后不一抓出来的，不是被验人抓出来的。"}
        if name == "speak":
            verdict = self.guard.check_evidence(str(args.get("content", "")), speaker=view["you"], current_round=view["round"],
                                                queried_players=queried_players, queried_vote_rounds=queried_vote_rounds)
            return {"check": "evidence", "reason": verdict.reason, "action": name} if verdict.blocked else None
        own = next((s["content"] for s in view["public"]["speeches"] if s["round"] == view["round"] and s["player_id"] == view["you"] and s["kind"] == "speech"), None)
        verdict = self.guard.verify_terminal(name, args, own_speech_this_round=own, injected_directives=directives,
                                             belief_evidence={p: e.evidence_refs for p, e in self.belief.entries.items()},
                                             player_id=view["you"], round_no=view["round"])
        return {"check": verdict.check, "reason": verdict.reason, "action": name} if verdict.blocked else None

    # ------------------------------------------------------------ forced terminal
    def _force_terminal(self, messages, spec: ActionSpec, task: str, view: dict, result: TurnResult, stats: RecoveryStats, step_no: int) -> dict:
        stats.forced_terminal = True
        if stats.fallback_used != "transport_failure":
            wanted = _WANTED.get(task, "night action")
            messages.append({"role": "user", "content": f"[guard] You are out of steps. Reply with exactly one `{wanted}` call now."})
            # the last chance of the turn goes to the strong model (cascade reason "forced")
            marker = ReActStep(step_no + 1, action="<forced>")
            self._escalate(result, marker, "forced")
            marker.model = self._active().name if self.cascade is not None else None
            response = self._call(messages, [t.openai_schema() for t in spec.tools], stats, kind="forced", result=result)
            if response is not None and response.tool_calls:
                call = response.tool_calls[0]
                term = spec.tool(call.name)
                if term is not None and not call.malformed and term.check(call.arguments) is None:
                    marker.args = dict(call.arguments)
                    self._file(result, marker)
                    result.draft_attempt = self._take_attempt(call.name)
                    self._note(messages, result, stats)
                    return {"name": call.name, "args": term.normalise(call.arguments)}
        stats.fallback_used = stats.fallback_used or "default_action"
        fb = spec.fallback
        self._file(result, ReActStep(step_no + 1, action=f"{fb['name']}(default)", args=dict(fb.get("args") or {}), guard_blocked=True, block_reason="fallback"))
        if stats.fallback_used != "transport_failure":
            self._note(messages, result, stats)
        return fb

    # ------------------------------------------------------------ note to the next turn
    def _note(self, messages: list[dict], result: TurnResult, stats: RecoveryStats) -> None:
        if not self.memory:
            return
        if not self.async_note:
            self._write_note(messages, result, stats, self._gen, self._scope)
            return
        msgs, gen, scope = list(messages), self._gen, self._scope

        def work():
            try:
                self._write_note(msgs, result, stats, gen, scope)
            except BaseException as exc:          # noqa: BLE001 -- re-raised by the seat's next flush()
                if gen == self._gen:
                    self._note_error = exc
        self._note_thread = threading.Thread(target=work, name=f"note-{result.player_id}", daemon=True)
        self._note_thread.start()

    def _write_note(self, messages: list[dict], result: TurnResult, stats: RecoveryStats, gen: int,
                    scope: tuple | None = None) -> None:
        """One extra call on the turn's own transcript: the model tells its next turn why.

        The environment records what the seat did and the belief table records
        what it concluded; the reasoning behind the action was thrown away with
        the ReAct chain. No tools are offered, so the reply is text. A failed or
        empty reply keeps the previous note rather than blanking it.
        """
        # The note is bookkeeping: under a cascade it is always the cheap model's job.
        response = self._call(messages + [{"role": "user", "content": NOTE_PROMPT}], None, stats, kind="note", scope=scope,
                              result=result, client=self.client)
        if gen != self._gen:
            return                                   # the turn it belonged to was thrown away
        if response is not None:
            result.prompt_tokens += response.prompt_tokens
            result.completion_tokens += response.completion_tokens
            result.latency_ms += response.latency_ms
            result.note_ms = response.latency_ms
            text = (response.text or "").strip()
            if text:
                self.note = text[:NOTE_MAX_CHARS]
        result.note_after = self.note
        # The note's own retries count toward the turn. Written again here because with
        # async_note act() snapshots the stats while the note may still be running, and
        # whether a note-call retry made it into the log depended on which thread won.
        result.recovery = stats.to_dict()

    def _file(self, result: TurnResult, step: ReActStep) -> None:
        """Record a step and let anyone watching see it as it happens.

        The callback is a read: it cannot change the step, and nothing the model
        does depends on whether anyone is watching.
        """
        result.steps.append(step)
        if step.guard_blocked and self._open_attempt is not None:
            self._abandon(f"guard:{step.block_reason}")      # refused: what was shown will not be said
        try:
            self.on_step(result.player_id, step)
        except Exception:                                  # noqa: BLE001 -- a watcher must never break a turn
            pass

    # ------------------------------------------------------------ drafts (streamed speech)
    def _draft(self, kind: str, attempt: int, text: str = "") -> None:
        try:
            self.on_draft(kind, attempt, text)
        except Exception:                                  # noqa: BLE001 -- a watcher must never break a turn
            pass

    def _abandon(self, reason: str) -> None:
        attempt, self._open_attempt = self._open_attempt, None
        if attempt is not None and self.on_draft is not None:
            self._draft("abandon", attempt, reason)

    def abandon_draft(self, attempt: int | None, reason: str) -> None:
        """The caller refused the action this attempt became (run_agent_turn's re-check)."""
        if attempt is not None and self.on_draft is not None:
            self._draft("abandon", attempt, reason)

    def _take_attempt(self, name: str) -> int | None:
        """The open draft becomes the turn's answer -- if the answer is the speech it streamed."""
        attempt, self._open_attempt = self._open_attempt, None
        if attempt is not None and name not in SPEECH_TOOLS:
            self._draft("abandon", attempt, f"not a speech: {name}")
            return None
        return attempt

    def _speech_stream(self) -> SpeechStream:
        def on_text(text: str) -> None:
            if self._open_attempt is None:
                self._attempts += 1
                self._open_attempt = self._attempts
            self._draft("delta", self._open_attempt, text)
        return SpeechStream(on_text, self._abandon)

    # ------------------------------------------------------------ one model call, metered
    def _active(self) -> LLMClient:
        """The client this turn's next step goes to: the strong one once the turn escalated."""
        return self.strong_client if self.cascade is not None and self._tier == "strong" else self.client

    def _escalate(self, result: TurnResult, step: ReActStep | None, reason: str) -> bool:
        """Move the rest of this turn to the strong model, if the policy says this reason does.

        Called at every failure the loop sees; a no-op without a cascade, on the strong
        tier already, or for a reason the policy does not escalate on. The transcript is
        not touched here: whatever the caller appends next is what the strong model reads.
        """
        if self.cascade is None or self._tier != "cheap" or not self.cascade.escalates_on(reason):
            return False
        self._tier = "strong"
        result.cascade["escalations"].append({"step": step.step if step is not None else None, "reason": reason,
                                              "from": self.client.name, "to": self.strong_client.name})
        if step is not None:
            step.escalated = reason
        return True

    def _call(self, messages: list[dict], schemas: list[dict] | None, stats: RecoveryStats, kind: str = "step",
              scope: tuple | None = None, result: TurnResult | None = None, client: LLMClient | None = None):
        """Every model call goes through here so nothing is spent unmetered.

        The request is measured from the strings we are about to send, before it
        leaves; the reply is measured from what came back. The gateway's own
        counts are recorded beside ours and never used for the cap.

        Under a cascade the call goes to the active tier (or to `client`), and is
        written into the turn's `cascade.calls` with the model that answered it:
        the cost of a game is per model, and the two tiers are not priced alike.
        """
        client = client or self._active()
        sent = estimate_messages(messages, schemas)
        if self._open_attempt is not None:
            self._abandon("retry")                 # a new call means the last one's speech was not used
        stream = self._speech_stream() if self._speechy and schemas else None
        # `kind` names the call in a recording (step | forced | note) and must not depend on
        # whether anyone watches; the savings estimate classes calls by what they cost instead.
        cost_kind = "note" if schemas is None else "speech" if stream is not None else "step"
        seat, turn, task = scope or self._scope
        try:
            with call_scope(seat, turn, task, kind):
                response = call_model(client, messages, schemas, policy=self.policy, stats=stats,
                                      temperature=self.temperature, max_tokens=self.max_tokens,
                                      on_delta=stream.feed if stream is not None else None, cancel=self.cancel)
        except Cancelled as exc:
            exc.saved = savings(client.name, cost_kind, exc.produced, sent, note_pending=cost_kind != "note",
                                max_tokens=self.max_tokens)
            raise
        if response is not None:
            expect_record(client.name, cost_kind, response.completion_tokens)
        if result is not None and result.cascade is not None:
            result.cascade["calls"].append({
                "kind": kind, "tier": "strong" if client is self.strong_client else "cheap", "model": client.name,
                "prompt": getattr(response, "prompt_tokens", 0) or 0,
                "completion": getattr(response, "completion_tokens", 0) or 0,
                "ms": getattr(response, "latency_ms", 0) or 0, "ok": response is not None})
        if self.token_budget is not None:
            got = estimate_tokens(getattr(response, "text", "") or "")
            for call in getattr(response, "tool_calls", None) or []:
                got += estimate_tokens(call.name) + estimate_tokens(call.raw_arguments or "")
            self.token_budget.spend(
                prompt=sent, completion=got,
                gateway_prompt=getattr(response, "prompt_tokens", 0) or 0,
                gateway_completion=getattr(response, "completion_tokens", 0) or 0)
        return response

    # ------------------------------------------------------------ transcript
    def _append_turn(self, messages: list[dict], call: ToolCall, observation: str) -> None:
        if getattr(self.client, "tool_mode", "native") == "native":    # both cascade tiers share it (__init__)
            messages.append({"role": "assistant", "content": None,
                             "tool_calls": [{"id": call.id, "type": "function",
                                             "function": {"name": call.name, "arguments": call.raw_arguments or json.dumps(call.arguments)}}]})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})
        else:
            messages.append({"role": "assistant", "content": call.raw_arguments or json.dumps({"action": call.name, "args": call.arguments})})
            messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def _thought_of(response) -> str:
    """The reasoning out of a reply, whatever wrapper it arrived in.

    The JSON-protocol path often fences its object in a markdown block, and a
    reader should see the thought, not the envelope it travelled in.
    """
    text = _FENCE.sub("", (response.text or "").strip()).strip()
    if text.startswith("{"):
        try:
            return str(json.loads(text).get("thought", ""))[:400]
        except Exception:  # noqa: BLE001
            return text[:400]
    return text[:400]
