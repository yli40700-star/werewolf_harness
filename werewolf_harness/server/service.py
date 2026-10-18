"""Games as long-lived objects: created, joined, watched, replayed.

One thread per game runs the same driver loop the CLI runs. HTTP threads only
ever read -- through a lock, and only through `observe()`, so a spectator is
just another seat's view rather than a second way into the world.
"""
from __future__ import annotations

import json
import itertools
import random
import threading
from pathlib import Path
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from ..arena import ActionSpec, Record, StepLedger, Terminal, Turn
from ..envs import VARIANTS, Werewolf, engine_version_of
from ..players import (BudgetExhausted, GatewayExhausted, GuardStack, HumanPlayer, LLMPlayer,
                       MockClient, OpenAICompatClient, RandomPlayer, RecoveryPolicy, TokenBudget,
                       TransportBudget)
from ..players.providers import PROVIDERS, ClaudeCLIClient, is_cli_model, provider_of, profile, retired
from ..players.providers.mock import STREAM_PROFILES

_ids = itertools.count(1)

# Where a running game leaves the journal that `cli resume` picks up.
SAVE_DIR = Path(__file__).resolve().parent.parent.parent / "games" / "resume"

# A resumed game has already spent whatever it spent; this is the room the rest
# of it gets. Measured: a whole 12p game runs about 2.4M.
DEFAULT_RESUME_TOKENS = 6_000_000


# A raw HTTP body is not an answer to "what do I do now".
_MONEY = ("insufficient balance", "insufficient account balance", "billing", "quota", "402")
_KEY = ("invalid api key", "incorrect api key", "unauthorized", "401")


def _in_plain_words(err: str) -> str:
    """Say what went wrong and where to go about it.

    A game that stops because an account ran dry should not present the vendor's
    JSON to whoever is watching; it should name the vendor and the page that
    fixes it.
    """
    low = err.lower()
    for prov in PROVIDERS.values():
        if prov.base_url.split("//")[-1].split("/")[0] not in low:
            continue
        if any(k in low for k in _MONEY):
            where = prov.topup or prov.console
            return f"{prov.label} 余额不够了，这一局停在这里。去充值：{where}\n（原文：{err[:160]}）"
        if any(k in low for k in _KEY):
            return f"{prov.label} 的 key 不对或已失效。去看看：{prov.console}\n（原文：{err[:160]}）"
        return f"{prov.label} 拒绝了这次请求：{err[:200]}"
    return err


def _public_action(record) -> dict:
    """One journal row with the private half taken out.

    A seat's own reasoning is its own: `thought` never travels to another player,
    not even on an action whose result the whole table saw.
    """
    action = record.action
    if not isinstance(action, dict):
        return action
    args = {k: v for k, v in (action.get("args") or {}).items() if k != "thought"}
    return {**action, "args": args}


def _client(model: str, seed: int, env_keys: dict):
    """The client for one seat.

    Three worlds: a model reached through the relay (one base_url, keys grouped by
    vendor), a Chinese model reached directly (its own base_url, its own key,
    its own body quirks), and the local Claude Code CLI (`claude-cli:<model>`,
    no key -- whatever the CLI is signed in with). The seat cannot tell the difference.
    """
    if model == "mock" or model in STREAM_PROFILES:
        return MockClient.named(model, seed)
    if is_cli_model(model):
        return ClaudeCLIClient(model, env=env_keys)
    p = profile(model)
    direct = provider_of(model)
    if direct is not None:
        key = env_keys.get(direct.env_key, "")
        if not key:
            raise ValueError(f"{model} 要 {direct.label} 的 key：在 .env 里设 {direct.env_key}（{direct.console}）")
        base = env_keys.get(f"{direct.env_key}_BASE_URL") or direct.base_url
    else:
        group = "CLAUDE" if model.startswith("claude") else "GEMINI" if model.startswith("gemini") else "OPENAI"
        key = env_keys.get(f"LLM_KEY_{group}_GROUP") or env_keys.get("LLM_API_KEY", "")
        base = env_keys.get("LLM_BASE_URL", "https://api.openai.com/v1")
    return OpenAICompatClient(model, api_key=key, base_url=base, tool_mode=p["tool_mode"],
                              tool_choice=p.get("tool_choice", "required"),
                              extra_body=p.get("extra_body"), temperature=p.get("temperature"),
                              stream=p.get("stream", True))


@dataclass
class Event:
    seq: int
    kind: str                       # turn | phase | end | error
    data: dict
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"seq": self.seq, "kind": self.kind, "ts": round(self.ts, 3), **self.data}


class Game:
    """One table. Everything a client can ask about it comes from here."""

    def __init__(self, *, variant: str, model: str | list[str], guard: str, playbook: str,
                 human_seats: list[int], turn_seconds: float | None, seed: int,
                 max_tokens: int, env_keys: dict, xray: bool = False, engine_version: int | None = None):
        self.id = f"g{next(_ids)}"
        models = [model] if isinstance(model, str) else list(model)
        self.models = models or ["mock"]
        self.model = self.models[0] if len(self.models) == 1 else "混合"
        self.variant, self.guard_label, self.playbook = variant, guard, playbook
        self.seed = seed
        self.n = VARIANTS[variant].num_players
        self.human_seats = [s for s in human_seats if 1 <= s <= self.n]
        self.turn_seconds = turn_seconds
        self.created = time.time()
        # 测试模式: every role and every thought is visible to whoever opens the page.
        # A deliberate switch, never a default -- what a seat may see is otherwise
        # decided by observe() alone, and this is the one door around it.
        self.xray = bool(xray)
        self.resumed_from: str | None = None      # set when this table continues another
        self.saved_to: str | None = None          # where the journal was last written
        self.acting: dict | None = None
        self._resumed_at = None          # set by replay_into: where the loop picks up
        self.traces: dict[int, list[dict]] = {}
        self.state = "waiting" if self.human_seats else "running"
        self.error: str | None = None

        self.env = Werewolf(variant, playbook_mode=playbook, engine_version=engine_version)   # ADR 0010
        self.lock = threading.RLock()
        self.ledger = StepLedger(game=self.id)
        self.journal: list[Record] = []
        self.events: list[Event] = []
        self.bell = threading.Condition()
        self.outcome: dict | None = None

        guards = tuple(x for x in guard.split("+") if x and x != "无")
        stack = GuardStack(guards)
        self.transport = TransportBudget()
        self.tokens = TokenBudget(max_tokens=max_tokens)

        # Which model sits where. Round-robin over a seeded shuffle so a mixed
        # table does not always hand seat 1 the same model -- otherwise "the
        # first speaker is always gemini" becomes a property of the batch.
        ai_seats = [s for s in range(1, self.n + 1) if s not in self.human_seats]
        rota = list(self.models)
        random.Random(seed).shuffle(rota)
        self.seat_models: dict[int, str] = {s: rota[i % len(rota)] for i, s in enumerate(ai_seats)}

        self.humans: dict[int, HumanPlayer] = {}
        self.players: dict[int, Any] = {}
        for s in range(1, self.n + 1):
            if s in self.human_seats:
                p = HumanPlayer(s, turn_seconds=turn_seconds, on_change=self._ping)
                self.humans[s] = p
            else:
                m = self.seat_models[s]
                if m == "script":
                    p = RandomPlayer(seed * 100 + s)
                else:
                    pol = RecoveryPolicy(timeout_s=profile(m)["timeout_s"])
                    p = LLMPlayer(_client(m, seed * 100 + s, env_keys), guard=stack, policy=pol,
                                  transport_budget=self.transport, token_budget=self.tokens,
                                  playbook=playbook != "none", on_step=self._on_step)
            self.players[s] = p

        self._thread: threading.Thread | None = None

    # ---------------------------------------------------------------- events
    def _emit(self, kind: str, **data) -> None:
        with self.bell:
            self.events.append(Event(len(self.events), kind, data))
            self.bell.notify_all()

    SAVE_EVERY = 25          # turns between autosaves; a lost tail is cheap, a lost game is not

    def _save(self) -> None:
        """Write the journal where `cli resume` can find it. Never fatal."""
        try:
            path = SAVE_DIR / f"{self.id}_{self.variant}_{self.playbook}_seed{self.seed}.json"
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.full_log(seat=None), ensure_ascii=False), encoding="utf-8")
            self.saved_to = str(path)
        except Exception:                                  # noqa: BLE001 -- a failed save must not end a game
            pass

    def _on_step(self, seat: int, step) -> None:
        """One ReAct step just landed. Keep it, and wake anyone watching."""
        row = {"n": step.step, "action": step.action, "args": step.args,
               "observation": (step.observation or "")[:300], "thought": (step.thought or "")[:300],
               "blocked": step.block_reason, "ms": step.latency_ms}
        self.traces.setdefault(seat, []).append({**row, "round": getattr(self.env, "_round", 0)})
        self.traces[seat] = self.traces[seat][-40:]
        if self.acting and self.acting.get("actor") == seat:
            self.acting.setdefault("steps", []).append(step.action)
        self._ping()

    def _ping(self) -> None:
        with self.bell:
            self.bell.notify_all()

    def wait(self, after: int, timeout: float = 25.0) -> list[Event]:
        with self.bell:
            if len(self.events) <= after:
                self.bell.wait(timeout)
            return self.events[after:]

    # ---------------------------------------------------------------- running
    def start(self) -> None:
        if self._thread:
            return
        self.state = "running"
        self._thread = threading.Thread(target=self._run, name=f"game-{self.id}", daemon=True)
        self._thread.start()
        self._emit("phase", phase="start", seed=self.seed)

    def replay_into(self, journal) -> None:
        """Rebuild this table's world from a record of what was already played.

        No model is asked anything: every action is read back from the journal
        and applied. That is what `world == f(seed, journal)` buys -- a game the
        budget or the gateway cut short can be picked up where it stopped,
        without paying twice for the half that already happened.
        """
        with self.lock:
            turn = self.env.reset(self.seed)
            for rec in journal:
                if not isinstance(turn, Turn):
                    raise ValueError("日志比这局还长：重放到终局之后还有记录")
                if (turn.actor, turn.task) != (rec.actor, rec.task):
                    raise ValueError(f"日志对不上：重放到 {turn.actor}/{turn.task}，"
                                     f"记录里是 {rec.actor}/{rec.task}")
                record, turn = self.ledger.apply(self.env, turn, rec.action, forced=rec.forced)
                self.journal.append(record)
            self._resumed_at = turn
        self._emit("phase", phase="resume", round=getattr(self.env, "_round", 0),
                   turns=len(self.journal))

    def _run(self) -> None:
        try:
            with self.lock:
                turn = self._resumed_at if self._resumed_at is not None else self.env.reset(self.seed)
                self._resumed_at = None
            while isinstance(turn, Turn):
                with self.lock:
                    view = self.env.observe(turn.actor)
                    spec = self.env.action_space(turn.actor, turn.task)
                self.acting = {"actor": turn.actor, "task": turn.task, "since": time.time(),
                               "human": turn.actor in self.humans,
                               "model": self.seat_models.get(turn.actor, "真人"), "steps": []}
                self._emit("phase", phase=self.env._phase, round=self.env._round,
                           actor=turn.actor, task=turn.task,
                           human=turn.actor in self.humans)
                action, forced = self._elicit(turn, view, spec)
                self.acting = None
                with self.lock:
                    before = self._snap()
                    record, turn = self.ledger.apply(self.env, turn, action, forced=forced)
                    self.journal.append(record)
                    news = self._news(before)
                self._emit("turn", actor=record.actor, task=record.task,
                           action=record.action, forced=record.forced,
                           n=len(self.journal), alive=sorted(self.env._alive),
                           round=self.env._round, phase=self.env._phase)
                for line in news:
                    self._emit("news", text=line)
                # The journal on disk is what makes a dead game resumable, and the
                # ways a game dies -- a dropped network, a budget, a kill -- are
                # exactly the ways nobody is left to write it. So write as we go.
                if len(self.journal) % self.SAVE_EVERY == 0:
                    self._save()
            self.acting = None
            self.outcome = turn.outcome
            self.state = "over"
            self._emit("end", outcome=turn.outcome, health=self.health())
        except (GatewayExhausted, BudgetExhausted) as exc:
            self.state, self.error = "aborted", _in_plain_words(str(exc))
            self._emit("error", message=self.error)
        except Exception as exc:                              # noqa: BLE001 -- a crash is a datum
            self.state, self.error = "crashed", f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            self._emit("error", message=self.error)
        finally:
            # However the run ended, nobody is on the clock any more. Without
            # this the page keeps showing the last seat "still thinking", with
            # the timer climbing, on a game that stopped minutes ago.
            self.acting = None
            self._save()

    def _snap(self) -> dict:
        e = self.env
        return {"dead": {d.player_id for d in e._deaths}, "round": e._round,
                "phase": e._phase, "exiles": dict(e._exiles), "sheriff": e._sheriff}

    def _news(self, before: dict) -> list[str]:
        """What the table just learned, in the words the table would use.

        Only public facts: who died, never how; what the ballot was, never who
        is what. The same rule the environment applies to `observe`.
        """
        e = self.env
        out: list[str] = []
        fresh = [d for d in e._deaths if d.player_id not in before["dead"]]
        for d in fresh:
            how = {"vote": "被放逐", "hunter": "被猎人带走", "detonate": "自爆"}.get(d.cause, "夜里出局")
            out.append(f"第 {d.round} 轮：{d.player_id} 号{how}。")
        for r, ex in e._exiles.items():
            if r not in before["exiles"]:
                counts = {}
                for v, t in (e._votes.get(r) or {}).items():
                    counts[t] = counts.get(t, 0) + 1
                tally = "、".join(f"{t} 号 {n} 票" for t, n in sorted(counts.items(), key=lambda x: -x[1])) or "无人投票"
                out.append(f"第 {r} 轮票型：{tally} → " + (f"{ex} 号出局。" if ex is not None else "平票，无人出局。"))
        if before["phase"] != e._phase and e._phase == "day_speech" and not fresh:
            out.append(f"第 {e._round} 天天亮：平安夜，没有人死。")
        if before["phase"] != e._phase and e._phase == "review":
            # the winner is already fixed at this point; the review cannot move it
            side = "狼人阵营" if e._winner == "werewolf" else "好人阵营"
            out.append(f"本局结束：{side}获胜。全场底牌公开，开始复盘。")
        if before["sheriff"] != e._sheriff and e._sheriff:
            out.append(f"{e._sheriff} 号当选警长。")
        return out

    def _elicit(self, turn: Turn, view: dict, spec: ActionSpec, max_retries: int = 3):
        player = self.players[turn.actor]
        feedback = None
        for _ in range(max_retries):
            action = player.act(view, turn.task, spec, feedback)
            reason = spec.check(action)
            if reason is None:
                # a legal action is not automatically a decision: a seat that ran out
                # of clock also returns a legal action -- the environment's fallback
                return action, bool(getattr(player, "last_forced", False))
            feedback = reason
        return spec.fallback, True

    # ---------------------------------------------------------------- reads
    def join(self, seat: int) -> dict:
        if seat not in self.humans:
            return {"ok": False, "error": "这个座位不是留给真人的"}
        if self.state == "waiting" and all(s in self.humans for s in self.human_seats):
            pass
        return {"ok": True, "seat": seat, "game": self.id}

    def health(self) -> dict:
        turns = len(self.journal)
        forced = sum(r.forced for r in self.journal)
        by_model: dict[str, dict] = {}
        for seat, m in self.seat_models.items():
            p = self.players.get(seat)
            hist = getattr(p, "history", []) or []
            row = by_model.setdefault(m, {"seats": [], "turns": 0, "forced": 0})
            row["seats"].append(seat)
            row["turns"] += len(hist)
            row["forced"] += sum(1 for h in hist if getattr(h, "fallback_used", None) or (isinstance(h, dict) and h.get("fallback_used")))
        return {"by_model": by_model, "turns": turns, "forced": forced,
                "forced_rate": round(forced / turns, 4) if turns else 0.0,
                "usable": (forced / turns if turns else 0) <= 0.05,
                "replays": self.ledger.replays,
                "transport_failures": self.transport.failures,
                "tokens": self.tokens.total_tokens, "max_tokens": self.tokens.max_tokens,
                "gateway_drift": self.tokens.drift()}

    def summary(self) -> dict:
        with self.lock:
            alive = sorted(self.env._alive) if hasattr(self.env, "_alive") else []
            rnd, phase = getattr(self.env, "_round", 0), getattr(self.env, "_phase", "-")
        return {"id": self.id, "variant": self.variant, "model": self.model,
                "models": self.models, "seat_models": self.seat_models, "guard": self.guard_label,
                "playbook": self.playbook, "state": self.state, "seed": self.seed,
                "seats": self.n, "human_seats": self.human_seats, "alive": alive,
                "round": rnd, "phase": phase, "reviewing": phase == "review",
                "outcome": self.outcome, "error": self.error,
                "xray": self.xray, "acting": self._acting_now(),
                "resumed_from": self.resumed_from, "saved_to": self.saved_to,
                "engine_version": self.env.engine_version,
                "health": self.health(), "created": round(self.created, 3)}

    def seat_state(self, seat: int | None) -> dict:
        """Everything one client needs: the public record it may see, and its own turn."""
        out: dict[str, Any] = {"game": self.summary(), "seat": seat}
        out["acting"] = self._acting_now()
        # Three doors to the roles, and no fourth: the game is over, this seat is
        # out of it, or the table was opened in 测试模式. A living seat never sees
        # them -- that is what makes the rest of the table worth reading.
        if self._may_see_all(seat):
            out["revealed"] = dict(self.env._roles) if hasattr(self.env, "_roles") else {}
            out["traces"] = {str(k): v[-12:] for k, v in self.traces.items()}
        # the review outlives its phase: once it is over, `observe` closes again, so the
        # page reads the transcript from the environment's read-only log instead
        out["review"] = self.env.review_log() if hasattr(self.env, "review_log") else []
        with self.lock:
            if seat and 1 <= seat <= self.n and hasattr(self.env, "_alive"):
                out["view"] = self.env.observe(seat)
            elif hasattr(self.env, "_alive"):
                out["view"] = self._spectator_view()
        if seat in self.humans:
            out["me"] = self.humans[seat].snapshot()
        out["log"] = [e.to_dict() for e in self.events[-60:]]
        return out

    def _may_see_all(self, seat: int | None) -> bool:
        # A fourth door, and it is the environment's own: during 复盘 the environment
        # itself hands every seat the whole truth, so the page showing it reveals nothing
        # `observe` has not already revealed.
        if self.xray or self.state in ("over", "aborted", "crashed") or self._reviewing():
            return True
        if seat and hasattr(self.env, "_alive"):
            return seat not in self.env._alive          # out of the game: the table is open to you
        return False

    def _reviewing(self) -> bool:
        return getattr(self.env, "_phase", None) == "review"

    def _acting_now(self) -> dict | None:
        a = self.acting
        if not a:
            return None
        return {"actor": a["actor"], "task": a["task"], "human": a["human"], "model": a["model"],
                "seconds": round(time.time() - a["since"], 1), "steps": list(a.get("steps") or [])}

    def _spectator_view(self) -> dict:
        """A watcher gets exactly the public record -- no seat's private half."""
        any_seat = next(iter(range(1, self.n + 1)))
        v = dict(self.env.observe(any_seat))
        v.pop("private", None)
        v.pop("night", None)
        v["role"] = None
        v["you"] = None
        return v

    # What a living player is allowed to read back. The day is public: who spoke,
    # who voted for whom, who ran for sheriff. The night is not, and neither is
    # anybody's reasoning -- `thought` rides along with every terminal action now,
    # so an ungated journal would hand a villager the wolves' plan verbatim.
    PUBLIC_TASKS = frozenset({"speak", "vote", "campaign", "campaign_vote",
                              "last_words", "hunter_shoot", "badge"})

    def full_log(self, seat: int | None = None) -> dict:
        """The whole game, for reading afterwards or over someone's shoulder.

        Gated exactly like `seat_state`: a seat still in a running game reads the
        public half only. `_may_see_all` opens the rest -- the game is over, this
        seat is out of it, or the table is in 测试模式.
        """
        wide = self._may_see_all(seat)
        with self.lock:
            pub = self._spectator_view()["public"] if hasattr(self.env, "_alive") else {}
            journal = [{"n": i + 1, "actor": r.actor, "task": r.task,
                        "action": r.action if wide else _public_action(r),
                        "forced": r.forced, "model": self.seat_models.get(r.actor, "真人")}
                       for i, r in enumerate(self.journal)
                       if wide or r.task in self.PUBLIC_TASKS or r.actor == seat]
        return {"game": self.summary(), "public": pub, "journal": journal,
                "traces": ({str(k): v for k, v in self.traces.items()} if wide else {}),
                "news": [e.to_dict() for e in self.events if e.kind == "news"],
                "review": self.env.review_log() if hasattr(self.env, "review_log") else [],
                "revealed": dict(self.env._roles)
                if (self.state in ("over", "aborted", "crashed") or self._reviewing()) else None}

    def submit(self, seat: int, seq: int, action: Any) -> dict:
        p = self.humans.get(seat)
        if p is None:
            return {"status": "not_a_human_seat"}
        return p.submit(seq, action)


class GameService:
    def __init__(self, env_keys: dict | None = None):
        self.games: dict[str, Game] = {}
        self.env_keys = env_keys or {}

    def create(self, **kw) -> Game:
        # A retired model is not a typo to be corrected silently -- say which
        # seat asked for it and why it is off the menu.
        wanted = kw.get("model") or []
        for m in ([wanted] if isinstance(wanted, str) else list(wanted)):
            why = retired(m)
            if why:
                raise ValueError(f"{m} 已经下架（{why}）")
        g = Game(env_keys=self.env_keys, **kw)
        self.games[g.id] = g
        if not g.human_seats:
            g.start()
        return g

    def get(self, gid: str) -> Game | None:
        return self.games.get(gid)

    def resume_saved(self, log: dict, *, max_tokens: int | None = None) -> Game:
        """Carry on a game from a saved log rather than from memory.

        The service keeps games in RAM, so a restart loses them -- and a restart
        is exactly what you do after fixing whatever cut the game short. The log
        on disk is the same journal, so it resumes the same way.
        """
        g = log.get("game") or {}
        records = [Record(r["actor"], r["task"], r["action"], bool(r.get("forced")))
                   for r in log.get("journal") or []]
        if not records:
            raise ValueError("这份日志里没有回合，直接重开就行")
        if any(r.action is None or "name" not in (r.action or {}) for r in records):
            raise ValueError("这份日志的动作不全（多半是没权限那半份），换一份完整的")
        game = Game(variant=g.get("variant", "12p"), model=list(g.get("models") or ["mock"]),
                    guard=g.get("guard", "L1+L2+L3"), playbook=g.get("playbook", "static"),
                    human_seats=list(g.get("human_seats") or []), turn_seconds=None,
                    seed=int(g["seed"]), max_tokens=max_tokens or DEFAULT_RESUME_TOKENS,
                    env_keys=self.env_keys, xray=bool(g.get("xray")), engine_version=engine_version_of(g))
        game.resumed_from = g.get("id", "存档")
        game.replay_into(records)
        self.games[game.id] = game
        game.start()
        return game

    def resume(self, gid: str, *, max_tokens: int | None = None) -> Game:
        """Carry on a game that was cut short, without replaying the cost.

        A new table is dealt from the same seed and the old journal is replayed
        into it, which costs nothing -- then the loop continues from the exact
        turn that was interrupted. The old game is kept as it was; this is a new
        row that says where it came from.
        """
        old = self.games.get(gid)
        if old is None:
            raise ValueError(f"没有这局：{gid}")
        if old.state not in ("aborted", "crashed"):
            raise ValueError(f"{gid} 现在是 {old.state}，只有中断的局才需要接着跑")
        if not old.journal:
            raise ValueError(f"{gid} 一个回合都没走完，直接重开就行")
        g = Game(variant=old.variant, model=list(old.models), guard=old.guard_label,
                 playbook=old.playbook, human_seats=list(old.human_seats),
                 turn_seconds=old.turn_seconds, seed=old.seed,
                 max_tokens=max_tokens or old.tokens.max_tokens * 3,
                 env_keys=self.env_keys, xray=old.xray, engine_version=old.env.engine_version)
        g.resumed_from = gid
        g.replay_into(old.journal)
        self.games[g.id] = g
        g.start()
        return g

    def list(self) -> list[dict]:
        return [g.summary() for g in sorted(self.games.values(), key=lambda x: -x.created)]
