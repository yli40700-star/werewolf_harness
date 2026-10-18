"""Command line.

    python -m werewolf_harness.cli play   --variant 12p --seed 8 --model mock
    python -m werewolf_harness.cli play   --variant 12p --seed 8 --model <gateway model> --guard L1,L2,L3 --out game.json
    python -m werewolf_harness.cli replay game.json          # rebuild the world from the journal, compare outcomes
    python -m werewolf_harness.cli play ... --record --out g.json   # also keep every model call (g.cassette.jsonl)
    python -m werewolf_harness.cli replay-llm g.json         # re-run the agents on the recorded replies, no model
    python -m werewolf_harness.cli random --variant 12p --seeds 20   # scripted seats, no model: exercises the rules
    python -m werewolf_harness.cli play --model claude-cli:haiku --strong claude-cli:sonnet --cascade both ...
                                         # model cascade: cheap by default, strong on failure / decisive turns

Models other than "mock" need LLM_API_KEY (and LLM_BASE_URL for a relay).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .arena import Record, Terminal, replay, run
from .envs import ENGINE_VERSION, VARIANTS, VIEW_REVISION, Werewolf, engine_version_of
from .players import (BudgetExhausted, GatewayExhausted, GuardStack, LLMPlayer, MockClient,
                      OpenAICompatClient, RandomPlayer, TokenBudget, TransportBudget)


def _client(model: str, seed: int):
    if model == "mock":
        return MockClient(seed=seed)
    from .players.providers.mock import cascade_mock
    offline = cascade_mock(model, seed)            # mock-strong, mock-flaky[:bad_rate]: a cascade's offline tiers
    if offline is not None:
        return offline
    if model.startswith("mock-slow"):              # mock-slow:300 -> a mock that takes 300 ms a call
        ms = int(model.split(":", 1)[1]) if ":" in model else 300
        return MockClient(seed=seed, sleep_ms=ms, latency_ms=ms)
    if model.startswith("mock-chaos"):             # mock-chaos[:error_rate[:malformed_rate]] -- only a recording replays it
        from .players.providers.mock import ChaosMockClient
        rates = [float(x) for x in model.split(":")[1:3]]
        return ChaosMockClient(seed=seed, **dict(zip(("error_rate", "malformed_rate"), rates)))
    from .players.providers.claude_cli import ClaudeCLIClient, is_cli_model
    if is_cli_model(model):
        return ClaudeCLIClient(model)
    if model.startswith(("anthropic:", "anthropic-nocache:")):
        # the Messages API directly: anthropic:<model> places cache breakpoints (players/providers/anthropic_api.py),
        # anthropic-nocache:<model> sends none -- the uncached baseline
        from .players.providers.anthropic_api import AnthropicClient
        from .server.http import read_env
        kind, name = model.split(":", 1)
        c = AnthropicClient(name, read_env().get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY", ""),
                            breakpoints=kind == "anthropic")
        c.name = model
        return c
    key = os.getenv("LLM_API_KEY", "")
    base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    return OpenAICompatClient(model, api_key=key, base_url=base, tool_mode=os.getenv("LLM_TOOL_MODE", "native"))


def play_settings(a: argparse.Namespace) -> dict:
    """What shapes the seats' prompts, saved in the log so `replay-llm` can rebuild them."""
    s = {"guard": a.guard, "evidence": a.evidence, "playbook": a.playbook, "temperature": a.temperature,
         "lang": a.lang, "max_tokens": a.max_tokens, "max_usd": a.max_usd, "parallel": a.parallel}
    s["invoke_calls"] = True        # read <invoke> replies as calls; logs without it replay as they were played
    # experiment switches, saved only when on: a log played without them keeps its old settings
    if getattr(a, "wolf_model", None):
        s["wolf_model"] = a.wolf_model
    if getattr(a, "wolf_hint", None):
        s["wolf_hint"] = a.wolf_hint
    if getattr(a, "no_memory", False):
        s["memory"] = False
    policy = cascade_policy(a)
    if policy is not None:
        # only when on: a log played without a cascade keeps exactly the settings it always had
        s.update({"cascade": policy.to_dict(), "cascade_mode": policy.mode, "strong": a.strong})
    return s


def cascade_policy(a: argparse.Namespace):
    """--cascade MODE [--decisive t1,t2] [--low-confidence] -> CascadePolicy, or None."""
    from .players.llm.cascade import CascadePolicy
    mode = getattr(a, "cascade", "off") or "off"
    if mode != "off" and not getattr(a, "strong", None):
        raise SystemExit("!! --cascade 需要 --strong <强模型>")
    decisive = [x for x in a.decisive.split(",") if x] if getattr(a, "decisive", None) else None
    return CascadePolicy.from_mode(mode, decisive=decisive, low_confidence=getattr(a, "low_confidence", False))


def cmd_play(a: argparse.Namespace) -> int:
    from .replay_llm import build_players
    env = Werewolf(a.variant, playbook_mode=a.playbook, engine_version=a.engine_version)
    n = VARIANTS[a.variant].num_players
    settings = play_settings(a)
    cassette = None
    if a.record:
        if not a.out:
            print("!! --record 要配 --out：录像写在日志旁边", file=sys.stderr)
            return 4
        from .players.providers.recorded import CassetteWriter, RecordingClient, cassette_path_for
        cassette = CassetteWriter(cassette_path_for(a.out), meta={"variant": a.variant, "seed": a.seed,
                                                                  "model": a.model, "strong": settings.get("strong"),
                                                                  "settings": settings,
                                                                  "started": time.time()})

    wolves = set()
    if settings.get("wolf_model"):
        probe = Werewolf(a.variant, playbook_mode=a.playbook, engine_version=a.engine_version)
        probe.reset(a.seed)
        wolves = {int(k) for k, v in probe.ground_truth()["roles"].items() if v == "werewolf"}

    def client_for(s: int, model: str | None = None, tier: str | None = None):
        model = model or (a.wolf_model if s in wolves else a.model)
        c = _client(model, a.seed * 100 + s)
        return RecordingClient(c, cassette, seat=s, tier=tier) if cassette is not None else c

    players, tokens, guard = build_players(settings, n, client_for,
                                           strong_for=lambda s: client_for(s, a.strong, "strong"))
    t0 = time.time()
    crashed = None
    try:
        ep = run(env, players, a.seed, parallel=a.parallel)
    except GatewayExhausted as exc:
        crashed = str(exc)
        print(f"!! 网关不再作答：{exc}", file=sys.stderr)
        return 2
    except BudgetExhausted as exc:
        crashed = str(exc)
        print(f"!! 预算用尽：{exc}", file=sys.stderr)
        return 3
    finally:
        if cassette is not None:
            cassette.close()
            # a game that died still leaves its calls: the cassette is the record of how
            print(f"   录像：{cassette.calls} 次模型调用 -> {cassette.path}")
    dur = time.time() - t0
    truth = env.ground_truth()
    print(f"== {a.variant} seed={a.seed} model={a.model} guard={guard.label()} engine=v{env.engine_version} -> {ep.outcome['winner']} in round {ep.outcome['round']} "
          f"({len(ep.journal)} turns, {dur:.0f}s, parallel={a.parallel}"
          + (f", {ep.misses} answers asked too early and thrown away" if ep.misses else "") + ")")
    h = ep.health
    print(f"   roles: {truth['roles']}")
    print(f"   健康度：{h.forced}/{h.turns} 回合兜底（{h.forced_rate:.1%}）"
          + ("  ✓ 可用" if h.usable else "  !! 超过 5%，这一批不能用"))
    print(f"   用量：本地估算 {tokens.total_tokens} tokens / 上限 {tokens.max_tokens}"
          + (f"，约 ${tokens.usd:.2f}" if any(tokens.price_per_1m) else "")
          + (f"；网关自报是本地口径的 {tokens.drift()}×" if tokens.drift() else ""))
    cascade = cascade_summary([t for p in players.values() for t in p.history]) if "cascade" in settings else None
    if cascade is not None:
        reasons = "、".join(f"{k} {v}" for k, v in sorted(cascade["escalations"].items())) or "无"
        print(f"   级联（{settings['cascade_mode']}，{a.model} → {a.strong}）：{cascade['strong_turns']}/{cascade['turns']} 回合由强模型作答；"
              f"调用 {cascade['calls']}（强 {cascade['strong_calls']}）；升级原因：{reasons}")
    if a.out:
        log = {
            # the rules this journal was written under: replay must use the same ones
            "engine_version": env.engine_version, "view_revision": VIEW_REVISION,
            "variant": a.variant, "seed": a.seed, "model": a.model, "guard": guard.label(), "playbook": a.playbook,
            "outcome": ep.outcome, "ground_truth": truth, "duration_s": dur, "crashed": crashed,
            "health": ep.health.to_dict(), "usage": tokens.to_dict(),
            "journal": [r.__dict__ for r in ep.journal],
            "turns": [t.to_dict() for p in players.values() for t in p.history],
            "night": env.night_records(), "pack_chat": env.pack_chat_log(),
            "parallel": a.parallel, "timings": ep.timings, "misses": ep.misses,
            "settings": settings,
        }
        if cassette is not None:
            log["cassette"] = os.path.basename(cassette.path)
        if cascade is not None:
            log.update({"strong": a.strong, "cascade": settings["cascade_mode"], "cascade_summary": cascade})
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(log, f, ensure_ascii=False, indent=1, default=str)
        print(f"   wrote {a.out}")
    return 0


def cascade_summary(turns: list) -> dict:
    """Per game: how many turns and calls each tier took, and the escalations by reason."""
    from collections import Counter
    rows = [t.cascade for t in turns if t.cascade is not None]
    calls = [c for r in rows for c in r["calls"]]
    return {"turns": len(rows), "strong_turns": sum(1 for r in rows if r["tier"] == "strong"),
            "escalated_turns": sum(1 for r in rows if r["escalations"]),
            "escalations": dict(Counter(e["reason"] for r in rows for e in r["escalations"])),
            "calls": len(calls), "strong_calls": sum(1 for c in calls if c["tier"] == "strong"),
            "tokens_by_model": dict(sum((Counter({c["model"]: c["prompt"] + c["completion"]}) for c in calls), Counter()))}


def cmd_replay_llm(a: argparse.Namespace) -> int:
    """Re-run a recorded game through the real LLMPlayer stack; the replies come from the cassette."""
    from .replay_llm import compare, load, replay_game
    try:
        log, cas = load(a.file, a.cassette)
    except FileNotFoundError as exc:
        print(f"!! {exc}", file=sys.stderr)
        return 2
    rep = replay_game(log, cas, allow_drift=a.allow_drift, upto=a.upto)
    diffs = compare(log, rep, upto=a.upto)
    if a.strict and not diffs and a.upto is None and len(rep.used) != rep.recorded:
        unused = [k for k in cas.order if k not in rep.used][:5]
        diffs.append(f"录像里有 {rep.recorded - len(rep.used)} 次调用回放时没用到，例如 {unused}")
    orig = log.get("duration_s")
    print(f"== replay-llm {a.file}: 从录像答了 {rep.calls} 次调用（录像共 {rep.recorded} 次），"
          f"真实模型调用 0 次，用时 {rep.ms:.0f} ms" + (f"（录制这局用了 {orig:.1f} s）" if orig else ""))
    for d in rep.drifts[:10]:
        print(f"   提示词变了 @ {d.key}\n      " + d.diff.replace("\n", "\n      "))
    if len(rep.drifts) > 10:
        print(f"   …… 一共 {len(rep.drifts)} 处提示词变化")
    if diffs:
        for d in diffs:
            print("   DIVERGED: " + d)
        return 1
    what = f"前 {a.upto} 步 journal" if a.upto is not None else "journal、结局、每个回合的信念和笔记"
    print(f"   {what}一致 ✓" + (f"（{len(rep.drifts)} 处提示词变化，录下的回复仍然适用）" if rep.drifts else ""))
    return 0


def cmd_replay(a: argparse.Namespace) -> int:
    log = json.load(open(a.file))
    # a log from before versioning has no engine_version: it was played under the baseline rules
    env = Werewolf(log["variant"], playbook_mode=log.get("playbook", "static"), engine_version=engine_version_of(log))
    journal = [Record(**r) for r in log["journal"]]
    end = replay(env, log["seed"], journal, upto=a.upto if a.upto is not None else None)
    if a.upto is not None:
        print(f"after {a.upto} steps -> next turn {end}")
        return 0
    same = isinstance(end, Terminal) and json.loads(json.dumps(end.outcome)) == log["outcome"]   # JSON turns int keys into str
    print(f"replay (engine v{env.engine_version}):", "identical ✓" if same else f"DIVERGED: {end}")
    return 0 if same else 1


def cmd_random(a: argparse.Namespace) -> int:
    n = VARIANTS[a.variant].num_players
    wins, forced, turns = {}, 0, 0
    for seed in range(1, a.seeds + 1):
        env = Werewolf(a.variant, playbook_mode=a.playbook)
        ep = run(env, {s: RandomPlayer(seed * 100 + s) for s in range(1, n + 1)}, seed)
        end = replay(Werewolf(a.variant, playbook_mode=a.playbook), seed, ep.journal)
        assert isinstance(end, Terminal) and end.outcome == ep.outcome, f"seed {seed} diverged"
        wins[ep.outcome["winner"]] = wins.get(ep.outcome["winner"], 0) + 1
    print(f"{a.seeds} games, all replay-consistent; winners: {wins}")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    """The online service (needs Postgres + Redis: `make dev`)."""
    import uvicorn
    uvicorn.run("werewolf_harness.app.api.main:app", host=a.host, port=a.port)
    return 0


def cmd_serve_legacy(a: argparse.Namespace) -> int:
    """The original single-process server: standard library only, games live in memory."""
    from .server import serve
    serve(a.host, a.port)
    return 0


def cmd_providers(a: argparse.Namespace) -> int:
    """Say which providers are reachable, and prove it with a real tool call.

    A documented endpoint is a claim; a returned tool_call is a fact. This is
    what moves a row in china.py from the first to the second, and it is the
    only place that spends money -- one short call per model.
    """
    from .players.providers import CANDIDATES, PROVIDERS, profile
    from .players.providers.base import ProviderError
    from .players.providers.openai_compat import OpenAICompatClient
    from .server.http import read_env

    env = read_env()
    tool = [{"type": "function", "function": {"name": "vote", "description": "Cast your vote.",
             "parameters": {"type": "object", "properties": {
                 "target_id": {"type": "integer", "description": "a living player", "enum": [2, 3, 4, 5]},
                 "reason": {"type": "string", "description": "why"}},
                 "required": ["target_id"]}}}]
    msgs = [{"role": "user", "content": "投 5 号出局。调用工具，不要用文字回答。"}]

    for key, prov in PROVIDERS.items():
        token = env.get(prov.env_key)
        models = [c for c in CANDIDATES if c.provider == key]
        if not token:
            print(f"\n{prov.label:22s} 没有 key（.env 里设 {prov.env_key}）")
            print(f"{'':22s} 注册：{prov.console}")
            if prov.free:
                print(f"{'':22s} 免费档：{prov.free}")
            continue
        print(f"\n{prov.label:22s} {prov.base_url}")
        for c in models:
            if a.model and c.model != a.model:
                continue
            p = profile(c.model)
            cl = OpenAICompatClient(c.model, api_key=token, base_url=prov.base_url,
                                    tool_choice=p.get("tool_choice", "required"),
                                    extra_body=p.get("extra_body"),
                                    temperature=p.get("temperature"))
            t0 = time.time()
            try:
                r = cl.chat(msgs, tools=tool, max_tokens=200, timeout=a.timeout)
                dt = time.time() - t0
                if r.tool_calls:
                    print(f"  ✓ {c.model:32s} {dt:5.1f}s  工具调用 {r.tool_calls[0].name}"
                          f"{r.tool_calls[0].arguments}")
                else:
                    print(f"  ! {c.model:32s} {dt:5.1f}s  通了但没返回工具调用"
                          f"（finish={r.finish_reason}）— 多半是 tool_choice 或厂商参数的问题")
            except ProviderError as exc:
                print(f"  ✗ {c.model:32s} {time.time() - t0:5.1f}s  {str(exc)[:150]}")
    print("\n没有 key 的行只是还没注册，不是坏了。表在 players/providers/china.py。")
    return 0


def cmd_resume(a: argparse.Namespace) -> int:
    """Pick a saved game up where it stopped, and keep it in the running server."""
    import json
    import urllib.request
    log = json.load(open(a.file, encoding="utf-8"))
    req = urllib.request.Request(f"http://{a.host}:{a.port}/api/games/resume",
                                 data=json.dumps({"log": log, "max_tokens": a.max_tokens}).encode(),
                                 method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        g = json.loads(r.read())
    # A game summary carries its own `error` field (usually null), so the presence
    # of the key says nothing -- only a failure with no game in it is a failure.
    if "id" not in g:
        print("接不上：", g.get("error") or g)
        return 1
    print(f"{g['id']} 从 {len(log.get('journal') or [])} 个回合处接着跑 "
          f"（{g['variant']} · {g['playbook']} · seed {g['seed']}）")
    print(f"http://{a.host}:{a.port}/#/g/{g['id']}?seat=0")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="werewolf_harness")
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("play")
    pl.add_argument("--variant", default="12p", choices=sorted(VARIANTS))
    pl.add_argument("--seed", type=int, default=1)
    pl.add_argument("--model", default="mock")
    pl.add_argument("--guard", default="L1,L2,L3", help="comma-separated layers, or empty")
    pl.add_argument("--evidence", action="store_true")
    pl.add_argument("--playbook", default="static", choices=["none", "static", "dynamic"])
    pl.add_argument("--temperature", type=float, default=0.7)
    pl.add_argument("--max-tokens", type=int, default=6_000_000, dest="max_tokens",
                    help="本地估算的每局 token 上限，超了停局")
    pl.add_argument("--max-usd", type=float, default=None, dest="max_usd")
    pl.add_argument("--out", default=None)
    pl.add_argument("--lang", default="zh", choices=["zh", "en"], help="AI 发言用的语言")
    pl.add_argument("--engine-version", type=int, default=None, dest="engine_version",
                    help=f"用哪一版规则（默认当前 v{ENGINE_VERSION}）；旧版规则只为对照实验和回放保留")
    pl.add_argument("--parallel", type=int, default=8,
                    help="最多几个座位同时想（投票、复盘、夜里互不依赖的角色）；1 = 一个一个来")
    pl.add_argument("--record", action="store_true",
                    help="录下每一次模型调用（请求+回复/错误），写在 --out 旁边的 .cassette.jsonl")
    # model cascade (players/llm/cascade.py, docs/adr/0014): --model is then the cheap tier
    pl.add_argument("--wolf-model", default=None, dest="wolf_model", help="狼人座位用这个模型，其余座位用 --model")
    pl.add_argument("--wolf-hint", default=None, dest="wolf_hint", choices=["hantiao"], help="给狼人加一条打法指令")
    pl.add_argument("--no-memory", action="store_true", dest="no_memory", help="消融：不跨回合保留信念表和笔记")
    pl.add_argument("--strong", default=None, help="级联的强模型（--model 是便宜模型），例如 claude-cli:sonnet")
    pl.add_argument("--cascade", default="off", choices=["off", "on-fail", "decisive", "both"],
                    help="on-fail：便宜模型的回复没过校验（格式/未知工具/非法参数/守卫拦截/无据引用/步数用尽/无应答）"
                         "就在同一回合、同一份对话上换强模型重试；decisive：关键回合（--decisive）直接用强模型；both：两者都要")
    pl.add_argument("--decisive", default=None,
                    help="关键回合，逗号分隔的任务名或终结工具名（默认 vote,night,hunter_shoot,badge）")
    pl.add_argument("--low-confidence", action="store_true", dest="low_confidence",
                    help="另外：便宜模型的发言一个座位号都没提（没有可核对的内容），就让强模型重答")
    rl = sub.add_parser("replay-llm", help="用录像重跑一局：真实的 LLMPlayer，模型换成录像，核对 journal")
    rl.add_argument("file")
    rl.add_argument("--cassette", default=None, help="录像路径（默认看日志里的 cassette 字段或日志旁边）")
    rl.add_argument("--strict", action="store_true", help="另外要求录像里每一次调用都被用到")
    rl.add_argument("--allow-drift", action="store_true", dest="allow_drift",
                    help="提示词变了也继续：报告哪里变了，只要录下的回复还适用就接着跑")
    rl.add_argument("--upto", type=int, default=None, help="只回放前 N 步")
    rp = sub.add_parser("replay")
    rp.add_argument("file")
    rp.add_argument("--upto", type=int, default=None)
    rd = sub.add_parser("random")
    rd.add_argument("--variant", default="12p", choices=sorted(VARIANTS))
    rd.add_argument("--seeds", type=int, default=10)
    rd.add_argument("--playbook", default="static", choices=["none", "static", "dynamic"])
    sv = sub.add_parser("serve", help="在线服务（Postgres + Redis），用浏览器坐进去玩")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8402)
    sl = sub.add_parser("serve-legacy", help="旧的单进程服务（标准库，不持久化）")
    sl.add_argument("--host", default="127.0.0.1")
    sl.add_argument("--port", type=int, default=8402)
    rs = sub.add_parser("resume", help="从存下来的日志接着跑一局被中断的游戏")
    rs.add_argument("file")
    rs.add_argument("--host", default="127.0.0.1")
    rs.add_argument("--port", type=int, default=8402)
    rs.add_argument("--max-tokens", type=int, default=None, dest="max_tokens")
    pv = sub.add_parser("providers", help="探活：哪几家有 key、通不通、能不能返回工具调用")
    pv.add_argument("--model", default=None, help="只探这一个模型")
    pv.add_argument("--timeout", type=float, default=60.0)
    a = p.parse_args(argv)
    return {"play": cmd_play, "replay": cmd_replay, "replay-llm": cmd_replay_llm, "random": cmd_random,
            "serve": cmd_serve, "serve-legacy": cmd_serve_legacy, "providers": cmd_providers, "resume": cmd_resume}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
