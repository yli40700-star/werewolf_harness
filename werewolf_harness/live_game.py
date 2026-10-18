"""Play one real-model 12p game, logging every turn as it lands."""
import json, os, sys, time
sys.path.insert(0, "/Users/li/Projects/gto-poker-calculator")
from werewolf_harness.arena import Terminal, Turn, Record, Episode
from werewolf_harness.envs import Werewolf
from werewolf_harness.players import GatewayExhausted, GuardStack, LLMPlayer, OpenAICompatClient, RecoveryPolicy, TransportBudget

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MODEL = os.environ.get("MODEL", "gpt-5.4")
GROUP = os.environ.get("GROUP", "OPENAI")
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/private/tmp/claude-501/-Users-li/42e7a7d6-de0b-4ad5-814b-5196b4450c5d/scratchpad/live{SEED}.json"
LOG = OUT.replace(".json", ".log")

env = {}
for line in open("/Users/li/Projects/gto-poker-calculator/.env"):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1); env[k] = v

log = open(LOG, "a")
def say(msg):
    log.write(msg + "\n"); log.flush()

game = Werewolf("12p", playbook_mode="static")
guard = GuardStack(("L1", "L2", "L3"))
budget = TransportBudget(limit=200)   # outages are waited out above; the budget only stops a dead gateway
policy = RecoveryPolicy(timeout_s=150.0, max_transport_retries=4, backoff_s=1.0)
players = {s: LLMPlayer(OpenAICompatClient(MODEL, api_key=env[f"LLM_KEY_{GROUP}_GROUP"], base_url=env["LLM_BASE_URL"]),
                        guard=guard, policy=policy, transport_budget=budget, playbook=True) for s in range(1, 13)}

t0 = time.time()
journal = []
turn = game.reset(SEED)
truth = game.ground_truth()
CKPT = OUT.replace(".json", ".journal.json")
if os.path.exists(CKPT):
    prior = [Record(**r) for r in json.load(open(CKPT))]
    from werewolf_harness.arena import replay
    turn = replay(game, SEED, prior)
    journal = prior
    say(f"== resumed from {CKPT}: {len(prior)} steps replayed -> {turn}")
say(f"== seed={SEED} model={MODEL} roles={truth['roles']}")
crashed = None
try:
    while isinstance(turn, Turn):
        view = game.observe(turn.actor)
        spec = game.action_space(turn.actor, turn.task)
        p = players[turn.actor]
        feedback, forced = None, False
        for _ in range(3):
            # a gateway outage is a delay, not a decision: wait and ask again before accepting a fallback
            for wait_no in range(4):
                action = p.act(view, turn.task, spec, feedback)
                if p.last_turn.fallback_used != "transport_failure":
                    break
                say(f"[{time.time()-t0:6.0f}s]    gateway not answering for #{turn.actor} {turn.task}; waiting 90s (attempt {wait_no+1}/8)")
                time.sleep(90)
            reason = spec.check(action)
            if reason is None:
                break
            feedback = reason
        else:
            action, forced = spec.fallback, True
        tr = p.last_turn
        journal.append(Record(turn.actor, turn.task, action, forced))
        json.dump([r.__dict__ for r in journal], open(CKPT, "w"), ensure_ascii=False)
        a = action["args"]
        brief = a.get("content") or a.get("reason") or {k: v for k, v in a.items()}
        say(f"[{time.time()-t0:6.0f}s] r{view['round']} {turn.task:13s} #{turn.actor:2d} ({view['role']:8s}) -> {action['name']}: {str(brief)[:160]}"
            f"   | steps={len(tr.steps)} tools={[s.action for s in tr.steps[:-1]]} blocks={len(tr.guard_blocks)} fb={tr.fallback_used or '-'} {tr.latency_ms/1000:.0f}s")
        turn = game.step(turn.actor, turn.task, action)
    outcome = turn.outcome
except GatewayExhausted as exc:
    crashed = str(exc); outcome = None
    say(f"!! gateway exhausted: {exc}")
except Exception as exc:
    import traceback; crashed = traceback.format_exc(); outcome = None
    say("!! crash\n" + crashed)

say(f"== done in {time.time()-t0:.0f}s: {outcome}")
turns = sum(len(p.history) for p in players.values())
fb = sum(1 for p in players.values() for t in p.history if t.fallback_used)
say(f"== health: {fb}/{turns} turns fell back; prompt_tokens={sum(t.prompt_tokens for p in players.values() for t in p.history)} completion={sum(t.completion_tokens for p in players.values() for t in p.history)}")
json.dump({"variant": "12p", "seed": SEED, "model": MODEL, "guard": guard.label(), "playbook": "static", "outcome": outcome,
           "ground_truth": truth, "crashed": crashed, "duration_s": time.time() - t0,
           "journal": [r.__dict__ for r in journal],
           "turns": [t.to_dict() for p in players.values() for t in p.history],
           "night": game.night_records(), "pack_chat": game.pack_chat_log()},
          open(OUT, "w"), ensure_ascii=False, indent=1, default=str)
say(f"== wrote {OUT}")
