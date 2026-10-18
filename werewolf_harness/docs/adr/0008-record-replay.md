# ADR 0008 — Record and replay model I/O

## Context

`world = f(seed, journal)` makes the *engine* reproducible: `cli replay` rebuilds any
game from its journal. The *agents* are not reproducible that way. A seat's decision
depends on the model's replies, and asking the model again gives a different game. Two
problems follow:

* An agent bug seen in a real game (a malformed reply the parser mishandled, a guard
  that blocked the wrong thing, a note that leaked) cannot be re-run locally.
* Every later harness change (streaming, fault injection, cascades, engine checks)
  would need fresh paid model runs to check that it changes nothing it should not.

What makes this harder than "cache by prompt": seats decide in parallel (`parallel=8`),
notes are written on background threads, and answers asked too early get thrown away and
asked again (`discard_last`). Calls from different seats interleave in a different order
on every run.

## Decision

**A wrapper client records every call; a replay client serves them back by a key that
does not depend on timing.**

* `players/providers/recorded.py`: `RecordingClient(inner, cassette, seat=...)` wraps
  any `LLMClient`. For each `chat()` it stores the request (messages, tool schemas,
  model, temperature, max_tokens, tool_mode) and the response (text, tool calls with
  their argument order, token counts, latency, finish reason, raw `usage`), or the error
  (type, message, status, hint).
* **Key** = `seat / turn / task / call / kind`: the seat; the seat's act() number,
  discarded asks included (`LLMPlayer.turns_started`); the task; the call's number
  within that act, retries included; and `step | forced | note`. The LLMPlayer sets
  seat/turn/task/kind in a `contextvars` scope around each call (`_call`), and
  `recovery._with_deadline` already copies contextvars into the model-call pool. The
  key works because **within one seat calls are strictly serial**: the driver keeps at
  most one open ask per seat, and the seat's next `act()` (or `discard_last()`) waits
  for the previous background note first. So a per-(seat, turn) counter is
  deterministic, whatever the cross-seat interleaving. Which asks are made, dropped and
  re-asked depends only on the world state, and that is a function of the replies.
* **Request hash**: sha256 of the request JSON, with key order kept. Wire JSON keeps key
  order, and so must a reply's tool arguments. The first version sorted keys, and the
  replay check caught it: a wolf plan `{"3":…, "10":…}` came back reordered and changed
  a later review prompt.
* **Replay**: `ReplayClient` looks the key up, hashes the incoming request and compares.
  Strict mode raises `ReplayDivergence`, naming the seat, turn, task, call and kind and
  showing a unified diff of the first message that differs (or which tool, or which
  parameter). `--allow-drift` logs the difference and serves the recorded reply anyway.
  A call that was never recorded is always an error. Recorded errors are raised again as
  the same type, message, status and hint, so retries, forced terminals, transport
  fallbacks and `TransportBudget` run exactly as they did. `recovery.call_model`
  re-raises `ReplayDivergence` and does not turn it into a fallback.
* **Timeouts**: the recorder enforces the hard deadline itself, one second before
  `_with_deadline` would. Otherwise an overrunning call would be recorded with the reply
  that arrived late, while the game had already moved on with a timeout.
* **Format: a separate JSONL file per game** (`g.json` → `g.cassette.jsonl`), not
  embedded in the game log. Reasons:
  1. It is written as the game runs, one flushed line per call. The game log is written
     only at the end, and not at all when the game dies with `GatewayExhausted`. Those
     crashed games are the ones we most want to reproduce.
  2. Size: requests repeat themselves. Messages and tool lists are content-addressed
     blobs (`{"b": hash, "v": …}`, written once) and calls refer to them. Measured on an
     8p game with 134 calls: 1.13 MB of naive request JSON becomes 373 KB of cassette,
     and 27 KB gzipped. Stored regression cases are gzipped. Live recordings stay plain
     so that every line is on disk at once.
  3. The viewer and analysis tools that read game logs stay unchanged.
* `cli play --record --out g.json` writes the cassette and saves `settings` (guard,
  evidence, playbook, temperature, lang, budgets, parallel) in the log. `cli replay-llm
  g.json [--strict] [--allow-drift] [--upto N]` rebuilds the seats with the same builder
  `play` uses (`replay_llm.build_players`), runs the real driver at the recorded
  parallelism with retry sleeps disabled, and compares journal, outcome and every
  per-turn record (beliefs, notes, recovery, traces). `--strict` also requires every
  recorded call to be used.
* **Regression cases**: `python -m werewolf_harness.scripts.make_regression g.json --name
  x [--upto N] [--expect …]` replays the game first, then writes
  `tests/regressions/x/{game.json.gz, cassette.jsonl.gz, case.json}`. With `--upto N`
  the environment is cut after N commits and only the calls used are kept.
  `tests/test_regressions.py` replays every case and checks the journal, the turn
  records and the expectations (`not_forced:SEQ`, `forced:SEQ`, `action:SEQ:NAME`,
  `winner:SIDE`, `max_forced:K`, `no_fallback`). Three cases come from mock games:
  a parallel 8p game, a 12p game cut at 40 steps, and a sequential game with 25–30%
  503s and 35% junk replies.
* **Online service**: `TableConfig.record=True`, or `LLM_RECORD=1` for every game,
  wraps the seat's client (the `GatewayClient`) in `Table.make_player`. It appends to
  `$LLM_RECORD_DIR/<game_id>.cassette.jsonl`, defaulting to `runs/cassettes`. The turn
  in the key is `<journal version>.<act number>`, because the service builds a fresh
  player per job. Writers append with one write per call, opening the file each time, so
  the actor's node, Streams workers and an adopting node can all add to one file without
  tearing lines.

## Consequences

* A real-model game becomes a free, deterministic test. Measured on three 8p games with
  a mock that sleeps 300 ms per call at `parallel=8`: recordings took 6.1 s, 18.8 s and
  8.8 s; replays took 78 ms, 107 ms and 91 ms, with 0 model calls. A full 12p chaos game
  replays in about 250 ms.
* Any change that alters a prompt breaks every regression case, with a precise
  message. That is intended, and it is the prompt-change detector. Either re-record and
  regenerate the case, or set `allow_drift` in `case.json` if the recorded replies still
  decide the same game. Changes that add a model call (a new guard retry, say) cannot be
  replayed past that point. Replay tests the harness's handling of fixed replies, not
  the model.
* Making the replay exact surfaced a race that already existed: the background note
  shared the turn's `RecoveryStats`, and whether a note-call retry appeared in the log
  depended on which thread won. `_write_note` now writes `result.recovery` again when
  the note lands.
* Remaining nondeterminism: when several seats run in parallel, the order in which they
  charge the shared `TokenBudget` and `TransportBudget` depends on timing. A game that
  ends exactly at a budget limit could stop at a different seat on replay. No test game
  comes near those limits.
* Online games are recorded and exportable (the file is the export), but `replay-llm`
  replays only CLI games. An online turn is a fresh player with memory restored, may
  end in `rules_npc`, and may be asked again after recovery. A second ask for the same
  (seat, version) in the same process gets new call numbers; in another process it
  repeats a key, which the loader counts in `Cassette.duplicates` and keeps the first.
  An online replay would need `run_agent_turn` driven per journal version. That is the
  next step if it is wanted.
