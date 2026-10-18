# werewolf_harness

12 人狼人杀（也支持 8 / 15 人局），真人和 LLM 同桌。

- 底层是一个五个方法的环境协议，加上带注入防护的 ReAct 座位。
- 上面是一个可恢复、可多节点运行的在线服务：进程被杀、换机器之后对局接着打，已经做过的模型决策不重新请求。

```
arena/  envs/  players/     原有核心库：协议、规则引擎、LLM 座位、四层防护（标准库，无第三方依赖）
app/                        在线服务：运行时、模型调用层、API、存储、观测（见 docs/architecture.md）
migrations/                 Postgres 表结构（Alembic，手写 SQL）
deploy/                     Docker / compose / Caddy / Prometheus / Grafana / CI
scripts/  loadtest/         chaos 演示、基准、确定性自检、缓存测量、压测
docs/                       架构、技术决策（adr/）、数字（numbers.md）
tests/                      核心 + integration（需要 Postgres / Redis）
```

## 快速开始

```
# 依赖（在上一级目录建 venv；包名是 werewolf_harness）。锁文件带哈希，CI 用的是同一份
uv venv ../.venv && uv pip install -p ../.venv --require-hashes -r requirements-dev.lock
make dev          # Docker 起 Postgres + Redis + Jaeger
make run          # http://127.0.0.1:8402  （首次启动会自动迁移数据库）
make test         # 全部测试；make test-core 不需要数据库
```

模型选 `mock`（离线、秒回）或 `sim` / `sim-cheap` / `sim-flaky`：模拟供应商，带延迟、故障和前缀缓存，用于压测和演练。
真模型需要在 `.env` 里配置 key，见 `.env.example`。

也可以不用 API key，直接用本机的 Claude Code 当模型：座位选 `claude-cli:sonnet` / `claude-cli:haiku` / `claude-cli:opus`（或 `claude-cli`，用 CLI 的默认模型）。每次调用就是一次 `claude -p`，用的是 CLI 当前登录的账号（订阅或 API key），Claude Code 自带的工具全部关掉、系统提示整个替换成本局的提示词、不保存会话，回复按 JSON 工具协议解析。只要 `claude` 在 PATH 上，开局页面的模型列表里就会出现这几项，可以和 API 模型混坐一桌。

```
python -m werewolf_harness.cli play --variant 8p --model claude-cli:haiku
```

注意：CLI 不支持 temperature 和 max_tokens，这两个参数对它无效；每次调用要多花 1–2 秒启动进程；同时最多起 `CLAUDE_CLI_CONCURRENCY`（默认 4）个进程，其余排队。没登录时第一次调用就会停局并提示去 `claude` 里登录；订阅额度用完时按限流处理，切到备用模型。工作目录是一个空的临时目录，项目里的 CLAUDE.md 不会被带进来；但用户级的 `~/.claude` 设置和 CLAUDE.md 仍可能被加载，想完全隔离就设 `CLAUDE_CLI_BARE=1`（只能用 API key 认证）。

## 在线服务做了什么

| 能力 | 做法 | 代码 / 文档 |
|---|---|---|
| 单写者对局 | 每局一个协程 + mailbox；真人提交、模型结果、倒计时都是消息 | `app/runtime/actor.py` |
| 先写库再生效 | journal 与模型记忆同一事务；主键 `(game_id, seq)` 保证一步只写一次；`request_id` 防重复提交 | ADR 0002 |
| 崩溃恢复 | 重放 journal，并从 `agent_state` 恢复模型座位的信念表和笔记；已提交的回合不重新问模型 | `runtime.py`、`scripts/chaos_demo.py` |
| 多节点 | Redis 租约 + Postgres fencing token；被暂停后醒来的旧节点无法写入 | ADR 0004 |
| 引擎版本 | 每局记录创建时的规则版本（`games.engine_version`），恢复和重放按记录的版本；规则改动挂在版本开关上；`compat_check` 列出一次规则变更会影响的老局和分叉步，CI 回放 74 份存档 journal | ADR 0010、`make compat` |
| 回合计时 | 分层时间轮（O(1) 挂 / 撤） | `app/runtime/timers.py` |
| 模型调用层 | 回退链、熔断器、按 token 的令牌桶（Redis + Lua）、每局预算、逐次记账、按任务路由模型；模型可以是各家 API，也可以是本机 `claude -p` | `app/llm/gateway.py`、`players/providers/claude_cli.py` |
| 模型任务队列 | Redis Streams：三条优先级流、至少一次投递、XAUTOCLAIM 认领、死信 | ADR 0005 |
| 局不卡死 | 所有模型都失败时，规则座位按信念表出手 | `app/runtime/rules_npc.py` |
| 实时通信 | WebSocket（心跳、断线按 `last_seq` 续传、单连接令牌桶）+ 观战用 SSE | `app/api/main.py` |
| 流式发言 | 模型发言边生成边显示为草稿（「正在说…」）；草稿要么提交时换成 journal 里的原文，要么明确撤回；超时、停局时取消上游调用 | ADR 0009 |
| 座位隐私 | 游客 JWT + seat token；公共事件去掉夜间行动者和 `thought` | ADR 0007 |
| 排行榜 | Redis ZSET（同分按先到者排前）；模型按阵营统计胜率 | `app/api/leaderboard.py` |
| 可观测 | Prometheus 指标 + Grafana 看板；OpenTelemetry：turn → invoke_agent → chat，跨线程、跨进程 | `app/obs/` |
| 提示缓存 | system message 按变化频率排序；模拟缓存命中率 62.9% → 68.8% | ADR 0006 |
| 故障注入 | 真实进程上 kill -9 worker / 节点、Redis 与 Postgres 重启、网络故障、模型故障窗口、重复投递、旧 owner 醒来；每个场景后自动检查 7 条不变量 | `faults/`，ADR 0013 |

所有数字、口径和样本量都在 `docs/numbers.md`。

## 常用命令

```
make chaos        # 两个节点，kill -9 其中一个，看另一个接管（约 5 s，由租约 TTL 决定）
make faults       # 故障注入：真实多进程 + 故障代理，7 个场景 × 7 条不变量 → faults/report.md（约 3 分钟；faults-quick 是 CI 子集）
make bench        # 接管耗时、空闲资源占用、优先级排队、故障演练、缓存命中
make load         # WebSocket 压测（先 make run）
python -m werewolf_harness.scripts.verify_replay   # 已结束的局逐一重放，检查结局一致
make up / make smoke / make down   # Docker 整套：双节点 + worker + Caddy + 监控；smoke 打一局 mock 并逐项检查
```

模型级联（便宜模型打底，失败或关键回合换强模型；docs/adr/0014）：

```
python -m werewolf_harness.cli play --variant 8p --model claude-cli:haiku --strong claude-cli:sonnet --cascade both --out g.json
python -m werewolf_harness.scripts.paired_cascade --seeds 2001-2010 --cheap claude-cli:haiku --strong claude-cli:sonnet \
    --arms strong,cheap,cascade --record --out runs/cascade/     # 同种子配对：成本、升级原因、合法率、非劣效结论
```

在线服务用 `TableConfig.cascade = {"cheap": ..., "strong": ..., "mode": "on-fail|decisive|both"}`（`POST /api/games` 的 `cascade` 字段）。

容器部署、CI 和验证记录见 `docs/deploy.md`。注意 `claude-cli` 座位只能在宿主机上跑（`cli serve`），容器里只有 mock / sim / API key 模型。

旧的单进程服务还在：`python -m werewolf_harness.cli serve-legacy`（标准库 HTTP，无持久化）。

---

# 核心库

## The contract

```python
class Environment(Protocol):
    def reset(self, seed) -> Turn | Terminal          # deal, build state; first actor
    def observe(self, actor) -> dict                  # the only information exit (G1)
    def action_space(self, actor, task) -> ActionSpec # what may be submitted now; rendered AND checked (G2)
    def step(self, actor, task, action) -> Turn | Terminal   # the only state write; says who is next
    def rules_text(self) -> str                       # rendered from the Variant, pinned by tests (G3)

class Player(Protocol):
    def act(self, view, task, spec, feedback) -> action
```

`run()` in `arena/driver.py` is 55 lines and contains no game word (a test greps
for it). Every seat -- random script, `LLMPlayer`, a browser later -- implements
`act`. The journal `[(actor, task, action)]` rebuilds the world exactly (G4);
`cli replay` proves it on any saved game.

The agents are reproducible too, once their model calls are kept: `play --record`
writes every request and reply (or error) to `g.cassette.jsonl` next to the log, and
`replay-llm g.json` re-runs the real LLMPlayer stack on those replies (0 model calls,
~100 ms a game) and checks journal, outcome, beliefs and notes; a changed prompt is
reported with the seat/turn/call and a diff. `scripts/make_regression.py` turns a
recording into a case under `tests/regressions/` (ADR 0008):

```
CLAUDE_CLI_CONCURRENCY=8 python -m werewolf_harness.cli play --variant 8p --seed 1001 \
    --model claude-cli:haiku --parallel 8 --record --out runs/rec/g1001.json
python -m werewolf_harness.cli replay-llm runs/rec/g1001.json
python -m werewolf_harness.scripts.make_regression runs/rec/g1001.json --name g1001 --upto 40
```

## The game (envs/werewolf.py)

- Seats 1..N, roles dealt from `Variant.role_counts` with `random.Random(seed)`; that and the
  fallback target order are the engine's only randomness.
- Night order: seer → guard → wolves (seat order, majority kill, first-named breaks ties) → witch.
  Pack chat and pack planning (static: once; dynamic: every night) precede the night for 12p/15p.
- Day: round-1 sheriff election (candidates do not vote; one candidate takes the badge; a tie leaves
  it unclaimed), speech in seat order rotated to the sheriff, vote with a 1.5 sheriff ballot, tie
  exiles nobody. Interrupts in order: hunter shot → badge → last words.
- Rule parameters on the `Variant` (`guard_self`, `guard_witch_overlap_kills`, `witch_self_save`,
  `hunter_shoots_on`) are each an engine branch **and** a sentence in `rules_text()`; `tests/test_rules.py`
  plays every sentence out.
- Deliberate choices where the old engine and its own rules text disagreed: the witch cannot save
  herself; a hunter-shot sheriff still passes the badge; a wolf who skips does not cancel the pack's
  knife; the guard stops the knife, not the poison; a self-detonation is public as "detonated".
- Terminal actions are `ToolSpec`s whose legal arguments (`choices`) are computed from state. The
  environment never sees an illegal action: the driver checks the same object it rendered.

## The LLM seat (players/llm)

`LLMPlayer.act` builds the system message (rules from the view, private identity block, L1 preamble,
output contract naming exactly the tools this turn offers) and the situation (public record, this
round's speeches -- others' fenced and filtered by the guard -- own beliefs, the tagged task line),
then runs up to 8 ReAct steps. Information tools (`query_history`, `query_votes`, `query_deaths`,
`read_belief`, `update_belief`, `check_ability`, and for wolves the playbook tools) read the view and
the seat's `BeliefState`; a terminal call is checked against the `ActionSpec` and the guard's
output-side checks (say/do consistency, directive compliance, evidence, forensics) and returned.
Malformed or unknown calls are bounded and retried with the reason; a turn that cannot be salvaged
returns the environment's fallback and records `fallback_used`. `TransportBudget` abandons a game
whose model stops answering; a balance/key refusal raises `GatewayExhausted` at once.

Prompt neutrality is a contract, not a style: see `PROMPTS.md`, enforced by `tests/test_prompts.py`.

