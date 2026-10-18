# 架构

```
                         ┌──────────────── Caddy (HTTPS, 健康检查, 主备切换) ────────────────┐
  浏览器 / bot ──WS/SSE/REST──▶ API 节点 (FastAPI)                           API 节点 (热备)
                         │  auth: 游客 JWT + seat token      │
                         │  限流: Redis 滑动窗口               │
                         ▼                                    │
               ┌─ 对局运行时 (app/runtime) ────────────────────────────┐
               │  GameActor × N   每局一个协程 + mailbox，单写者        │
               │    提交: ① journal(+模型记忆, fencing) ② 生效 ③ 推送   │
               │  TimingWheel     所有回合的倒计时 (O(1) 挂/撤)          │
               │  Leases          Redis 租约 + Postgres fencing token  │
               │  recovery scan   没人持有的活局 → 重放 journal 接管     │
               └───────────┬───────────────────────────────────────────┘
                           │ AgentJob(game, seat, version, task, memory)
             ┌─────────────┴──────────────┐
      LocalExecutor (线程池)       StreamExecutor ── Redis Streams ──▶ worker 进程 × M
                                   llm:jobs:{human,online,offline}      重放 journal 得到视角
                                   llm:results:{node}  ◀──────────────  跑回合, XACK
                           │
               ┌─ 模型调用层 (app/llm) ─────────────────────────────────┐
               │  GatewayClient: 每局预算 → 熔断器 → token 桶 → 调用     │
               │    → 分类(429/超时/5xx/超长/拒答) → 下一个模型 → 记账   │
               │  规则座位: 所有模型都失败时，按信念表出手，局不卡死        │
               └───────────────────────────────────────────────────────┘
存储: Postgres (games, game_seats, journal, agent_state, llm_calls, users)
      Redis    (租约, 在线状态, 限流, token 桶, 每局预算, 排行榜 ZSET, Streams)
观测: /metrics (Prometheus) · OpenTelemetry trace (turn → invoke_agent → chat, turn → commit) · JSON 日志
```

## 一步是怎么走完的

1. `_prompt()` 算出当前回合的视角和合法动作，打开一个 `turn` span，然后分两种情况：
   - 真人回合：挂一个倒计时，把待办写进 `pending`；
   - 模型回合：投递一个 AgentJob，同时挂模型超时。
2. 结果以消息形式进入 mailbox，来源有三种：真人的提交（HTTP 或 WS）、模型结果、倒计时到期。
3. `_commit()` 按顺序做三件事：
   - 写 journal，与 `agent_state` 在同一个事务，校验 fencing token；
   - `env.step()`，version 加 1；
   - 发布事件，然后回到第 1 步。
4. 读取方（REST / WS / SSE）只通过 `Table.seat_state(seat)` 等读模型拿数据，座位由 seat token 决定。

## 三个不变量

- `world == f(seed, journal)`：重放是恢复、只读加载、worker 构造视角、确定性自检这四件事的共同基础。更准确地说是 `world == f(engine_version, seed, journal)`：每局记录创建时的引擎版本，这四件事都按记录的版本重放（ADR 0010）。
- 一局任意时刻只有一个写者：进程内靠 actor，进程间靠租约加 fencing。
- 模型结果只对它依据的那个 version 生效：迟到或重复的结果一律丢弃。

## 目录

```
app/api/       REST + WebSocket + SSE，鉴权，限流，排行榜
app/auth/      JWT（user / seat）
app/runtime/   Table（状态与读模型）、GameActor、TimingWheel、Leases、Runtime（创建/接管/续约/排空）、规则座位、
               过载信号 load.py + 准入排队 admission.py（ADR 0011）
app/llm/       gateway（回退链、熔断、token 桶、预算、记账）、sim（模拟供应商 + 前缀缓存）、queue + worker
app/storage/   Store（SQL）、迁移
app/obs/       日志、指标、trace
migrations/    Alembic（手写 SQL）
deploy/        Dockerfile、compose、Caddy、Prometheus、Grafana 看板
scripts/       chaos 演示、基准、确定性自检、缓存测量
loadtest/      WebSocket 压测机器人（Python / k6）
```
