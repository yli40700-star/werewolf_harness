# ADR 0001：对局运行时自研，不用 Temporal / LangGraph checkpoint

- 状态：已采纳（2026-09-30）
- 相关代码：`app/runtime/actor.py`、`app/runtime/runtime.py`、`app/runtime/lease.py`

## 背景

一局 12 人局约 90–150 步，其中大部分是模型回合，每回合几秒到几十秒；桌上可能坐着真人。要求如下：

1. 进程崩溃或换机器后能接着打；
2. 已经做过的模型决策不重新请求（不花第二遍钱，结果也不变）；
3. 同一局在任意时刻只有一个写者；
4. 给每个座位推送裁剪后的视角。

## 选项

| 方案 | 优点 | 代价 |
|---|---|---|
| Temporal（一局一个 workflow，模型调用做 activity） | 重放、重试、超时、可视化都是现成的 | 多一套集群要运维；workflow 代码必须确定性，不能直接用现有引擎的同步写法；推送和座位视角仍要自己做 |
| LangGraph checkpoint | 和 Agent 编排放在一起 | 解决的是「单个 Agent 图的断点」，不是「多人对局的单写者和推送」；checkpoint 存的是图状态，要么把引擎状态塞进去，要么自己再管一份 |
| **自研：Actor + 以 journal 为准 + 租约/fencing** | 已有 `world = f(seed, journal)`，恢复只需要重放；依赖只有 Postgres 和 Redis；推送和视角裁剪写在同一处 | 幂等、租约、恢复、超时都要自己保证，并且要自己写测试证明 |

## 决定

采用自研方案。关键点：

- **单写者**：每局一个 asyncio 任务加一个 mailbox，状态只由这个任务修改（`GameActor`）。
- **先写 journal 再生效**：见 ADR 0002。
- **恢复**：读取 journal 重放，并从 `agent_state` 表恢复每个模型座位的信念表和笔记。已提交的回合不会再发给模型（有测试覆盖，chaos 演示里数值为 0）。
- **多节点**：Redis 租约决定谁来推进这一局，Postgres fencing token 决定谁能写（ADR 0004）。

## 什么时候换

满足以下任一条件时，换成 Temporal：流程开始跨多个服务；需要人工审批节点并持续数天；步骤数上万、需要 continue-as-new。届时 `Environment` / `Player` 协议不用改，把 `GameActor` 换成 workflow 即可。
