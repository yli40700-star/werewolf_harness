# ADR 0009：流式发言——草稿（draft）与提交（commit）的一致性

- 状态：已采纳
- 代码：`players/providers/base.py`（`chat_stream`、`CancelToken`、`Delta`）、`players/providers/{openai_compat,claude_cli,mock}.py`、`app/llm/{sim,gateway,queue}.py`、`players/llm/streaming.py`（增量提取器）、`players/llm/loop.py`、`players/recovery.py`、`app/runtime/{agents,actor,table,drafts}.py`、`app/api/main.py`、`web/app.js`
- 测量：`scripts/bench_streaming.py`；测试：`tests/test_streaming.py`、`tests/test_streaming_providers.py`、`tests/integration/test_streaming_runtime.py`

## 背景

模型发言要等整段生成完才出现，玩家干等 5–15 秒。改成流式以后，客户端会看到 **还没写进 journal 的字**。这时如果节点崩溃、回合超时、对局被停、守卫拒掉这次发言重问、或者提前问的结果作废，屏幕上显示过的内容和最终提交的内容就可能对不上：观众看到 5 号说了「我是预言家」，journal 里却是另一段话，或者根本是超时兜底。

要守住的不变式：

> 客户端看到过的每一段草稿，最后要么 **一字不差地** 变成该回合 journal 里的发言，要么被 **明确撤回**。

## 决定

### 1. 提供方契约：`chat_stream`

`LLMClient.chat_stream(...)` 先产出若干 `Delta`，最后产出一次 `LLMResponse`——和 `chat()` 返回的是同一个对象。默认实现调用 `chat()`，把整段当作一个片段吐出来，所以没实现流式的客户端行为不变。

- `MockClient`：可配首 token 时间、tokens/s、分块大小、发言长度（`STREAM_PROFILES`：`mock-stream` = 1 s / 45 tok/s / 400 token）；决策本身与这些参数无关，`chat()` 和 `chat_stream()` 给出同一个动作。
- `OpenAICompatClient`：`stream: true` 的 SSE；`iter_sse` 做标准分帧，并容忍漏掉空行的中转站；`StreamAssembler` 按 `index` 拼回 tool call 参数，`reasoning_content` 不外传。某个中转站流式 tool call 有问题时，可以在 profile 里设 `"stream": false`。
- `ClaudeCLIClient`：`claude -p --output-format stream-json --include-partial-messages --verbose`，逐行解析，认不出的行跳过；最后的 `result` 行按原来的 JSON 输出处理，所以用量、错误码、状态完全一样。没有 partial 的老版本 CLI 也能工作（最后一次性吐出全文）。
- `SimClient` 与网关：网关按回退链逐个模型流式调用；某个模型流到一半失败时，先发一个 `reset` 片段再试下一个模型——已经显示的内容来自一个不会被采用的回复。

### 2. 只流发言文本：增量 JSON 提取器

模型流出来的不是发言，而是 tool call 的参数 JSON：`{"thought": "...", "content": "我觉得 5 号..."}`，而且 `thought`（座位的私下推理）往往先到。`players/llm/streaming.py` 的 `ContentExtractor` 跟踪当前所在字符串的 key 路径，只输出目标字符串解码后的字符：转义（`\n`、`\"`、`\uXXXX`、代理对）被分块切断时也能正确解码；目标字符串之外的任何字符都不会输出。只对说出口的终止动作（`speak`、`last_words`、`campaign_run`、`pack_say`、`review_say`）启用。

`SpeechStream` 按名字跟踪原生 tool call（名字到达前的片段先扣着，不是发言工具的片段直接丢弃）；JSON 协议下一旦读到 `"action"` 不是发言，就作废已显示的部分。两个通道（参数和正文）哪个先产出发言就跟哪个，同一段话不会显示两次。

### 3. 草稿协议

一次模型调用里的一次发言尝试 = 一个草稿，id 为 `d{epoch}.{seq}.{seat}.{run}.{attempt}`：

| 事件 | 字段 | 何时发出 |
|---|---|---|
| `draft` | `draft_id, seat, seq, task, offset, delta` | 新增文字；按偏移追加，重复或重放的消息无害 |
| `draft_commit` | `draft_id, n, text` | 该回合的动作 **已写入 journal**，`text` 就是 journal 里的原文；在该回合的 `turn` 事件之前发出 |
| `draft_retract` | `draft_id, reason` | 守卫拒绝重问（`guard:*`、`retry`、`spec`）、换模型（`reset`）、超时（`timeout`）、提前问的结果作废（`stale`、`turn_gone`）、停局（`shutdown`、`aborted`）、同一任务被别的 worker 重跑（`rerun`）、最终动作不是这段话（`not_said`） |
| `draft_reset` | `epoch, reason` | 新的一任 actor（崩溃后被接管）开始；上一任显示过的所有草稿作废 |

机制：

- **线程侧**：`LLMPlayer` 通过 `on_draft(kind, attempt, text)` 报告「新文字」和「这次尝试作废」；每次尝试最终要么成为 `last_turn.draft_attempt`，要么收到一次 abandon。`DraftPump` 给片段编偏移、合并成约 20 次/秒的消息。
- **actor 侧**：草稿属于发出它的那次 ask。`_commit` 在第③步（publish）里用 journal 的原文提交它；ask 的其他所有结束方式都经过 `_drop_ask`，撤回它的所有草稿并取消还在跑的调用。actor 退出时 `_close_all` 兜底。
- **不入库**：草稿是实时缓冲（`Table.drafts` + `draft_log`），有自己的序号 `dseq`，不占用 journal 派生事件的 `seq`，所以重放和断线续传的语义不变。每个草稿事件记录发出时已有多少事件（`at`），`drafts.outbox` 按它把两条序列合并，保证 commit 先于对应的 turn。
- **断线续传**：客户端带 `(epoch, dseq)` 重连，同一任期且日志还在就原样补发；否则发 `draft_sync`：作废别的任期的草稿，按客户端声明的偏移补齐仍在进行的草稿，并告知它提到的已结束草稿的结局。
- **崩溃恢复**：新任期 `draft_epoch = fencing token`；接管时先发 `draft_reset("recovered")`，再重问未完成的回合。
- **隐私**：草稿的可见范围和它所属的回合一致（ADR 0007）：发言、遗言、竞选、复盘是公开的；`pack_chat` 只给狼人看（以及 `may_see_all` 的座位）。WebSocket 按 seat 过滤，SSE（`?drafts=1`）只有公开草稿，轮询快照 `drafts_view` 同样过滤。

### 4. 取消

`CancelToken` 是线程安全的停止标志，还可以注册「立刻拔插头」的回调：杀掉 `claude -p` 进程、`shutdown()` HTTP 连接的 socket（`close()` 叫不醒正阻塞在 `recv()` 里的线程）。每次模型调用用子 token，所以单次调用的硬超时只杀这一次调用，不会结束整个回合。`Cancelled` 不是 `ProviderError`：不重试、不回退、不记熔断；网关按已生成的部分记账。

Redis worker：节点在 `llm:cancel` 频道 PUBLISH `{job_id, ...}`，正在跑这个任务的 worker 取消它；还没开始的任务记住 10 分钟，领到时直接丢弃。用 pub/sub 而不是 stream：丢一条取消只多花 token，不影响正确性。

**省下的 token 怎么估**（`streaming.savings`）：被取消的调用要为 prompt 和取消前已生成的部分付费，省下的是剩余的回复，加上这个回合本来还要做的调用。这两者都观察不到，只能用同类已完成调用的花费来估：

- `cut = max(0, E[completion | 模型, 类型] − 已生成)`，类型分 speech / step / note（发言约是投票的 10 倍）；
- `skipped = prompt + E[completion | 模型, note]`：每个回合结尾的「给自己记笔记」调用，它的 prompt 和被取消那次差不多大；
- `E[...]` 是本进程已完成调用的滑动平均（最多 200 个样本），冷启动用保守默认值。回合里后面可能还有的 ReAct 步骤没算，所以这个数偏低。

### 4b. 和录制回放（B）、公平队列（H）、批量接管（J）的结合

- **录制**：流式只是交付方式，不是内容。`RecordingClient.chat_stream` 边转发片段边等最终回复，写进录像的那一行和 `chat()` 写的完全一样（同一个 key、同一个请求哈希、同一个回复/错误）；硬超时同样提前 1 s 生效（拔掉这次调用的子 token，记成同一条 timeout 错误）。外部取消（回合没了）记为 `Cancelled` 错误，回放时还原成真正的 `Cancelled`。`ReplayClient` 用基类的 `chat_stream`（把录下的回复当片段吐出），所以流式录的局可以不流式回放，反之亦然。录像 key 里的 kind（step/forced/note）和流式无关；估算省下多少 token 用的「speech/step/note」分类另起名字 `cost_kind`。`TurnResult.draft_attempt` 只在流式时出现，`replay-llm` 比较时忽略它（它说明的是有没有人在看，不是座位的决定）。
- **取消 × 公平队列**：`LocalExecutor.cancel` 对还在排队的任务直接从该局的队列里拿掉（不占线程、不花 token、不投递结果——actor 已经结束了这次 ask）；已经在跑的任务拔插头。`StreamExecutor.cancel` 对还压在节点上（每局上限）的任务直接丢掉；已经进流的照旧发 `llm:cancel`，没领到的 worker 记住它，领到时回一个 stale 回复，把该局的名额还回来。
- **恢复**：`draft_reset("recovered")` 在 `_resume` 里发，单个接管和 J 的批量接管（`adopt_many`）都经过它；表按库里记录的引擎版本重建，未完成的回合在那个版本下重问。

### 5. 前端

草稿就地显示在发言流里，标「正在说…」，字色浅一些，末尾有闪烁光标；`draft_commit` 后换成正式文本（随后被拉到的正式发言替换），`draft_retract` 后显示一行小字「撤回了还没说完的话（原因）」，5 秒后消失。

## 数字

`python -m werewolf_harness.scripts.bench_streaming`（需要 Postgres + Redis；模拟模型，见脚本说明）。

**延迟**：10 桌 8 人局，模拟座位首 token 1 s、每座 30–60 tok/s（随机）、发言约 400 token（中位 449 字），共 137 段已提交发言；时间都从「轮到这个座位」算起（包含发言前查信息的 ReAct 步骤，每步也要 1 s 首 token）：

| 指标 | p50 | p95 |
|---|---|---|
| 第一个字可见（流式） | 2.17 s | 4.41 s |
| 最后一个字流完 | 11.41 s | 16.50 s |
| 整段出现（不流式，= 提交时刻） | 13.35 s | 17.83 s |
| 提前量（提交时刻 − 第一个字） | 11.62 s | 14.87 s |

**真实模型**（2026-10-15）：`claude-cli:sonnet`（sonnet 5.5，本机 `claude -p`，CLI 并发 8），4 桌同时打，30 段已提交发言，发言中位 100 字：

| 指标 | p50 | p95 |
|---|---|---|
| 第一个字可见（流式） | 2.32 s | 4.84 s |
| 最后一个字流完 | 3.80 s | 6.52 s |
| 整段出现（不流式，= 提交时刻） | 8.37 s | 13.40 s |
| 提前量（提交时刻 − 第一个字） | 5.89 s | 8.83 s |

真实发言比模拟的短很多（中位 100 字，模拟约 400 token），所以“最后一个字流完”比模拟早得多。“流完”和“提交”的 p50 相差约 4.6 s。这段时间里有发言之后的守卫检查、CLI 进程收尾和提交本身，这次没有拆开测。真实多进程下的草稿不变式见 ADR 0013：worker 和节点在发言流到一半时被 kill -9，客户端看到的草稿全部闭合。

**不变式**：50 局（40 局带故障注入：8% 的调用流到一半断线 → 传输重试，5% 的调用卡住直到 1 s 回合期限 → 超时撤回并取消；另外 10 局在草稿显示到一半时直接杀掉节点，由另一节点接管），每局一个观众、一个狼人两个读者，通过和 WebSocket 相同的合并逻辑（`drafts.outbox` + `DraftMirror`）读流。**通过率 50/50 = 100%**：读者共看到 2402 段草稿，2004 段提交（文本与 journal 完全一致），398 段撤回（`reset` 228、`timeout` 148、崩溃后 `sync` 22），0 段悬空。

**取消**：581 次被取消的回合，估计共省 1,522,744 token（每次约 2,621）；其中截断的剩余回复只占 15,237，绝大部分来自省掉的回合末笔记调用（它的 prompt 和被取消的调用一样大）。这是模拟模型下的数字，真实模型的笔记 prompt 和发言长度不同，比例会变。

## 取舍

- 草稿不入库：崩溃时正在显示的半句话会丢，换来的是 journal 和重放完全不受影响。新任期用 `draft_reset` 明确撤回，而不是假装它没发生过。
- 提交前不做「草稿 == 最终文本」的检查：最终文本以 journal 为准，`draft_commit` 直接带全文覆盖。就算增量提取和最终解析有细微差别（例如模型写了非法转义），客户端最后显示的也一定是 journal 里的原文。
- 取消是尽力而为：取消消息丢了、或者客户端没法中断，结果按过期丢弃，只是多花了 token。
- 合并成约 20 次/秒的消息：比逐 token 发送少一个数量级的消息量，代价是最多 50 ms 的额外延迟。
