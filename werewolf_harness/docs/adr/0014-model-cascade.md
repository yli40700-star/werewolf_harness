# ADR 0014：模型级联——便宜模型打底，失败与关键回合才用强模型，配对实验验收

- 状态：已采纳（机制与统计已用 mock 验证，见 [docs/cascade_mock.md](../cascade_mock.md)；真实模型的数字待跑）

## 背景

每个回合都打给同一个模型，成本只有一个旋钮：「全桌换便宜模型」。而便宜模型的失败要付两次钱——浪费掉的那次调用，加上往往以同样方式再失败的重试。基线 A（haiku，30 局）里 16.4% 的 ReAct 步被拦，2.6% 的回合走到强制终结；这些正是强模型最值钱的地方。另一方面，一局里大部分回合（发言、复盘、狼队频道）出错的代价很小，投票、夜里刀/验/药、猎人开枪、警徽这几类回合一步走错就改变胜负。

已有的 `TableConfig.routing`（`table.py`，按任务分强/弱）只做了后一半，而且一个回合内不会换模型：便宜模型在投票以外的回合里失败，只能自己重试。

## 决定

**一个小而显式的策略对象**：`players/llm/cascade.py` 的 `CascadePolicy`（可序列化，进日志 settings、录像头、`TableConfig.cascade`）。三类升级理由，全部有名字：

| 理由 | 何时 | 强模型看到什么 |
|---|---|---|
| (a) 失败（`on_fail`）：`schema` / `unknown_tool` / `illegal_args` / `guard:<check>`（evidence、say_do_consistency、directive_compliance、forensics）/ `unsupported_citation` / `forced`（步数用尽）/ `transport`（便宜模型不应答） | 便宜模型这一步没过闸门 | **同一份对话**：便宜模型的坏回复 + 守卫给的拒绝理由原样留在 transcript 里，强模型接着这份对话答 |
| (b) 关键回合（`decisive`，默认 `vote, night, hunter_shoot, badge`，可写任务名或终结工具名如 `night_check`） | 回合开始前 | 回合从第一步起就在强模型上 |
| (c) 低置信度（`low_confidence`，可选） | 便宜模型的发言一个座位号都没提 | **该回复之前**的对话（不给它看那句空话），重问一次 |

- 升级是**黏性的**：一个回合升上去就不再降回来，交回刚失败的模型没有意义。
- 写给下一回合的笔记永远交给便宜模型：那是记账，强模型的决定已经在 transcript 里。
- `unsupported_citation` 只在级联开着时变成拦截（否则仍只计数）：有强模型可换，才值得拒绝便宜模型的无据引用。
- 两档必须说同一种工具协议（`tool_mode`），因为同一份 transcript 要能被任一档接着写；不一致在构造 LLMPlayer 时就报错。claude-cli 的 haiku/sonnet 都是 json_prompt，满足。
- 模式：`off`（默认，行为与日志字节不变）/ `on-fail` / `decisive` / `both`。旧的 `routing` 配置照常可读；它等价于没有回合内重试的 `decisive`。

**机制在 loop 里，只加不改**：`LLMPlayer(client=便宜, strong_client=强, cascade=策略)`。每回合 `start()` 定起始档；每个失败点调一次 `_escalate(reason)`（无级联、已在强档、或策略不认这个理由时是空操作），之后 `_call` 自动打给当前档。

**记录**：
- `TurnResult.cascade = {start, tier, escalations: [{step, reason, from, to}], calls: [{kind, tier, model, prompt, completion, ms, ok}]}`；`TurnResult.model` = 最终作答的模型；`ReActStep.model / escalated`。三者在无级联时都不出现在 `to_dict()` 里——没有级联的日志与以前逐字节相同，`replay-llm` 的回合比较不受影响。
- `cli play` 日志顶层加 `strong`、`cascade`（模式）、`cascade_summary`（各档回合数、调用数、按理由的升级次数、按模型的 token）。
- 在线服务：`AgentResult.meta["cascade"]`，actor 提交时计入 `Table.cascade_stats`，`summary()["cascade"]` 只给全局计数（哪个座位哪一回合升级不公开——夜里升级与否会泄露谁在夜里行动）；步骤 trace 带 `model / escalated`，随 trace 只给有权看的人。

**录像（0008）保持确定**：两档各包一个 `RecordingClient`，共享该座位的每回合调用计数——升级后的那次调用就是这一回合的下一次调用，key 格式 `seat/turn/task/n/kind` 不变；强档在录像头里登记为客户端 `"<seat>:strong"`，`ReplayClient(tier="strong")` 由此读回模型名。模型名在请求哈希里，所以回放时把强档的调用交给弱档（或反过来）是 `ReplayDivergence`，不会悄悄替换。是否升级只取决于（录下的）回复和闸门，回放时同样的回复产生同样的升级。测试覆盖了：同局再打一次，录像 key 与哈希完全相同；`replay-llm --strict` 一致。

**配对实验**：`scripts/paired_cascade.py`。种子决定发牌，所以每个实验臂（strong / cheap / cascade[-mode]）打同一批牌桌，以种子为配对单位：每个指标按局算，求「实验臂 − strong」的逐种子差，再按种子成对 bootstrap（B=2000，少于 5 对不给区间）。配对消掉了局与局之间最大的方差——发牌。指标全部来自 `eval_baseline` 的函数（按局调用以便配对，按臂汇总时用它的按局簇 bootstrap）：

- 成本：CLI 报告的 token（输入 + cache 读写 + 输出）按模型拆开（级联局用每次调用的记录，没落进任何回合的零头记到便宜模型），乘每个模型的标价（USD/百万 token，`--price` 可改）；同时给「折合便宜模型 token」。默认价格：Haiku 4.5 $1/$5、Sonnet 4.5 $3/$15（anthropic.com/pricing，2026-10 所知；用前核对）。cache 读写按普通输入计价，绝对值偏高；但 sonnet:haiku 在每类 token 上都是 3:1，**两臂之比不受 cache 比例影响**，这才是实验要的。
- 升级率（按理由）、合法动作率（回合在 ReAct 步数内由模型自己给出合法终结动作：没有强制终结、没有兜底）、一次过率、兜底率、守卫拦截率、言行一致、无据引用、好人胜率、时长与单次调用延迟。
- 非劣效表述：「级联的合法动作率不比强模型差超过 δ」，当且仅当配对差的 95% 区间下界 > −δ（默认 δ = 2 个百分点，`--delta`）。
- 开跑前打印每臂局数与耗时/token 估计；日志写成普通的 `cli play` 日志（`<out>/<arm>/g<seed>.json`，`--record` 带录像），已存在且能解析的跳过（断点续跑），按种子逐个跑完所有臂（中途停下也全是成对的）；每局内 `--parallel 8`。

## 后果

- 成本有了旋钮，而且是按理由可调的：`--cascade on-fail` 只为失败付强模型的钱，`decisive` 只为关键回合付，`both` 两者都要；`CascadePolicy.from_mode(fail_reasons=...)` 还能只对某几类失败升级。
- mock 验证（20 种子 × 4 臂，便宜档每次调用 30% 概率犯错）：级联（both）成本是强模型的 0.73 [0.69, 0.76]，on-fail 0.56 [0.48, 0.63]，全便宜 0.39；便宜档的无据引用 6.9% 在级联臂里降到 0；合法动作率三臂都不劣于强模型（δ=2 点）。**这些数只证明机制和统计是通的**，mock 的「质量」是我们编的。
- 「合法动作率」在真实模型上可能很不敏感（基线 A 兜底 0%、强制终结 2.6%），所以报告同时给一次过率、兜底率、守卫拦截率和言行一致；胜率在 10–20 对种子上区间很宽（±15–20 点），只能看方向。
- 一回合内可能出现两个模型的笔迹（便宜模型的坏回复 + 强模型的修正）。这是有意的：强模型需要看到错在哪。
- 强档的超时：命令行两档共用一个 RecoveryPolicy；在线服务取两档 profile 超时的较大值。
- 风险：`loop.py` 的每个失败点都多了一行 `_escalate`；`_call` 多了 `client`/`result` 参数。无级联时这些都是空操作，现有 488 个测试不变通过。

## 真实模型怎么跑

```
python -m werewolf_harness.scripts.paired_cascade --seeds 2001-2010 \
    --cheap claude-cli:haiku --strong claude-cli:sonnet --arms strong,cheap,cascade --cascade-mode both \
    --variant 8p --parallel 8 --record --out runs/cascade/
```

估计（haiku 一局 3.4 分钟、约 56 万 CLI token；sonnet 假设慢一倍、token 相同；级联取两者之间）：每个种子三臂约 15 分钟、168 万 token；10 个种子约 2.5 小时、16.8M token。之后把 `--seeds` 改成 `2001-2020` 再跑同一条命令即续跑到 20 对（再 2.5 小时）。

## 真实模型结果（2026-10-15，`docs/cascade_real.md`）

用 haiku 5.5 和 sonnet 5.5 跑了 10 个种子 × 3 臂，共 30 局，用时 1 小时 48 分钟。

- 级联（both）的成本是强模型的 **0.34 倍 [0.30, 0.40]**，全便宜是 0.08 倍。这是按现行价格算的：两档相差 20 倍。如果按上一代 3 倍价差，级联是 0.64 倍。
- 合法动作率三臂都在 99.6–99.8%，非劣效成立。
- 44.8% 的回合升级到强模型。其中 43.7 个点来自 `decisive` 策略（投票、夜间行动），因为出错而升级的只有 1.1%。便宜模型的一次过率反而比强模型高 3.9 个点 [+0.7, +7.2]。
- 结论：这一代模型上，级联省钱靠的是“哪些回合交给强模型”这条策略，而不是靠“失败后补救”；后者几乎没有触发的机会。胜率每臂只有 10 局，只能看方向。
