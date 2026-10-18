# ADR 0010：引擎版本——每局按它创建时的规则重放

- 状态：已采纳

## 背景

整个服务建立在 `world = f(seed, journal)` 上：崩溃恢复、节点接管（ADR 0004）、只读加载、`cli replay`、
`verify_replay` 都是「用种子和 journal 重放一遍」，而且我们故意不存快照（ADR 0003）。这条等式只对**同一份
`step()`** 成立。改一条规则、修一个引擎 bug，老局再重放就是另一个世界：轻则 `journal diverged` 直接起不来，
重则悄悄接受了整条 journal、赢家也一样，但中间死的人、票型不一样了——恢复出来的局和玩家看到的不是一局。

在这之前没有任何东西记录「这局是哪套规则下打的」。

## 决定

### 1. 一个显式的引擎版本，只数影响 `step()` 的改动

`envs/werewolf.py` 里有 `ENGINE_VERSION`。它只在**某个 (seed, journal) 重放后的世界可能不同**时才加一：
`reset` 的发牌、`step` 做什么、`_next` 下一个问谁、以及 `action_space` 认为什么动作**合法**（`step` 会拒绝不合法的
动作，所以合法性就是 step 行为）。

**视图不算**：`observe` 的内容、工具描述、`rules_text`、`open_turns`。视图不在 journal 里，每次读都从世界现算，
老局在新视图代码下恢复，世界是同一个，只是换了一种呈现。先例就是上一次改动（封闭投票、复盘按轮同时发言）：
它改变了投票进行中一个座位能看到什么、驱动器能提前问哪些回合，但改动之前写下的每一条 journal 仍然一步不差地
重放——所以它是视图修订，不是引擎版本。视图修订记为 `VIEW_REVISION = 2`，和引擎版本一起写进日志，用来解释
「同一条 journal 为什么模型表现不同」，但从不用来选代码。

起点：`BASELINE_ENGINE_VERSION = 1` 就是这次改动之前的规则；没有版本号的日志和数据库行一律按 v1 读。
34 份真实日志在 v1 下全部一致（下面的数字）。

### 2. 规则分支挂在版本上，不复制引擎

每一条影响 step 的改动是 `RULE_CHANGES` 里的一条：一个开关名和引入它的版本。`Rules.at(v)` 把版本变成一组布尔开关，
引擎只按开关分支（`if self.rules.tie_revote:`），从不写 `if version >= 2`，也不复制一份旧引擎。
`Werewolf(variant, engine_version=v)`；不传就是当前版本。超出窗口的版本抛 `UnsupportedEngineVersion`。

为了证明机制，v2 引入了一条真实的小规则：**平票 PK**（`tie_revote`）。白天投票平票时，平票的人不投票，其余人在
平票者之间再投一次；再平则无人出局（v1：平票直接无人出局）。新规则加的公开字段（`tie_votes`、`revote_candidates`）
只在 v2 的视图里出现，v1 的视图一个字节都不变。

### 3. 每局记录它的版本，恢复按记录的版本重建

- `cli play --out` 的日志写 `engine_version` 和 `view_revision`；`cli replay` 用日志里的版本（缺省 = v1）。
- 数据库：`games.engine_version integer NOT NULL DEFAULT 1`（迁移 `e1_engine_version`）。默认值留着是对的：
  滚动发布期间还在跑老代码的节点插入时不带这一列，它打的正是 v1。
- `TableConfig.engine_version`：新建时由 `Table` 钉成当前版本，随 config 一起存；读回一个没有这个键的老 config 时是 v1。
  列是恢复的依据（`Runtime.table_from_row`），config 里的副本始终与列相同，给 Streams worker 用。
- 老节点遇到新版本的局（滚动发布时 v1 节点扫到 v2 节点建的局）：不接管，放掉租约，记进 `Runtime.incompatible`，
  等能重放它的节点来接。

### 4. 兼容性检查器

`python -m werewolf_harness.scripts.compat_check <日志 glob…> [--db] [--fingerprint] [--against V] [--strict] [--json]`

每局重放两次：按**记录的版本**（必须一致，否则就是旧规则分支被误改，恢复会出错，退出码 1）和按**当前版本**
（分叉不是 bug，而是「这次规则变更会影响哪些老局、从第几步开始」）。每局报告：一致 / 第 N 步分叉
（journal 这一行是谁的什么回合、引擎实际要什么、首个不同的公开状态是哪些字段、相关的规则变更）/ 崩溃。

`--fingerprint` 再逐步比较公开状态（`Werewolf.public_state()`：票按实际投出的记，不管是否封闭；不含规则文本）的哈希，
以及完整私有世界（`world_state()`）的哈希。这能抓住「journal 全部接受、赢家相同、中间世界不同」的改动，
仅比较结局是看不见的（测试里用「复盘发言被截断」的改动演示：结局模式说一致，指纹模式指出第一句复盘那一步）。

### 5. CI

`tests/corpus/` 存 74 份 journal：34 份真实日志（只留 variant/playbook/seed/outcome/journal，去掉动作里引擎不读的
`thought`；仍不带版本号，走 v1 缺省路径）+ 4 个变体 × 若干 playbook 的 mock 局，v1、v2 各 20 局。每份文件带上写入时
在它自己版本下算出的逐步指纹。`tests/test_engine_compat.py` 逐份按记录版本重放并比较指纹，任何一份不一致即失败。
`make compat` 跑同样的检查并打印当前版本下的影响面；`make compat-db` 对 Postgres 里的全部对局做同样的事。
语料只**追加**（`scripts/build_compat_corpus.py`），不重新生成已有版本的文件——用改过的代码重算旧文件等于取消检查。

## 量出来的数

`compat_check '/home/claude/data/runs/*/*.json' --fingerprint`（34 份真实日志，都是 8 人局，均无版本号）：

| | 一致 | 分叉 | 崩溃 |
|---|---|---|---|
| 按记录版本 v1 | 34 | 0 | 0 |
| 按当前 v2 | 30 | 4 | 0 |

分叉的 4 局正好是出现过「平票、且平票者以外还有活人可以重投」的那几局：`base/g1007`、`base/g1013`（第 18 步）、
`base/g1029`（第 20 步）、`pilot/g101`（第 35 步）；每一局都是「期望 X 号 / night（v1：平票无人出局，入夜），
实际 Y 号 / vote（v2：PK 重投）」，首个不同的公开状态是 `exiles, phase, revote, round, tie_votes, votes`。
测试用一个只看 v1 世界的独立判定（找「平票、且有不在平票者里的活人」的那一票）核对：检查器列出的局集合与分叉步数
与它完全相同。整个语料 74 局：记录版本 74/74 一致；对 v2 66 一致 / 8 分叉。耗时：34 局双版本 + 指纹约 3 s
（含解释器启动），74 局约 8 s。

集成测试（`tests/integration/test_engine_versions.py`）：节点 A 以 v1 代码导入 g1007 的前 18 步（恰好停在平票那一票之后），
停服；代码升到 v2；节点 B 接管——按 v1 重建，下一回合是 `4 号 / night`，指纹与独立 v1 重放相同（同一 journal 在 v2 下
根本重放不了），之后用 mock 模型打完，平票保持无人出局、从未出现 PK，全局 journal 在 v1 下一致；B 新建的局记为 v2。
另一个测试：v1 老节点扫描到 v2 局时不接管、不占租约，升级后的节点随即接管。

## 支持窗口

`MIN_ENGINE_VERSION..ENGINE_VERSION` 之间的每个版本都能重放。退役一个版本（提高 `MIN_ENGINE_VERSION`、删掉只为它存在的分支）
的条件：数据库里没有该版本的活局（`compat-db` 或 `SELECT engine_version, count(*) FROM games WHERE state IN
('waiting','running') GROUP BY 1`），并且已结束的局不再需要逐步重放（复盘页、审计）——实践上保留最近 3 个版本，或者
最老的活局结束后再过 90 天，取较晚者。退役后该版本的 corpus 文件一并删除，已结束局的结局仍在 `games.outcome` 里。

## 后果

- 规则变更变成一个有成本的、显式的动作：加一条 `RULE_CHANGES`、在分支处读开关、`ENGINE_VERSION` 自动加一、往
  语料里追加该版本的局。忘了挂版本直接改，CI 会在语料上按局报出来。
- 引擎里会累积 `if self.rules.x:` 分支，这是保留老局可重放的代价；窗口和退役规则限制了它的增长。
- `public_state()` 本身成为兼容面的一部分：给它加字段会让存下的指纹全部失配，所以新字段只在新规则真正生效后才出现。
- 视图改动不需要版本号，但也就得不到保护：恢复后的老局会用新视图呈现。这是有意的——视图不影响世界。
- 引擎版本和 variant 是两个维度：variant 是一局的配置（几个人、有没有警长），版本是规则代码的演进。原来 `Variant.tie_policy`
  这个只有一个取值的字段删掉了，平票怎么处理现在是版本化的引擎规则。
- 实验可以显式用旧规则开局（`cli play --engine-version 1`，或在代码里 `TableConfig(engine_version=1)`），做规则版本之间的对照实验。
