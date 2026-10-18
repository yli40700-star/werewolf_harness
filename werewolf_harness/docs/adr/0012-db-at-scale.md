# ADR 0012：10 万局时的数据库——索引、keyset 分页、批量接管、归档，不分区

- 状态：已采纳（数字见 [docs/db_scale.md](../db_scale.md)）

## 背景

schema（0001）是按「一个节点几十局」写的。造了一个半年、10 万局已结束 + 360 局活局的库（`scripts/seed_bulk.py`，12.6 GB）把每条查询量了一遍，问题集中在四处：

1. **按时间倒序的列表没有索引**。大厅首页（`ORDER BY created_at DESC LIMIT 50`）、verify_replay 选样、归档选批，每次都是全表并行扫描 + 排序，60–80 ms，且随局数线性涨；翻页没有游标，只能 OFFSET，第 200 页 133 ms。
2. **「我的对局」先 join 再排序**。`game_seats_user_idx (user_id, game_id)` 能找到一个人所有的座位，但排序键 `created_at` 在 games 上，重度用户（2014 局）每次要 join 全部 2014 局再排，20 ms。
3. **用量看板全表扫 llm_calls**。900 万行，热 5.9 s、首次 14.7 s。
4. **冷数据占满热表**。agent_state 每个模型步一行 ~1.5 KB 的信念表，占了库的 73%；结束的局永远不会再写，却和活局挤在同一张表、同一个索引里。
5. 另外：启动恢复扫描 360 局要 3.4 s——每局 5 次往返的驱动开销，不是 SQL。

恢复扫描本身的 SQL（`games_live_idx` 部分索引）、journal 主键、fencing 检查在这个规模下都还是亚毫秒，不用动。

## 决定

**迁移 j1_db_at_scale**（全部 CONCURRENTLY，在线可上）：

- `games_created_idx (created_at DESC, id DESC)`：大厅、verify_replay 选样都走它；`id` 做平局决胜，keyset 游标 `(created_at, id)` 才是全序。
- `game_seats.game_created_at` + `game_seats_user_time_idx (user_id, game_created_at DESC NULLS LAST, game_id DESC) INCLUDE (seat) WHERE user_id IS NOT NULL`：把局的创建时间抄到座位上（认领座位时写），「我的对局」就是一次索引范围读 20 行再 20 次主键。替换掉旧的 `game_seats_user_idx`。这是反范式，但这个值写一次就不会变（局的 created_at 不变），没有一致性维护负担。
- `journal_request_uidx` 部分唯一索引（`WHERE request_id IS NOT NULL`）替换 `UNIQUE (game_id, request_id)`：只有真人点击带 request_id，旧约束给 600 万行每行都建了一个索引项（59 MB），新的几乎为零，`append` 的「同一次点击只落一次」语义不变（仍然是唯一索引冲突 → DuplicateStep）。
- `games.archived_at`、`game_archive`、`games_archivable_idx`：见下。

**查询改写**：

- `Store.recent_games(limit, before)`、`Store.user_games(user_id, limit, before)` 用 keyset 分页；API `GET /api/games?before=<cursor>` 和新增的 `GET /api/me/games?before=` 返回 `next` 游标（base64url，内容是最后一行的 `(created_at, id)`）。不用 OFFSET。
- `GET /api/stats/llm` 默认最近 24 h（`llm_calls_time_idx`），`hours=0` 才全量。单局用量（`?game=`）不变。
- `verify_replay` 在 SQL 里筛 `state='over'`（以前在 Python 里筛，最新的活局会把样本挤掉，500 只查到 140），journal 按 100 局一批取（`Store.journals`）。

**批量接管**：`Runtime.scan` 先一次 MGET 问清所有活局的租约，再按 `recovery_batch`（默认 50）调 `adopt_many`：租约一个 pipeline、fencing 加一并返回 games 行一条 `UPDATE … WHERE id = ANY(…) AND state IN (live) RETURNING`、journal 一条、记忆一条。语义和逐局 `adopt` 一样（先拿租约、再 bump fencing、只接还活着的局，结束了的把租约还回去），一局失败不影响同批其它局。3.4 s → 1.4 s。单局 `adopt()` 保留给别的调用方。

**归档**：`scripts/archive_games.py --days N`（`Store.archive_batch`）把结束超过 N 天的局打包：journal 变成 `game_archive.journal` 里一个 jsonb 数组 `{"v":1,"rows":[[seq,actor,task,action,forced,request_id],…]}`（TOAST + lz4），模型记忆只留每个座位最后一条，热表行删除。一条语句一批（数据修改 CTE 共享一个快照，打包的就是删掉的那些行），`FOR UPDATE SKIP LOCKED` 让它能和服务、和另一个归档进程同时跑。

**games 行不动**，只打 `archived_at`。这是有意的：大厅、我的对局、座位和归档表的外键都指向它，它很小（10 万局 70 MB），而真正的体积在 journal 和 agent_state（每局 107 KB → 归档后 3.1 KB）。把 games 也搬走就得在所有列表查询里 UNION 两张表。

读路径不变：`Store.journal / journals / agent_memory` 在热表里没有时回退到归档表（`load_readonly` 看到 `archived_at` 就先读归档），重放、`/log`、verify_replay 照常；格式带版本号，`unpack_journal` 遇到不认识的版本直接报错。

**不分区**。llm_calls 按月分区实测：时间窗查询快 1.5–2 倍（本来就 < 100 ms），按 game_id 查慢 25 倍（要探每个分区），真正的收益是 DETACH 一个月 2 ms vs DELETE 0.6–0.9 s + 膨胀——而现在没有删 llm_calls 的需求。改造要拷全表、主键改成 (id, created_at)、期间停写或双写。journal 按时间分区会让主键必须含 created_at，(game_id, seq) 就不再唯一——这是「同一步只写一次」的根基，不能拿它换。

## 后果

- 大厅任意一页、重度用户的我的对局、归档选批、verify 选样都变成固定的 < 1 ms 索引读，不再随局数增长。
- 看板默认语义从「全部历史」变成「最近 24 h」。要看全量的调用方得显式 `hours=0`（还是全表扫，秒级）。
- 归档后热表的大小只和最近 N 天的局数有关。第一次大归档后要一次 `VACUUM FULL`/`pg_repack` 把空间还给 OS（12.6 GB → 4.6 GB）；之后每晚跑，DELETE 留下的空间被新局复用，不需要再压实。
- 归档过的局被重新「激活」（比如从中途续局）不支持，也不需要：续局走 `import_log`，会生成新局。
- `game_created_at` 在滚动发布期间可能有 NULL（老节点认领的座位），排序用 `NULLS LAST` 兜住，升级完再跑一次回填即可。
- 合并注意：`app/runtime/runtime.py`（scan/adopt 重构）、`app/api/main.py`（`/api/games` 返回多了 `next`，新增 `/api/me/games`，`/api/stats/llm` 多了 `hours`）、`app/storage/db.py`、`app/config.py`（`recovery_batch`）。

## 没做、但下一步值得做

- **agent_state 只留最新一条**。只有 `agent_memory()`（取每座最新）读它，历史从来没人读；如果改成每 (game_id, seat) 一行的 upsert，活局的 agent_state 会从 ~60 行/局降到 ≤12 行/局。没做的原因：信念表的演变可能是实验分析要的数据，改不改由负责实验的人定。归档已经把老局的这部分压掉了。
- **llm_calls 分区**：表过 ~5 千万行或要定保留期时做。步骤：建按月分区的新表（PK (id, created_at)）→ 双写 → 回填历史 → 切读 → 删旧表；之后每月建下个月分区、DETACH 超期分区。
- **看板全量统计**：要的话加一张按天的汇总表，由归档任务顺手写，别再全表扫。
