# 10 万局时的数据库（phase J）

目标：先造出一个「跑了半年」的库（10 万局已结束 + 几百局还在跑），把服务真正会发的每条查询在这个规模下量一遍；慢的改掉，改完再量；再评估分区和归档，把值得做的做掉。决策和取舍见 [ADR 0012](adr/0012-db-at-scale.md)。

## 1. 数据怎么造的

`scripts/seed_bulk.py`（参数见 `--help`，会 **TRUNCATE** 目标库）：

1. **池子**：用真的 Runtime（mock 模型）在目标库里下一池真局——journal、座位、模型记忆都是服务自己写的，所以每个克隆都能重放到记录的结局。8 人局多下 3 倍、只留 ≥45 步的（mock 局常常 37 步左右就狼胜，太短不像真局），最后 34 局 8p（平均 52 步）+ 10 局 12p。
2. **克隆**：每局换新 id、新 created_at（近 180 天，流量从头到尾涨约 3 倍）、自己的模型组合；30% 的局有 1–2 个真人座位，被 2 万个用户按 Zipf 分布认领（最重的用户 2014 局，典型用户 1 局）；每个模型步 1–3 行 llm_calls（决策调用、约一半有记笔记调用、3% 失败 + 回退）。seed 保持原值——world = f(seed, journal)，种子跟着 journal 走。
3. **装载**：asyncpg 二进制 COPY，已有索引照常在线维护（量的是「按现在的 schema 一路写进来」的库）。另留 300 局 `running`（journal 截在中途、无人持有租约）和 60 局 `waiting`，让启动时的恢复扫描有活干。

```
python -m werewolf_harness.scripts.seed_bulk --url postgresql://postgres@127.0.0.1:5432/wh_scale \
    --games 100000 --running 300 --waiting 60 --rev 0001
```

| 项 | 数 |
|---|---|
| 池子 | 44 局真局，31.9 s |
| COPY 装载 | 500.9 s（约 200 局/s，2 核机器） |
| users / games / game_seats | 20,000 / 100,360 / 882,876 |
| journal / agent_state / llm_calls | 6,168,482 / 5,922,156 / 9,059,275 |
| 库大小 | **12.6 GB**（agent_state 9.2 GB，llm_calls 1.7 GB，journal 1.5 GB） |

`verify_replay` 在这个库上抽最新 500 局已结束的：一致率 1.0。

## 2. 怎么量的

`scripts/bench_db.py`：每条查询先在新连接上跑一次（"首次"），再同一连接跑 9 次取 `EXPLAIN (ANALYZE, BUFFERS)` 的 Execution Time 中位数（"热"）。`--set before|after` 选改动前/后的 SQL，参数（哪个用户、哪局）从数据里挑一次，两边一样。`--e2e` 再走服务自己的 Store/Runtime 代码量端到端耗时。

说明：容器里没权限 drop OS page cache，所以"首次"只是 Postgres 侧冷（新连接、计划缓存空、shared_buffers 只有 128 MB 而库 12 GB），不是磁盘冷；BUFFERS 的 read/hit 数可以看出真冷时要读多少页。机器只有 2 核，同时还有别的 agent 在用同一个 Postgres，端到端数字有 ±30% 的抖动，表里给的是中位数。

## 3. 查询：改前 → 改后

热 / 首次，单位 ms（Postgres 执行时间，不含驱动开销）。"改后" 是迁移 j1 之后、归档之前。

| 查询（谁发的） | 改前 | 改后 | 计划变化（一行） |
|---|---|---|---|
| 恢复扫描 live_games（runtime.scan） | 0.69 / 1.4 | 1.4 / 5.4 | 不变：`games_live_idx` 部分索引，360 行；本来就快 |
| 租约后 bump fencing（runtime.adopt） | 0.10 / 0.38 | 0.04 / 0.23 | 不变：`games_pkey` |
| 接管读 journal | 0.026 / 0.19 | 0.023 / 0.12 | 走的索引从 `journal_game_id_request_id_key` 换成 `journal_pkey`（前者删了） |
| 接管读模型记忆 | 0.034 / 0.16 | 0.029 / 0.15 | 不变：`agent_state_pkey` + Incremental Sort |
| 写 journal 前的 fencing 检查（FOR SHARE） | 0.021 / 0.13 | 0.018 / 0.07 | 不变 |
| 按 request_id 查重 | 0.011 / 0.07 | 0.007 / 0.03 | 全量唯一索引 → 部分唯一索引 `journal_request_uidx`（只收有 request_id 的行） |
| **大厅首页 50 局**（GET /api/games） | **77.7 / 66.9** | **0.10 / 0.62** | Parallel Seq Scan + Sort（读 10 万行） → `games_created_idx` 顺序读 50 行 |
| **大厅第 200 页** | **133 / 156**（OFFSET 9950） | **0.04 / 0.25**（keyset） | Seq Scan + top-N Sort → 同一索引从游标处读 50 行 |
| **我的对局，最重用户**（2014 局） | **20.7 / 52.3** | **0.08 / 0.43** | Bitmap 取全部 2014 座 → join 2014 局 → Sort → `game_seats_user_time_idx` Index Only Scan 取 20 座 → 20 次主键 |
| 我的对局，典型用户 | 0.029 / 0.18 | 0.021 / 0.18 | 同上（行少，差别小） |
| 对局详情：games 行 | 0.011 / 0.06 | 0.010 / 0.04 | 不变：主键 |
| 对局详情：整局 journal | 0.054 / 0.79 | 0.057 / 0.16 | 换成 `journal_pkey`；归档后见 §5 |
| 单局 LLM 用量（/api/stats/llm?game=） | 0.062 / 0.35 | 0.061 / 0.15 | 不变：`llm_calls_game_idx` |
| **LLM 用量看板**（/api/stats/llm） | **5,867 / 14,657** | **68 / 111** | 全表 Seq Scan 900 万行 → `llm_calls_time_idx` 只扫最近 24 h（语义变了：默认 24 h，`hours=0` 才是全量） |
| verify_replay 选样 | 63.9 / 81.3 | 1.9 / 4.3 | Seq Scan + Sort → `games_created_idx`，`state='over'` 在 SQL 里过滤 |
| 归档任务选下一批 | 61.1 / 87.1 | 0.26 / 1.7 | Seq Scan + Sort → 部分索引 `games_archivable_idx`（归档完后 0.018） |

排行榜（/api/leaderboard）在 Redis ZSET 里，不碰 Postgres，这里没有它。

## 4. 端到端：服务自己的代码路径

| 场景 | 改前 | 改后 | 归档 + 压实后 |
|---|---|---|---|
| **启动恢复扫描**（360 局无主：查 live → 问租约 → 接管全部） | **3,376 ms**（9.4 ms/局） | **1,626 ms** | **1,401 ms**（3.9 ms/局） |
| 大厅列表 p50 / p95（Store.recent_games(50)） | **67.0 / 85.9** | **1.8 / 3.1** | 1.9 / 2.9 |
| 我的对局 p50 / p95 / p99（随机 200 个用户） | 1.07 / 2.7 / 6.5 | 1.14 / 2.5 / 6.5 | 1.06 / 1.7 / 5.9 |
| 我的对局，最重用户 | **16.1** | **2.2** | 2.1 |
| 单局重放（读 game + journal + 重建）p50 / p95 | 5.4 / 9.5（读 2.8） | 6.2 / 13.0（读 3.4） | 6.8 / 12.6（读 3.8；热 4.2、归档 3.7） |
| verify_replay 500 局 | 785 ms，**只查到 140 局** | 1,989 ms，查满 500 局 | 2,011 ms |

- 典型用户的「我的对局」两边都是 ~1 ms：SQL 只要 0.02 ms，剩下是 SQLAlchemy/asyncpg 往返开销；只有重度用户才看得出差别。
- 恢复扫描的瓶颈不在 SQL：每局 5 次往返（租约、fencing、games 行、journal、记忆）每次 ~1.5 ms 的驱动开销，加 ~1.7 ms 重放，全在一个 Python 进程里吃 CPU。并发接管（semaphore 8）只到 2.5 s，因为是 CPU 不是 I/O；改成**批量接管**（`Runtime.adopt_many`：租约一个 pipeline、fencing 和 games 行一条 `UPDATE … RETURNING`、journal 一条、记忆一条，每批 50 局）才到 1.4 s，剩下的基本是重放和拉起 actor。
- verify_replay 以前先取最新 500 局再在 Python 里筛 `over`，最新的几百局正好是还在跑的，所以只查了 140 局；现在在 SQL 里筛，journal 每 100 局一条查询（原来每局一条）。每局成本 5.6 ms → 4.0 ms。

## 5. 归档

`scripts/archive_games.py --days 30 [--vacuum]`：结束超过 N 天的局，journal 打包成一个 jsonb 数组进 `game_archive`（TOAST + lz4 压缩），模型记忆只留每个座位最后一条，热表里的行删掉；`games` 行留在原地并打上 `archived_at`，大厅、我的对局、座位外键都不受影响。读路径 `Store.journal/journals/agent_memory` 自动回退到归档表（`load_readonly` 看 `archived_at` 直接先读归档，省一次往返），所以 `/api/games/{id}/log`、重放、verify_replay 对归档局照常工作。

在 10 万局库上跑 `--days 30`：

| 项 | 数 |
|---|---|
| 归档局数 | 76,640（154 批 × 500，单条语句一批） |
| 耗时 | 319.6 s（240 局/s），之后 VACUUM (ANALYZE) 97.8 s；服务在跑也可以（SKIP LOCKED，短事务） |
| journal 行 | 6.17 M → 1.45 M |
| agent_state 行 | 5.93 M → 1.39 M |
| game_archive | 236 MB（heap 49 + TOAST 185 + 索引 3），每局平均 journal 2,270 B、记忆 572 B（压缩后），平均 61.6 步 |
| 每局占用 | 热表约 **107 KB/局**（agent_state 91 + journal 15）→ 归档约 **3.1 KB/局**（约 34×） |
| 读一局（归档 vs 热） | Store 层 p50 3.7 ms vs 4.2 ms；归档读本身 `game_archive_pkey` 一次索引扫描 0.06–0.1 ms |
| 库大小 | DELETE + VACUUM 后物理大小不变（12.8 GB，空间可被新局复用）；一次性 `VACUUM FULL journal, agent_state, games`（46 s，期间锁表；生产用 pg_repack）后 **12.6 GB → 4.6 GB** |

压实后各表：agent_state 2,138 MB（剩下 2.4 万局没归档的）、llm_calls 1,740 MB、journal 314 MB、game_archive 236 MB、game_seats 102 MB、games 71 MB。

## 6. 分区评估

**llm_calls 按月 RANGE 分区**（实测，9 个分区，数据同上；实验 SQL 在本文末尾）：

| 项 | 不分区 | 分区 |
|---|---|---|
| 最近 24 h 看板 | 50–68 ms | 35–39 ms |
| 最近 30 天汇总 | 1.35–1.59 s | 0.73–1.17 s |
| 单局用量（按 game_id） | 0.06 ms | 1.5–2.0 ms（9 个分区的 game 索引都要探一遍） |
| 删最老一个月（36 万行） | DELETE 0.6–0.9 s + 之后 VACUUM、膨胀 | DETACH PARTITION 2 ms |
| 改造成本 | — | 拷 900 万行 31 s + 建索引 21 s，期间要么停写要么双写；主键得变成 (id, created_at) |

结论：**现在不做**。时间窗查询快 1.5–2 倍，但 1.7 GB 的表 24 h 窗口本来就 < 100 ms；单局查询反而慢 25 倍；真正的收益是按月删数据，而目前没有删 llm_calls 的需求。触发条件：llm_calls 过 ~5 千万行或要上保留策略时，按 ADR 0012 里的步骤迁。

**journal 按时间分区**：不做。分区表的主键必须包含分区键，(game_id, seq, created_at) 就不再保证「同一步只写一次」——那是 journal 写入正确性的根基（ADR 0002/0004）。按 game_id 哈希分区能保住主键，但不带来任何「删旧数据」的能力，只是把一个 B 树切成几个；访问永远是按 game_id 点查，一个 B 树已经是 3–4 层，切了也不会更快。归档解决的才是真正的问题（冷数据占热表）。

## 7. 迁移 j1 在线上库上怎么上

`migrations/versions/j1_db_at_scale.py`（down_revision = 0001）。在这 10 万局库上整体 **5.6 s**。

- 建/删索引全部 `CREATE/DROP INDEX CONCURRENTLY`，放在 alembic 的 `autocommit_block()` 里（CONCURRENTLY 不能在事务里跑）：不挡读、不挡 journal 写入，代价是每个索引扫两遍表、而且不是事务性的——失败会留下 INVALID 索引，`IF NOT EXISTS` 会把它当成已存在跳过，所以迁移开头先删掉这几个名字的 INVALID 索引再建。
- 加列（`games.archived_at`、`game_seats.game_created_at`）都没有默认值，只改 catalog，不重写表。
- 回填 `game_seats.game_created_at` 只碰被认领的座位（这里 3.6 万行），而且放在 DDL 事务之外，持的是行锁不是表锁。
- 老的 `UNIQUE (game_id, request_id)` 在新的部分唯一索引**建好之后**才删，唯一性一刻都没断过；`DROP CONSTRAINT` 要一个瞬时的 ACCESS EXCLUSIVE 锁（只改 catalog）。
- 滚动发布期间老节点认领座位不会写 `game_created_at`：这些行排序时 `NULLS LAST`（在「我的对局」里排到最后），不会出错；全部节点升级后可以再跑一次迁移里的回填语句（幂等）。
- downgrade 先把 `game_archive` 里的局解包回 journal / agent_state，再删表——不然降级会丢局。

## 8. 复现

```
createdb -h 127.0.0.1 -U postgres wh_scale
python -m werewolf_harness.scripts.seed_bulk --url postgresql://postgres@127.0.0.1:5432/wh_scale --rev 0001
python -m werewolf_harness.scripts.bench_db --url postgresql://postgres@127.0.0.1:5432/wh_scale --set before --e2e --out before.json
python -m alembic -c werewolf_harness/alembic.ini upgrade head          # DATABASE_URL 指向 wh_scale
python -m werewolf_harness.scripts.bench_db --url ... --set after --e2e --out after.json
python -m werewolf_harness.scripts.archive_games --url postgresql+asyncpg://postgres@127.0.0.1:5432/wh_scale --days 30 --vacuum
python -m werewolf_harness.scripts.bench_db --url ... --set after --e2e --out archived.json
```

注意：`bench_db --e2e` 里的恢复扫描会 bump 活局的 fencing token，只对造出来的库跑。

### 附：llm_calls 分区实验

```sql
CREATE TABLE llm_calls_p (LIKE llm_calls INCLUDING DEFAULTS) PARTITION BY RANGE (created_at);
ALTER TABLE llm_calls_p ADD PRIMARY KEY (id, created_at);
-- 每月一个分区：CREATE TABLE llm_calls_p_YYYYMM PARTITION OF llm_calls_p FOR VALUES FROM (...) TO (...);
INSERT INTO llm_calls_p SELECT * FROM llm_calls;
CREATE INDEX ON llm_calls_p (game_id, seq);
CREATE INDEX ON llm_calls_p (created_at);
-- 比较：24 h / 30 天汇总、单局 EXPLAIN、BEGIN; DELETE 最老一个月; ROLLBACK 对比 BEGIN; DETACH PARTITION; ROLLBACK
```
