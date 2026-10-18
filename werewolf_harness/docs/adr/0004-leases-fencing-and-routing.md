# ADR 0004：多节点——租约 + fencing；做故障转移，不做分片

- 状态：已采纳
- 代码：`app/runtime/lease.py`、`Store.bump_fencing`、`Store.append`

## 谁来推进一局

- 租约：Redis `SET lease:{game} {node} NX PX 10000`。每 3 s 用 Lua 脚本续期（检查 value 仍是自己才延期）；释放同样用 Lua，保证只删自己的租约。
- 抢到租约后：`UPDATE games SET fencing_token = fencing_token + 1 … RETURNING`，拿到新 token。
- 每次写 journal 都在同一个事务里执行 `SELECT fencing_token … FOR SHARE`，token 不等就拒绝写入（`Fenced`）。

## 为什么只有 Redis 锁不够

一个节点可能被暂停：GC、SIGSTOP、虚拟机卡住。暂停时间超过 TTL 后，它的租约已经过期，别人接管了这一局，但它自己不知道。醒来以后，它手上还有一个模型答案要提交。只靠锁的话，它会写进去，同一局出现两个写者。有了 fencing token，它的写入会被数据库拒绝，actor 随即停止。

测试 `test_a_paused_node_that_wakes_up_cannot_write` 复现了这个场景：A 手里拿着答案被「冻住」，B 接管并走了几步，A 醒来提交，结果是 Fenced，A 的世界一步都没有前进，journal 连续无重复。

## 路由：现在做故障转移，不做分片

`deploy/` 里的两个 API 节点是主备关系：Caddy 用 `lb_policy first` 加 `/readyz` 健康检查，流量全部打到 api1；api1 挂了以后，它持有的局在 TTL 到期后由 api2 接管，Caddy 同时切流。WebSocket 连接会断开，客户端带 `last_seq` 重连。

真正分片（多个节点同时各自驱动一部分局）需要让请求到达「持有这一局的节点」。有三种做法：
1. 代理层按 game_id 做一致性哈希，节点按同一个哈希环抢租约；
2. 在 Redis 维护一张 `game → node` 路由表，由网关（Envoy / OpenResty）查表转发；
3. 任意节点都能接 WebSocket，写操作通过 Redis Stream 转发给 owner，事件通过 pub/sub 回推。

这三种都要额外的组件。当前的规模下，单节点压测已经能承载数百桌（见 `docs/numbers.md`），所以先不做，把路径写在这里。
