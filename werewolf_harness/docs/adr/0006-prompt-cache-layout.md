# ADR 0006：system message 按变化频率排序，提高前缀缓存命中率

- 状态：已采纳
- 代码：`players/llm/context.py`（`cache_layout=True`）

## 问题

厂商的 prompt cache 是前缀缓存：新请求开头有多少和之前的请求逐字相同，就有多少按约 1/10 的价格计费，首 token 延迟（TTFT）也更低。原来的 system message 顺序是：

```
规则 → 【本座位私有身份】 → 防护前言 → 输出契约（含【本回合工具列表】） → 你是 N 号
```

私有身份排在第二段，12 个座位只能共享最前面的规则；本回合工具列表嵌在中间，同一个座位换一个回合也会断。

## 改法

```
规则 → 防护前言 → 静态输出契约 → 发言要求        ← 全桌共享，每回合相同
私有身份 → THIS TURN: 本回合工具 → 你是 N 号       ← 本座位、本回合
```

## 量化

量化时用的是模拟缓存，不是真实账单：`app/llm/sim.py` 按厂商公开的规则实现，128 token 一块做链式哈希，不足 1024 token 不缓存，TTL 5 分钟；喂进去的是真实 harness 构造的 prompt。`scripts/measure_prompt_cache.py` 跑 5 局 12 人局，每种组合 1344 次调用：

| tool schema 在前缀里的位置 | 旧布局 | 新布局 |
|---|---|---|
| tools 在最前（Anthropic 文档写明的顺序） | 62.9% | **68.8%** |
| tools 在最后 | 55.7% | **59.5%** |

按缓存部分 0.1 倍价格估算，输入成本约下降 12%（tools 在最前的情况）。

## 没解决的部分

- 大部分命中来自同一回合内 ReAct 多步之间的复用。跨座位的共享受限于 tool schema：原生 tool calling 把本回合的 tools 放在最前，而 tools 随角色和任务变化。要再提高，就要让 tools 列表稳定（每局固定一份，合法性只靠引擎校验），但这会改变 Agent 的白名单机制，暂不做。
- 真实命中率要用真实 key 跑。gateway 已经从 usage 里读取 `cached_tokens` / `prompt_cache_hit_tokens` / `cache_read_input_tokens`，写进 `llm_calls` 表，所以真跑一次就能拿到真实数字。
