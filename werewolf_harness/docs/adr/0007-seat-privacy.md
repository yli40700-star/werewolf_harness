# ADR 0007：座位隐私——seat token，公共事件流去掉夜间行动者和思考

- 状态：已采纳
- 代码：`app/auth/tokens.py`、`app/runtime/table.py`（`PUBLIC_TASKS`、`public_action`）

## 旧服务上发现的三个泄露

1. `GET /api/games/{id}?seat=N` 不做鉴权：改 URL 里的 N，就能读任意座位的私有视角（身份、狼队友、查验结果），`POST /act` 也可以替别人出手。
2. SSE 事件流里的 `turn` 事件带完整 action，包括夜里刀谁、毒谁，以及每个座位的 `thought`。任何拿到 SSE 地址的人都能看到。
3. 「现在轮到谁」会显示夜间行动者。夜里第一个行动的是预言家，所以观众从行动顺序就能推出身份。

## 修复

- 游客登录拿 user token；`/join` 由数据库做先到先得（`UPDATE … WHERE user_id IS NULL OR user_id = :u`），成功后签发 **seat token**，内容为 (game, seat, exp)，有效期 6 小时。读私有视角、出手、开局、连 WebSocket 都要 seat token。
- 公共事件和 `acting`：任务不在 `PUBLIC_TASKS` 里的，行动者和动作都隐藏成 `hidden`；公开任务的 action 也要经过 `public_action` 去掉 `thought`。
- 以上都有回归测试：`test_a_seat_is_private_to_whoever_claimed_it`、`test_events_never_name_a_night_actor_or_carry_a_thought`、`test_sse_for_spectators_carries_only_public_events`。

## 取舍

JWT 是无状态的：校验只需要验签加检查过期时间，不用查会话表；代价是不能提前吊销。所以 seat token 的有效期设得短，user token 长。需要踢人时，靠 `game_seats.user_id` 换人，再加一个黑名单（尚未实现）。
