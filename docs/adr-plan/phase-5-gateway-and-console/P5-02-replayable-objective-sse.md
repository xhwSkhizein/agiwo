# P5-02：提供可重放 Objective SSE

状态：planned

## 目标

通过 SSE 按 ObjectiveLog sequence 发送已经提交的 facts。客户端断线后可以从最后确认的 sequence 补发，再切换到实时通知；SSE 只传播 Objective 状态，不承载用户输入或未验收 Run token。

## 对应决定

- ADR 0022：Objective 事件使用可重放 SSE。
- ADR 0020：普通用户只看到已提交的 Objective 进度和正式交付。

## 依赖

- P1-03、P5-01 已完成。

## 实施步骤

1. 提供 `GET /objectives/{objective_id}/events`，event id 等于 ObjectiveLog sequence。
2. 游标支持标准 `Last-Event-ID` 和 query `after_sequence`；两者同时存在且值不一致时返回 400，一致时使用该值。
3. 建立连接后先从 ObjectiveStore 查询 sequence > cursor 的 facts，按升序发送。
4. 历史补发后订阅进程内 commit notification，再次查询最新 sequence，填补“查询结束到订阅生效”之间的竞态窗口。
5. 实时 notification 只用于唤醒；事件内容始终重新从 ObjectiveStore 读取，不能以 queue payload 为真相。
6. 慢客户端使用有界缓冲；超过限制主动断开，让客户端凭 sequence 重连，不能阻塞 ObjectiveLog commit。
7. SSE 断开、浏览器关闭或 proxy timeout 不 pause/cancel Objective。
8. 事件 payload 使用稳定 API DTO，包含 kind、sequence、occurred_at 和用户可见摘要/引用；敏感 debug payload 不直接广播。
9. 不发送 Run live token delta、未验收 candidate report 或 finalization reasoning。开发下钻继续使用现有 AgentStreamItem/RunLog 查询。
10. terminal Objective 在补发完最后 facts 后可结束连接；重连仍能完整读取。

## 主要改动位置

- `console/server/routers/objectives.py`
- `console/server/services/objective_event_stream.py`
- ObjectiveStore commit notification
- `console/server/models/objective.py`
- `console/tests/test_objective_sse.py`

## 测试计划

- 从 0、任意中间 sequence 和 terminal sequence 补发。
- Last-Event-ID/after_sequence 相同、冲突、非法值。
- 历史查询与实时订阅竞态不漏不重。
- 慢客户端断开不阻塞 producer，重连补齐。
- 服务重启后只靠 store 补发。
- payload 不含 run delta、reasoning 或未交付 report。

## 完成标准

- 客户端可以证明 sequence 连续；重复连接不会丢事件。
- 内存通知丢失不导致事实丢失。
- 用户输入和控制命令不通过 SSE 反向提交。
- SDK core 没有新增第二套 Run stream protocol。
- SSE tests 与 Console backend 门禁通过。

## 风险与回退

“先订阅再查历史”也可能导致重复；允许客户端按 id 去重，但服务端仍应使用双读游标把重复限制在确定范围。绝不能用仅内存广播替代 store replay。

