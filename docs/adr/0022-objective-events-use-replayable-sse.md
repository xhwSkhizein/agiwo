# Objective 事件使用可重放 SSE

`GET /objectives/{objective_id}/events` 使用 Server-Sent Events 推送已提交的 ObjectiveLog facts。每条 SSE event 的 id 等于 ObjectiveLog sequence；客户端通过标准 `Last-Event-ID` 或显式 `after_sequence` 指定最后确认位置。服务端先从 ObjectiveStore 补发缺失 facts，再切换到实时通知。用户输入继续通过 POST 提交，SSE 连接只负责服务器到客户端的 Objective 状态传播。

## Status

accepted

## Considered Options

- WebSocket 同时承载事件与输入：双向能力完整，但重连、幂等输入和持久化游标更复杂，当前交互不需要任意双向消息。
- 只推实时内存事件：延迟最低，但断线期间事件丢失，客户端无法证明时间线完整。
- 轮询 Objective 当前快照：实现简单，但看不到中间 Decision 和预算变化，也会重复传输整个投影。

## Consequences

- Objective SSE 只发布已经进入 ObjectiveLog 的可重放事实；内存通知可以用于唤醒订阅者，但不能成为事件内容的真相源。
- SSE `id` 使用 ObjectiveLog sequence。重连时优先读取 `Last-Event-ID`，也允许 API 客户端使用 `after_sequence`；两者冲突时边界层必须明确拒绝或采用唯一固定优先级。
- 服务端先查询并发送 sequence 大于游标的历史 facts，再订阅新提交通知；切换过程必须再次检查最新 sequence，避免补发与实时订阅之间留下空洞。
- 慢客户端不得阻塞 Objective 执行或 ObjectiveLog 提交；连接缓冲超过限制时可以断开，让客户端凭 sequence 重连。
- SSE 断开不停止或暂停 Objective。用户输入、预算调整、用户暂停和恢复都使用独立的幂等 HTTP command。
- Objective SSE 不发送 Run 的 live token delta。普通用户只接收已提交的进度 fact 和通过验收后的 `ObjectiveDelivered`；Run 明细继续复用现有 AgentStreamItem，并只在开发调试下钻中展示，避免在 SDK core 创建第二套 Run live-output protocol。
