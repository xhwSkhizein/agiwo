# Objective Gateway 使用统一的异步 Objective API

Objective Gateway 接收用户目标后持久化创建 Objective，并立即返回稳定 `objective_id`，不让初始 HTTP 请求阻塞到动态 workflow 结束。客户端通过独立接口读取当前 Objective 投影、按 ObjectiveLog sequence 订阅事件，并提交用户输入或恢复请求。Console 可以对很快完成的 Objective 自动保持等待，从用户视角呈现即时回复，但这只是界面行为，不产生第二套同步后端生命周期。

## Status

accepted

## Considered Options

- 请求一直阻塞到 Objective 终态：短任务体验直接，但无法自然承载断线重连、WAITING_USER、预算暂停和长时间运行。
- 同时提供同步 Objective 与异步 Objective 两套 API：调用方选择灵活，但会造成两套取消、错误、stream 和恢复语义。
- 只返回流、不提供持久化 objective_id：实时体验简单，但连接断开后客户端难以定位原任务并恢复观察。

## Consequences

- 创建入口采用 `POST /objectives`，在 ObjectiveCreated 与可行的初始 Assignment 派发可靠提交后返回 `objective_id` 和当前状态。若初始用户输入本身超过模型上下文容量，则提交 ContextCapacityExceeded 并返回 WAITING_USER，不创建不可执行的 outbox。
- 所有写命令使用持久化 command receipt 实现幂等。scope 固定为命令种类与目标：create 使用 `session:{session_id}:objective:create`，已有 Objective 的命令使用 `objective:{objective_id}:{command_kind}`；receipt 保存 idempotency key、规范化请求摘要、状态和首次响应。
- 同一 scope/key 与相同请求摘要重放首次响应；同 key 不同摘要返回 `idempotency_conflict`，不能静默复用。create 的 receipt 与 ObjectiveCreated、Session 活动占用、初始 Assignment/outbox 在同一 ObjectiveStore 事务中提交，因此 objective_id 产生前也能全局查重。
- `GET /objectives/{objective_id}` 返回从 ObjectiveLog 构建的当前 Objective 投影，不从 AgentState 拼接领域状态。
- `GET /objectives/{objective_id}/events` 提供按 ObjectiveLog sequence 排序的增量事件，支持客户端从已确认位置恢复读取。
- `POST /objectives/{objective_id}/inputs` 持久化新的用户输入，再依据当前 Objective 状态决定恢复等待、调整预算或通知执行中的 Assignment。
- `POST /objectives/{objective_id}/inputs/{input_id}/externalize` 是幂等控制命令：只有真实用户边界可以调用；它把原文物化到 `sessions/<session_id>/artifacts/`、登记 Artifact（path/summary）、追加外置授权 fact，并在容量重新检查通过后继续待执行 Assignment。该命令不删除 ObjectiveLog 中的原始 ObjectiveUserInput，也不产生新的用户语义消息。
- `POST /objectives/{objective_id}/pause` 请求 Objective 经 DRAINING 保存 checkpoint 后进入 USER_PAUSED；`POST /objectives/{objective_id}/resume` 恢复相同 Runs。Objective Gateway 不用现有 Scheduler.cancel 实现暂停。
- 网络连接、浏览器页面和 Objective 生命周期彼此独立；客户端断线不会停止、暂停或复制 Objective。
- Objective 到达可交付终态后，当前投影包含普通文本 report、文件 Artifact 列表与交付状态；客户端不需要回到最初的创建请求取得结果。
- Console、Web channel 和其他入口复用同一 Objective Gateway use case，不直接为某个渠道建立独立 workflow 生命周期。
- 现有 `POST /sessions/{session_id}/input` 保留为兼容的 Session 交互入口，但内部不再直接调用 `Scheduler.route_root_input()`：Session 没有活动 Objective 时调用 `ObjectiveService.create_objective()`，存在活动 Objective 时调用 `ObjectiveService.submit_input()`，并向调用方返回关联 objective_id。
- Feishu 等渠道继续使用现有 Session 解析与当前 Session 指针，解析完成后的普通消息统一进入同一个 ObjectiveService 路径。
- `Scheduler.route_root_input()` 继续作为底层 SDK 和开发调试能力；正常 Console/渠道用户路径不得绕过 ObjectiveService，否则不会获得 ObjectiveBudget、Assignment、Outcome、验收和恢复语义。
