# Non-retryable Fault 按 Run 是否可继续分流

`non_retryable` 只表示不得再次执行同一操作，不直接把 Objective 判为失败。若普通工具失败已经形成结构化 ToolResult，且当前 Run 仍能调用模型，ToolResult 留在原 Run 中，由当前 agent 判断是否更换参数、工具或策略。若模型认证、配置、权限等故障直接阻断 Run，执行系统从结构化 fault 和已提交事实生成 普通文本 report，并机械地产生 `HandoffDecision(target=user, expects_reply=true)`；Objective 进入 `WAITING_USER`，不创建大概率遭遇同一基础设施故障的新 agent。

## Status

accepted

## Considered Options

- 所有 non-retryable fault 都终结 Objective：状态最简单，但把局部工具失败错误地上升为全局失败。
- 所有 non-retryable fault 都 handoff 给新 agent：自动化程度高，但认证、配置和权限故障通常由多个 agent 共享，新上下文无法修复。
- 所有 non-retryable fault 都交给用户：最保守，但会打断 agent 本可通过替代工具自行恢复的工作。

## Consequences

- Fault 必须明确记录 `run_blocking`，执行系统不能通过错误文本猜测当前 Run 是否还能继续。
- `run_blocking=false` 的 tool fault 作为 ToolResult 提交到当前 Run；它不结束 Assignment，也不自动产生 handoff。
- `run_blocking=true` 的 non-retryable fault 结束当前 Run 和 Assignment，系统直接生成 fault report 与 `HandoffDecision(target=user, expects_reply=true)`，不调用故障模型完成总结。
- 用户 handoff 不消耗 `max_handoffs`，并关闭当前 Objective 活动窗口。
- 用户修复配置、权限或认证问题并回复后，系统创建新 Assignment；不恢复因不可重试故障结束的旧 Run。
