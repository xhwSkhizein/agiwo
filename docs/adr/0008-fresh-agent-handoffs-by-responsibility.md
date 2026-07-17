# 接力按职责创建新的 Assignment Run

`HandoffDecision.target` 只表达 `agent`、`verifier` 或 `user` 三种下一阶段职责，不引用具体 agent config、agent 名称或 pattern。`agent` 和 `verifier` handoff 都创建同一 Objective 下的新 Assignment，并在当前 Session 的 persistent agent identity 下启动新 Run；接棒 Run 根据 ObjectiveView、最近一次 AssignmentOutcome 与 Session 可见消息历史自主构建具体的 Agent、Parallel、Pipeline 或 pattern。`user` handoff 使用 `expects_reply` 明确机械状态：`true` 暂停等待用户输入，`false` 交付结果并结束 Objective。

## Status

accepted

## Considered Options

- 由前一 agent 选择具体 executor 或 pattern：可以少一次干净上下文判断，但要求它读取能力目录，并让已经受执行轨迹影响的上下文承担重大路线选择。
- 由 Scheduler 根据任务语义选择 executor：路由集中，但会把语义决策泄漏到机械调度者。
- 预先规划完整 workflow：执行路径清楚，但违背由问题复杂度在运行时决定展开深度的原则。

## Consequences

- 前一 agent 只决定下一阶段属于继续工作、验证还是用户边界，不决定具体执行实现。
- 接棒 Run 获得 ObjectiveView，并继续使用 Session 中可见的消息历史；ObjectiveView 提供权威状态，调试级 RunLog 不因 handoff 被复制进输入。
- 动态构建仍受系统给定的模型、工具、skill、pattern 和预算权限约束，接棒 agent 无权扩大权限。
- handoff 创建 peer Assignment 和新 Run，不复用 delegation 的 parent-child 关系，也不恢复已经结束的旧 Run。
- `target=user, expects_reply=true` 令 Objective 进入等待用户状态；`false` 令 Objective 在交付后进入终态。
- `target=agent` 与 `target=verifier` 只决定下一 Assignment 的职责和输入模板；二者复用同一个 Session agent identity，不映射到不同的 registry config 或具名 agent。
- `target=agent` 与 `target=verifier` 属于自动接力并消耗 `max_handoffs`；`target=verifier` 还同时消耗 `max_verification_attempts`。
- `target=user` 不消耗 `max_handoffs`，用户回复后创建新的 Assignment 也不计为自动接力。用户边界不能因为自动执行额度耗尽而被阻断。
