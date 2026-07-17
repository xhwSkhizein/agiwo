# Objective 是完整工作的全局边界

Agiwo 使用 `Objective` 表示从一个用户目标开始，到结果被接受、中止或交还用户决定为止的完整问题解决过程。`Session` 可以包含多个 Objective；一个 Objective 可以包含多个 `Assignment`；agent 为履行 Assignment 产生一个或多个现有意义上的 `Run`。Objective 拥有不可变目标、全局预算、协作关系、决策记录和最终结果。

## Status

accepted

## Considered Options

- `ObjectiveRun`：能强调这是一次运行实例，但名称冗余，也容易与现有 agent `Run` 混淆。
- 直接复用 `Session`：无需新增模型，但长期对话会包含多个独立目标，无法为一次问题解决过程设置准确的预算与终态。
- 直接复用 `AgentState`：无需新增模型，但完整 Objective 可以跨越多个接力执行者，任何单个 agent 的状态都无法代表整体。

## Consequences

- `Objective` 保留为全局领域术语，不再表示某个 agent 收到的局部工作。
- 现有 `AgentState.task` 应在实施时改名为 `assignment` 或 `input`。
- 现有 Scheduler `TaskLimits` 和 `TaskGuard` 保持原名；它们表达调度执行限制，不是 Objective 领域对象。Objective 模块不得复用或包装这两个名称来实现 `ObjectiveBudget`。
- 一个 Session 的后续用户目标可以创建新的 Objective，而不会重置 Session 的对话历史。
