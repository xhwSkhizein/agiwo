# 重试耗尽后接力给全新 Assignment

幂等且标记为 `retryable` 的阻断性操作在当前 Run 内耗尽自动重试次数后，执行系统不暂停或终结整个 Objective，也不要求已经失败的模型再生成总结。系统从结构化 fault、attempt 历史和最后已提交事实构造普通文本 report，并机械地产生 `HandoffDecision(target=agent)`。ObjectiveService 在 ObjectiveBudget 允许时原子结束当前 Assignment、消费配额并创建全新的 peer Assignment/outbox；Scheduler 随后派发新 Run，令其选择其他执行方式。

## Status

superseded by ADR-0046 / ADR-0047

> 重试耗尽后机械产生 NextAction=continue_work 并派发新的 work root Run；不再创建 Assignment，也不再提交 HandoffDecision JSON。

## Considered Options

- 在原 Run 中无限或继续追加重试：保留上下文，但会让局部基础设施故障变成无边界循环。
- 立即暂停 Objective 交给用户：最保守，但会把新 agent 可能自动绕开的暂时性故障过早暴露给用户。
- 让失败模型先生成总结和 Decision：符合普通 Assignment 收口形态，但模型本身可能就是故障源，无法保证收口发生。

## Consequences

- 只有会阻止当前 Run 继续推进的 retryable fault 在重试耗尽后触发该规则；普通 tool failure 若已经形成可供 agent 判断的 ToolResult，仍可留在当前 Run 中由 agent 选择替代方案。
- 普通文本 report 由执行系统创建，至少包含失败操作、结构化 fault、各 attempt 结果、最后 checkpoint 和已有 Artifact 引用；它不伪装成 LLM 总结。
- `HandoffDecision(target=agent)` 是重试策略确定的机械结果，不由模型选择，也不允许改成继续旧 Run。
- 当前 Run 和 Assignment 以可审计的 interrupted outcome 结束；接棒方创建新的 Assignment 与 Run，不复活失败的旧 Run。
- 该 handoff 消耗 `max_handoffs`。若无法取得 handoff 配额，整个 Objective 进入 Budget Pause，等待用户调整配额。
- 新 Assignment 可以选择不同工具或执行策略，但仍受默认 AgentConfig、Objective 权限和剩余预算约束。
