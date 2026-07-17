# Console 使用 Objective 时间线并下钻 RunLog

Console 以 ObjectiveLog sequence 为主干投影完整 Objective 时间线，展示一个用户目标如何经过 Assignment、Decision、handoff、预算、故障与用户交互到达当前状态。每个 Assignment 和 Run 节点保留稳定引用，用户需要执行细节时再查询对应 RunLog。开发模式必须展示 Assignment 收尾调用的固定模板、完整输入、输出、reasoning、解析结果、重试和预算结算，且这些调试数据不进入后续普通业务上下文。

## Status

accepted

## Considered Options

- 只展示 AgentState 当前快照：实现简单，但无法解释状态为何形成，也看不到已完成的 peer Assignment。
- 把所有 RunLog step 平铺到一个页面：信息最全，但 Objective 决策主线被大量 token、tool 和 stream 细节淹没。
- 只保存收尾摘要：界面清爽，但无法诊断固定模板、结构解析、模型推理或预算检查中的问题。

## Consequences

- Objective 时间线至少显示 Objective 创建与状态、ObjectiveUserInput、ObjectiveContribution 与 annotations、Assignment 生命周期、文件 Artifact、Decision、handoff、ObjectiveBudget 变化、活动窗口、checkpoint、execution fault 和最终交付。
- ContextCapacityExceeded 必须显示哪些 input 超出容量、原文字数/token 估算和可外置候选；用户授权后显示 ObjectiveUserInput 与 Artifact（path/summary）的稳定关联。普通界面不能把外置表现为原输入被删除。
- 时间线顺序以 ObjectiveLog sequence 为准，不以客户端到达时间、AgentState.updated_at 或多个 Run 的本地 sequence 混排。
- Assignment 节点显示 assignment_id、职责、状态、Outcome 摘要与关联 run_id；Run 节点按需查询 RunLog，不在 ObjectiveLog 中复制 step 内容。
- Objective 尚未完成时，普通用户视图以已提交的 Objective 进度为主体，不展示任何尚未通过验收的 Run token delta 或候选 report。
- Objective 完成并提交 `ObjectiveDelivered` fact 后，页面切换到最终交付视图：普通文本 report 位于主内容区，文件 Artifact 作为附件列表；Objective 时间线默认折叠但可以展开。
- `ObjectiveDelivered` 必须明确记录最终 outcome id、主 report 文本引用（或内嵌文本）、其他交付 artifact ids 与交付时间；前端不得以“最后一条 assistant message”推断交付内容，也不得把文本 report 当成 Artifact。
- `WAITING_USER`、`BUDGET_PAUSED`、`USER_PAUSED` 和 `FAILED` 不进入最终交付视图；它们继续显示当前状态、所需用户动作和已提交过程。
- 普通用户展开的是 Objective 时间线；Run Trace、收尾调用完整输入输出和 token 细节仍位于 Assignment/Run 节点下的开发模式下钻视图。
- 收尾调用调试面展示调用 phase、固定 prompt、完整 messages、模型参数、输出、reasoning、结构解析结果、校验错误、纠正 attempt、token usage 和实际成本计算来源。
- “是否持久化”和“是否进入后续 agent 上下文”是两个独立判断；收尾调用调试数据完整持久化，但不会自动加入 ObjectiveView 或普通对话历史。
- Console API 应提供 Objective 当前投影、按 sequence 分页的 Objective facts，以及按 assignment_id/run_id 下钻的查询入口。
- AgentState 仍可用于实时调度状态和树视图，但不能作为 Objective 历史时间线的来源。
- 文件 Artifact 规则见 ADR 0045。
