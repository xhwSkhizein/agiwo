# Objective 只在用户输入与 Assignment 收尾边界更新

Objective 是 Session 中当前活动目标的全局语义模型，RunPlan 是某次 Run 完成当前责任的阶段计划。两者由不同事实驱动：用户输入立即追加 Objective 的权威目标事实；Assignment root Agent 只在系统必然发起的收尾调用中提出非权威的结构化 Objective 修订、ObjectiveContribution 与 Contribution annotations。Run 内不存在修改 Objective 的 runtime tool，Agent 只通过 `update_plan` 自主管理当前 RunPlan。

## Status

accepted

## Considered Options

- 提供 `update_task` runtime tool，让 root Agent 在 Run 内随时修改 Objective：即时性最高，但临时推测会在 child 结果尚未汇合、Assignment 尚未结束时提前成为全局状态，也让一次 Objective 版本难以对应明确的用户输入或 Outcome。
- 让 child Agent 直接更新 Objective：发现可以立即共享，但多个并行分支会竞争写入全局语义状态，且 child 不承担 Assignment 的最终责任。
- 只在 Objective 完成时生成一次最终摘要：写入最少，但 handoff 链中的后继 Agent 无法获得持续演进的全局目标理解。
- 用户输入与 Assignment 收尾分别形成 Objective 修订：写入时机与权威来源明确，同时保留动态接力所需的跨 Assignment 目标连续性。

## Consequences

- 用户语义输入以稳定 input id 和完整 UserMessage 原样写入 ObjectiveLog，成为唯一的 ObjectiveUserInput 事实；程序不从自然语言中抽取片段，也不创建第二份用户侧模型。pause、resume、预算调整等结构化控制命令不产生 ObjectiveUserInput；agent 贡献不能提升为用户事实。
- 每个正常或中断结束的 Assignment 都通过既有系统收尾路径形成唯一 AssignmentOutcome；收尾结构可以包含可选 `objective_update`、`new_contributions` 与 `contribution_annotations`（格式见 ADR 0006）。
- `objective_update` 只表达意图、范围、成功标准理解、假设及其依据，不包含 RunPlan、下一步动作、handoff target、预算或状态变更。没有值得全局保留的新理解时该字段为 `null`。
- ObjectiveService 校验字段、来源引用、预期 Objective revision 与不可变边界后，把合法更新作为 append-only ObjectiveLog facts 提交；它不判断分析在语义上是否正确。revision 不匹配时跳过该 update 并记录拒绝 fact，Outcome 仍可提交。
- pause、cancel 或预算暂停不结束 Assignment，因此不提交 Objective 修订。当前进展继续由 RunLog、RunPlan、Artifact 和 checkpoint 保存，恢复同一个 Run 后继续工作。
- child Agent 只把发现返回 root。root 决定哪些内容进入 Outcome、ObjectiveContribution、annotations 或 Objective 修订，避免并行写入竞争。
- Scheduler 不导入 `agiwo.objective`、不读取 ObjectiveLog，也不理解 Objective 修订；Objective 模块不向 Agent 注入 Objective update system tool。
- 所有收尾输入输出、解析、Objective 修订校验和拒绝原因完整进入 RunLog 与开发视图；提交后的 Objective facts 进入 ObjectiveLog 和 Objective Timeline。
