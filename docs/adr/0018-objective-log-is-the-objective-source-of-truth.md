# ObjectiveLog 是 Objective 的 append-only 真相源

Objective 使用独立的 append-only `ObjectiveLog` 保存跨 Assignment 生命周期事实，`Objective` 只是从日志重建的当前投影视图。ObjectiveLog 跟随 RunLog 的全局存储配置并使用同一个物理数据库，写入独立表（MVP：memory / SQLite）。现有 AgentStateStorage 继续保存可覆盖的调度快照，RunLog 继续保存单次 Run 的执行事实。三者通过稳定的 `objective_id`、`assignment_id` 与 `run_id` 关联，不在 ObjectiveLog 中复制完整 RunLog。

## Status

accepted

## Considered Options

- 继续扩展 AgentState 快照承载 Objective：改动入口少，但 AgentState 属于单个调度 agent，无法自然表达 peer Assignment、ObjectiveBudget 和完整变更历史。
- 只保存 Objective 当前快照：读取简单，但预算调整、等待区间、贡献备注和 handoff 原因会在覆盖后丢失，无法可靠复盘或重建 checkpoint。
- 把所有 RunLog 复制进 Objective 存储：查询集中，但形成两份执行真相，容易发生顺序和内容漂移。
- 把 Objective fact 作为特殊 RunLog/StepView kind：物理表更少，但 ObjectiveCreated、预算调整、用户等待和 Objective 终态没有真实 run_id，只能伪造字段，并迫使 Agent 的 Run 查询和存储理解 Objective 语义。

## Consequences

- 新增独立的 `objective_log_entries` 表、ObjectiveLog model、storage contract 与 projection；它不塞入 `run_log_entries` 或现有 AgentStateStorage，也不让 Scheduler 快照成为 Objective 权威状态。
- ObjectiveLog、RunLog 和 outbox 共用同一套存储配置和物理数据库；MVP 仅支持 memory 与 SQLite（同一 sqlite 文件与共享 runtime）。其他 backend 在 wiring/factory 层 fail-closed（ADR 0040），不建立第二套 Objective 数据库服务，也不提交半成品集合实现。
- Objective facts 至少覆盖 Objective 创建与状态变化、ObjectiveUserInput 及外置授权、ObjectiveContribution 及 annotation、Assignment 生命周期、Decision、Artifact 引用、ObjectiveBudget 调整与实际用量、活动窗口、checkpoint 和终态交付。
- 每条 fact 具有稳定 id、objective_id、严格递增 sequence、发生时间、fact kind 和类型化 payload；重复提交使用幂等键拒绝或返回原结果。
- Objective 投影必须能够仅从 ObjectiveLog 重建；任何仅写入内存对象或 AgentState snapshot、却未写入 ObjectiveLog 的 Objective 状态都不具有领域效力。
- RunLog 仍是 Run 输入输出、LLM 调用、tool call 和执行指标的真相源。ObjectiveLog 通过 assignment_id 与 run_id 引用所需 Run，不复制 step 或完整调用内容。
- AgentStateStorage 仍可覆盖保存 Scheduler 查询和唤醒所需的当前快照，但其丢失或重建不能改变 ObjectiveLog 已经确认的领域事实。
- Console 的 Objective 时间线以 ObjectiveLog 为主干，需要执行细节时再按 run_id 查询 RunLog。
