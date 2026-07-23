# 分布式语义决策，集中式执行控制

Agiwo 将任务含义相关的判断交给执行中的 agent，并保留确定性的集中执行控制。Agent 以结构化指令表达下一步或结束提议；执行系统只校验指令、记录事实并实施物理边界，不进行语义匹配或完成度判断。Objective 级控制由 ObjectiveService 拥有，Run/agent 级机械调度由 Scheduler 拥有，具体分工由 ADR 0044 取代本 ADR 早期的单一 Scheduler owner 表述。

## Status

accepted

## Considered Options

- 由中心编排器理解任务并规划完整 workflow：容易获得全局控制，但会把语义判断重新集中到单点，与运行时动态展开的目标冲突。
- 连执行控制也去中心化：自治程度更高，但会引入重复消费、预算竞态、终止竞态和故障恢复等分布式一致性问题，也会绕开 Agiwo 现有的 `Scheduler` 边界。

## Consequences

- 所有影响任务走向的语义判断都必须留下结构化决定，不能隐藏在 ObjectiveService 或 Scheduler 的条件分支中。
- ObjectiveService 统一拥有 Objective 全局硬限制及状态迁移；Scheduler 与 Agent 在实际执行点消费 ObjectiveService 签发的准入租约，任何一层都不能绕过。
- “无中心编排器”不再作为架构术语；准确表述是“分布式语义决策、集中式执行控制”。
- ObjectiveService 与 Scheduler 的分层 owner 关系见 ADR 0044。
