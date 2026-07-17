# Objective 是 Scheduler 之上的深模块

新的 Objective 能力作为独立 `agiwo.objective` 深模块存在，负责跨 Assignment 生命周期、ObjectiveLog、ObjectiveBudget、Outcome、用户交互和可靠派发。它只向外暴露 `ObjectiveService`；`ObjectiveService` 使用 Scheduler 的公开接口执行 Assignment，Scheduler 再按现有路径运行 Agent。ObjectiveLog 重放、outbox、预算 hook、输入装配和 Outcome 归一化都是 `agiwo.objective` 的内部实现，不再各自形成一层公开抽象。

## Status

accepted

## Considered Options

- 把 Objective 逻辑加入 `scheduler/objective_runtime/`：表面接近执行位置，但现有 Scheduler 已负责 agent 状态机、runner、runtime tools 和调度存储，继续加入 ObjectiveLog、预算、模板和用户交互会形成新的大模块。
- 让 Objective 直接调用 Agent.run：路径短，但会绕过 Scheduler 已有的并发、nested agent、stream、状态存储和执行生命周期。
- 把 Objective 仅作为数据模型，控制仍散落在 Console 和 Scheduler：新增目录少，但没有唯一 Objective 不变量 owner，跨存储事务和恢复逻辑会分散。
- 在 ObjectiveService 与 Scheduler 之间增加 `AssignmentRunner`、`AssignmentExecutionControl` 和 `ObjectiveCommandHandler` 等多层协议：纸面依赖更严格，但当前只有一种执行实现，调用者却需要理解更多名称和往返关系，形成浅模块。

## Consequences

- 稳定执行关系只有 `ObjectiveService -> Scheduler -> Agent.run`。调用者不需要了解 Objective 包内部如何拆分文件和类。
- `agiwo/objective/__init__.py` 只暴露 `ObjectiveService` 与稳定公共 DTO；外部调用方不得直接依赖 aggregate、store codec、outbox 或模板渲染实现。
- `ObjectiveService` 负责接收用户命令、保存 Objective、创建 Assignment、消费 AssignmentOutcome、检查 ObjectiveBudget，并依据结构化 Decision 推进 Objective；它只执行机械规则，不进行语义 planning。
- `ObjectiveService` 只调用 Scheduler facade，不读取 `AgentStateStorage`，也不调用 Engine、Runner、TaskGuard 等 Scheduler 内部对象。
- Scheduler 仍然只理解 Agent 的运行和调度，不读取 ObjectiveLog，也不解释 Decision。Scheduler 不导入 `agiwo.objective`。
- Objective 绑定的 Agent 通过现有 `BEFORE_LLM / AFTER_LLM` hook 完成模型调用前后的成本与活动时间检查；不新增一套平行的 Run 控制链。
- handoff 和 verification 额度在 `ObjectiveService` 消费 Outcome、创建下一项 Assignment 之前检查，不塞入 Agent.run 的单次运行限制。
- ObjectiveLog 重放、Objective 投影、outbox 事务和可靠派发都留在 `agiwo.objective` 内部。内部可以按实现复杂度拆分，但这些拆分不成为公共架构概念。
- Console 只调用 `ObjectiveService` 和 Objective HTTP/SSE 接口，不导入 Objective 内部存储或状态机。
- 依赖方向由 import-linter 固化：`objective -> scheduler -> agent`；`scheduler` 和 `agent` 均不得反向依赖 `task`，Console 不得依赖 Objective 内部模块。
- 不预设 `task/runtime/`、`task/projection/`、`task/templates/`、`task/ports.py` 或 `scheduler/objective_adapter.py`。只有出现真实的第二种实现时，才从内部代码中提取新的 seam。
