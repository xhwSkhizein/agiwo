# Session 包含依次执行的 Objective

面向用户的 Session 是可长期存在、可以多轮交互和分叉的对话容器；Objective 是其中一个具有明确目标、预算和终态的问题解决过程。一个 Session 可以依次保存多个 Objective，但同时最多只有一个非终态 Objective。Objective 运行或等待用户时收到的新输入交给当前 Objective；Objective 一旦进入 `COMPLETED / FAILED` 就不再打开，之后的用户输入创建下一个 Objective。

## Status

accepted（同一 Session 仍可顺序存在多个 Objective；但 Objective 按需升级，不是每条消息默认创建——见 ADR-0047）

## Considered Options

- Session 与 Objective 一对一：模型简单，但每完成一个目标都要创建新 Session，会把对话列表变成任务列表，也无法自然保留长期交互关系。
- 一个 Session 同时运行多个 Objective：并发能力更强，但普通聊天输入无法无歧义地判断应当 steer、补充或启动哪一个 Objective，必须增加显式 Objective 选择界面和路由规则。
- 不建立 Session 与 Objective 的关系：Objective API 可以独立运行，但 Console、渠道消息、RunLog 和用户回复无法形成统一归属。

## Consequences

- 用户面对的顶层交互对象仍是 Session；默认路径是 Session 对话 Turn。Objective 仅在升级后作为可查看状态、进度、预算和结果的工作项呈现。
- 一个 Session 可以关联任意数量的终态 Objective，但非终态 Objective 合计最多一个。
- “最多一个非终态 Objective”必须由 ObjectiveStore 的 Session 活动占用记录原子保证（见既有约束）；无活动 Objective 时新消息默认不创建 Objective（ADR 0047）。
- 终态后的新消息若仍无升级信号，继续作为普通 Turn；只有再次出现升级条件才创建下一个 Objective。
- Objective 的 `COMPLETED / FAILED` facts 不会被恢复、删除或以追加事件抵消。只有 `BUDGET_PAUSED / USER_PAUSED` 等非终态状态能够恢复原 Assignment 和 Run。
- 用户希望同时处理另一个独立目标时，应新建 Session。一个 Objective 同时最多有一个非终态 Assignment，但该 Assignment 内仍可以通过 Parallel、Pipeline 或其他委派方式并行执行多个 child Run。
- 每个 Objective 保存一个不透明的 `session_id` 关联。`agiwo.objective` 不导入 Console 的 Session 模型；Console 或其他入口负责保证该引用存在。
- Objective 管理的 Assignment 复用 Session 的 persistent agent/state identity；每个 Assignment 使用新的 `assignment_id` 和 `run_id` 表达独立责任与执行，不通过更换 `agent_id` 隔离。
- 同一 Session 中的历史是不同 Assignment 的候选上下文；系统依据 Objective 当前全局目标与本次 Assignment 触发信息的相关性，决定原样保留、压缩或排除哪些历史，并显式加入 ObjectiveView。该判断不以 Assignment 边界作默认隔离。
- 当前 `AgentState.task: UserInput` 不是新的领域 Objective。实现 Objective 时应将该字段改为表达实际含义的名称，避免两套概念同名。
- 现有直接使用 Scheduler persistent root 的 SDK 能力不因此删除；上述规则只约束 ObjectiveService 管理的用户任务路径。
- Fork Session 创建新的 Session 与 persistent root identity，但不复制或移动源 Session 的活动 Objective；源 Objective 继续原有生命周期，新 Session 初始没有 Objective。
- 用户提供的 `fork_context_summary` 在新 Session 第一个 Objective 的 root Run 中渲染为临时 `<system-notice>`，不伪装成用户消息；使用一次后仍作为 Session fork provenance 保留。
- Fork 不复制 ObjectiveBudget、Assignment、checkpoint、Outcome 或 RunLog，也不能让两个 Session 恢复同一 Run。需要引用旧 Artifact 时由用户在 summary 或后续输入中明确指出。
- 当前 Console fork 只保存 `source_session_id / fork_context_summary` 而没有把 summary 注入运行上下文；Objective 集成必须补齐该消费路径。
- 普通用户对 Session 的删除请求执行归档而非物理删除。若存在活动 Objective，先按用户暂停语义保存 checkpoint；恢复 Session 只恢复可见性，不自动继续模型执行。
