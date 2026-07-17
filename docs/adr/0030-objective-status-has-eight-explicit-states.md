# ObjectiveStatus 使用八个显式状态

ObjectiveStatus 固定为 `CREATED`、`RUNNING`、`DRAINING`、`WAITING_USER`、`BUDGET_PAUSED`、`USER_PAUSED`、`COMPLETED`、`FAILED`。状态由 ObjectiveLog facts 投影，不由 AgentState 或当前 Run 状态反推。普通 agent、模型和工具故障通过 AssignmentOutcome、retry、handoff 或用户边界处理；只有 ObjectiveLog、存储或领域不变量已经无法可靠恢复时，Objective 才进入 FAILED。

## Status

accepted

## Considered Options

- 用一个通用 PAUSED 表示所有等待：状态较少，但无法区分是否需要用户输入、增加预算或仅确认继续。
- 增加 CANCELLED 作为用户停止终态：符合常见 API 命名，但本系统的用户停止是 checkpoint 可恢复 pause，不是终结。
- 普通 Run 失败时把 Objective 标为 FAILED：映射直接，但会绕过已经设计的 fresh handoff 和用户处理路径。

## Consequences

- `CREATED` 表示 ObjectiveCreated 已提交但初始 Assignment 尚未进入实际执行；dispatcher 观察到 committed RunStarted 并向 ObjectiveLog 幂等追加 `AssignmentExecutionStarted` 后进入 RUNNING。outbox 的 dispatched 状态本身不具有领域效力。
- `RUNNING` 表示允许按当前 Objective 事实创建和推进工作；它最多包含一个非终态 Assignment，该 Assignment 可以包含一个 root Run 和多个并行 child Run。
- `DRAINING` 是预算触顶或用户 pause/停止触发后的机械 barrier 状态，禁止创建新工作；运行中用户输入不进入本状态（ADR 0023）。
- `WAITING_USER` 表示系统需要用户提供新语义输入或完成明确控制动作。常见来源是 `target=user, expects_reply=true`；当 `ContextCapacityExceeded` 发生时，则等待用户授权指定 ObjectiveUserInput 外置为文件 Artifact（上下文仅 path 与 summary）。满足等待条件后才创建新 Assignment 并回到 RUNNING。
- `BUDGET_PAUSED` 保存原 Assignment/Run checkpoint；提高必要配额或确认 active-time 新窗口后恢复原 Runs 并回到 RUNNING。
- `USER_PAUSED` 保存原 Assignment/Run checkpoint；用户 resume 后无需新 Assignment，直接恢复原 Runs 并回到 RUNNING。
- `COMPLETED` 表示 Objective 已经过适用验收并完成最终交付；它是正常不可恢复终态。
- `FAILED` 只用于不可恢复的 ObjectiveLog 损坏、持久化失败或领域不变量破坏；进入前应尽最大可能保存结构化系统故障，且不能被普通 agent fault 触发。
- `COMPLETED / FAILED` 都不能重新打开。终态后同一 Session 收到的用户输入创建新 Objective；只有非终态暂停状态能够恢复既有 Objective、Assignment 和 Run。
- 每次状态变化都必须有独立 Objective fact、来源和原因；直接覆盖 Objective.status 不具有领域效力。
