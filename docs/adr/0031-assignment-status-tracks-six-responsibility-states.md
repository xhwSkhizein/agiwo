# AssignmentStatus 使用六个责任状态

AssignmentStatus 固定为 `CREATED`、`RUNNING`、`PAUSED`、`COMPLETED`、`INTERRUPTED`、`FAILED`。它描述本次局部职责是否尚未开始、正在承担、暂时冻结或已经结束，不复制 outbox、AgentState 或 Run 内部等待细节。Assignment 进入任何终态都必须同时提交唯一 AssignmentOutcome。

## Status

accepted

## Considered Options

- 复用 Scheduler AgentStateStatus：可以少一套枚举，但 PENDING、QUEUED、IDLE、WAITING 表达的是调度实例状态，不是 Assignment 责任生命周期。
- 只使用 active/completed 两态：模型简单，但无法区分 checkpoint 可恢复、用户语义中断和系统不变量失败。
- 为 outbox pending/claimed 增加 Assignment 状态：观察方便，但会把可靠投递实现泄漏进领域模型。

## Consequences

- `CREATED` 表示 AssignmentCreated 已提交但尚无 RunStarted；outbox pending、claimed 和 retry 都不改变该状态。
- dispatcher 观察到第一个关联 RunStarted 后向 ObjectiveLog 追加 `AssignmentExecutionStarted`，该 Objective fact 使 Assignment 进入 `RUNNING`；agent 内部等待 tool、delegation 或 pattern 汇合时仍保持 RUNNING。
- Objective 进入 BUDGET_PAUSED 或 USER_PAUSED 且该 Assignment 保存 checkpoint 后，Assignment 进入 `PAUSED`；恢复同一 Run 时回到 RUNNING。
- `COMPLETED` 表示责任正常结束并提交 Outcome，包括正常 handoff、completion proposal 和 verifier rejection；进入前 root Run 的 `RunPlan` 中不能存在 `pending / active` 项（ADR 0038）。
- `INTERRUPTED` 表示责任因 retry exhaustion、outcome_unknown 或阻断性 execution fault 提前结束；它仍必须提交说明已有进展、中断原因和 `carry_forward` 计划项的 Outcome。运行中用户输入与可恢复中断（pause）不把 Assignment 标为 INTERRUPTED。
- `FAILED` 只表示 AssignmentOutcome 或 Assignment 领域事实无法可靠形成的不变量故障，不用于普通模型/tool 失败。
- `COMPLETED`、`INTERRUPTED`、`FAILED` 都是终态，不能重新打开；需要继续工作时创建新 Assignment。只有 PAUSED 可以恢复。
