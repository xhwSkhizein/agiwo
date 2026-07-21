# 每个结束的 Assignment 都必须产生 Outcome

Assignment 只要进入终态，就必须产生唯一 AssignmentOutcome，无论它是正常完成、被新用户输入中断、重试耗尽、结果未知或遭遇不可重试故障。Outcome 是执行信息离开 Run、进入 Objective 后续协作的正式边界；没有 Outcome 就结束执行，等同于丢弃已经通过模型和工具调用获得的信息。checkpoint 和 Budget Pause 不结束 Assignment，因此在暂停时不提前生成 Outcome。

## Status

superseded by ADR-0046

> Outcome 挂在 Objective 管理的 root Run 上；普通 Session Turn 不强制写 Objective Outcome。

## Considered Options

- 只为成功 Assignment 生成 Outcome：模型简单，但中断和故障路径会丢失已完成工作，后续 agent 只能重查 RunLog 猜测现状。
- 把完整 RunLog 当作 Outcome：不会丢信息，但把执行轨迹噪声直接推给后续 agent，也混淆存储真相与交接摘要。
- 中断时使用普通 summary：实现可复用已有终止总结，但容易复述大量轨迹，无法突出哪些信息仍适用于最新用户输入。

## Consequences

- ObjectiveLog 不能接受没有 AssignmentOutcome 的 Assignment 终态 fact；outbox 也不能在 Outcome 提交前完成。
- Outcome 必须记录 assignment_id、关联 run_id、终态状态、结束原因、普通文本 report、可选文件 Artifact 引用、ObjectiveContribution、可选 Decision 以及生成 provenance。
- 正常语义完成的 Outcome 包含 agent Decision；故障策略可以产生机械 Decision。运行中用户输入不结束 Assignment，因此不产生专用 steering Outcome。
- Outcome report 保持精简且为普通文本；原始消息、模型调用、tool result 与调试数据留在 RunLog；Outcome 通过 run_id 和 artifact ids 保留可追溯性。report 本身不是 Artifact。
- 模型无法完成收口时，执行系统从 committed RunLog facts 和已有 Artifact 引用构造最小 Outcome（普通文本 report），并明确标记 system provenance 与缺失的语义提炼。
- 非 Assignment 触发的 Run 继续使用其原有 RunOutput/调用方返回契约，不伪造 Objective AssignmentOutcome。
- Assignment 内受委派的 child Run 也只向 root Run 返回 RunOutput/child result；唯一 AssignmentOutcome 由承担责任的 root Run 汇总后提交。
- `INTERRUPTED` Outcome 必须列出中断时 root Run 的 `RunPlan` 中仍未完成（`pending / active`）的项，并把需要后继责任继续处理的项目标记为 `carry_forward`；不能因中断而静默丢弃（ADR 0038）。
- PAUSED Assignment 不产生 Outcome；计划状态随 RunLog 事实与 checkpoint 保留，恢复同一 Run 后继续维护。
