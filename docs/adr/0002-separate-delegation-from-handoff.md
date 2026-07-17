# 区分委派关系与接力关系

Agiwo 将委派与接力建模为两种不同的协作关系。现有 `AgentState.parent_id` 继续表达委派所有权：parent 保留责任并等待 child 的结果；接力则使用独立的前后继关系，前一执行者交出控制权后完成，后一执行者直接承接同一个 Objective，不成为前一执行者的 child。Objective 同时最多只有一个非终态 Assignment；需要并行时，由该 Assignment 的 root Run 委派多个 child Run，而不是创建多个并行 root Assignment。

## Status

accepted

## Considered Options

- 让接力复用 parent-child 树：可以少建一套关系，但会让已经完成责任的前一执行者继续承担等待和汇总，并使长接力链错误地消耗派生深度。
- 所有协作都改成接力：模型更单一，但会丢失 Parallel、Fan-out 等场景中 parent 等待并汇总 child 结果的必要语义。

## Consequences

- 委派继续沿用现有 child 生命周期、等待和结果回传机制。
- Assignment 由一个承担责任的 root Run 和零到多个受委派 child Run 构成；child 使用独立 agent/run identity，但继承同一 objective_id、assignment_id 和 ObjectiveBudget 约束。
- child Run 只产生现有 RunOutput/child result 并回传 root，不单独生成 AssignmentOutcome 或 HandoffDecision；root 汇总全部委派结果后完成 Assignment 收口。
- ObjectiveService 只有在当前 Assignment 已经结束并提交唯一 AssignmentOutcome 后，才能接受 handoff 并创建下一 Assignment。Objective 级责任所有权因此保持串行。
- Parallel、Pipeline、Fan-out 或其他并发 pattern 继续在 Assignment 内运行；本约束限制的是最终责任主体数量，不限制实际并发执行。
- 接力不能通过增加 `AgentState.parent_id` 深度来表达，必须拥有独立的因果关系。
- 全局循环边界需要按完整任务计算，不能只检查单棵委派树的深度和唤醒次数。
