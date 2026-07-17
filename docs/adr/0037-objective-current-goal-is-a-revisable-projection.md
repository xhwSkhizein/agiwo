# Objective 当前目标是可修订投影

Objective 既要给后续 Agent 一个简洁、持续演进的全局目标，又不能让某次模型分析成为改写用户意图的权威操作。因此 `current_goal` 分成不可变基础与可变分析：程序在每次真实用户语义输入时原样追加 ObjectiveUserInput；Assignment root Agent 只在系统必然发起的收尾调用中提出结构化分析修订。两者共同投影“现在需要完成什么”，但可变内容永远不能覆盖不可变事实。

## Status

accepted

## Considered Options

- 只把全部事实直接交给每个 Agent：最忠实，但链路增长后信息密度下降，Agent 需要反复重建全局意图。
- 维护一段可被 Agent 覆盖的目标文本：输入最简洁，但一次错误总结就可能永久丢失用户原始输入或改变验收边界。
- 每次目标变化都要求用户确认：权威性最强，但会把普通语义整理变成人工审批，破坏自动接力。

## Consequences

- `current_goal` 的不可变基础由按 input id 追加的全部 ObjectiveUserInput 构成。程序不对自然语言分句抽取片段；Agent 无权新增、改写、停用或删除这些内容，agent 贡献也不能晋升为用户事实。
- `current_goal` 的可变部分包含带版本的意图、范围、成功标准理解、假设及其依据，不包含带顺序和完成状态的执行计划；当前 Run 的阶段计划只存在于 `RunPlan`。
- Run 内不注入修改 Objective 的 runtime tool。root Agent 在 Assignment 收尾调用中输出可选、精简且结构化的 `objective_update`（见 ADR 0006 线格式）；没有新的全局理解时该字段为 `null`。
- `objective_update` 允许字段仅为 `expected_revision`、`intent`、`scope`、`success_criteria`、`assumptions`、`sources`；`sources` 必须引用已存在的 user_input / contribution / artifact / outcome / objective_fact id。
- ObjectiveService 对允许字段、来源引用和不可变边界做确定性校验，将合法修订作为新版本写入 ObjectiveLog；`expected_revision` 不匹配时拒绝该次 update 并记录原因，但不因此丢弃合法 Outcome/Decision。Scheduler 与 agent runtime 不理解 Objective 修订语义。
- 最新用户语义输入必须先原样写入 ObjectiveLog，完整成为新的 ObjectiveUserInput。程序不判断它是否改变目标；Agent 对其含义的理解只能进入可变分析，不能伪装成用户原始事实。
- handoff、verification 或普通工作 Assignment 都可以在新证据实质改变目标理解时，于收尾调用中提出 Objective 修订，但必须引用对应用户输入、Contribution、Outcome、Artifact 或其他 Objective fact。
- ObjectiveService 不判断可变分析在语义上是否正确。后续 root Agent可以结合不可变事实继续追加修订；旧版本完整保留，不能覆盖删除。
- 后续用户输入改变先前含义时，旧 ObjectiveUserInput 及关联仍按原始顺序保留，不做物理删除；`current_goal` 的可变分析只表达当前理解。
- ObjectiveView 同时携带 `current_goal` 和支撑它的权威事实。上下文装配不能只传目标投影而隐藏其依据。
- ObjectiveContribution 与 `current_goal` 可变分析分开表达；前者记录可能帮助后续执行的非权威发现，后者维护当前有效的目标理解。
- Objective 同时最多有一个非终态 Assignment；并行 child Run 只把发现返回 root。pause、cancel 或预算暂停不结束 Assignment，因此不提交 Objective 修订；恢复同一 Run 后继续工作。正常或中断收尾产生唯一 Outcome 时，才允许提交本次 Assignment 的 Objective 修订。
