# P1-01：建立 Objective 领域模型

状态：planned

## 目标

建立 Objective、Assignment、Decision、Outcome、Artifact、ObjectiveUserInput、Contribution 和 Budget 的纯领域语言，明确哪些对象属于全局目标、局部责任或实际执行。该任务只定义结构和确定性不变量，不调用 Scheduler，也不让模型参与状态判断。

## 对应决定

- ADR 0002、0003：Objective、Assignment、委派和接力的边界。
- ADR 0005、0007、0008：Decision、ObjectiveUserInput、Contribution 和 Handoff target。
- ADR 0025：Assignment kind 与默认 AgentConfig 的关系。
- ADR 0030、0031、0035、0037：状态、Session 关系和 current_goal。

## 依赖

- P0-03 已完成，领域输入可以引用带来源的 UserMessage。

## 当前源码现状

- 仓库没有 `agiwo/objective/`。
- Scheduler 的 `AgentState.task` 是调度输入，不是 Objective。
- RunStatus 只有 STARTING/RUNNING/COMPLETED/FAILED/CANCELLED，尚不能表达 Objective/Assignment 状态。
- Console Session 可以跨多轮存在，但没有 Objective 关联或“至多一个活动 Objective”的约束。

## 范围

包含：类型、枚举、值对象、字段校验、合法状态转换表和领域错误。

不包含：日志存储、投影、模型调用、Scheduler 派发和 Console API。

## 实施步骤

1. 新建 `agiwo/objective/` 内部包与 `models.py`，定义稳定 ID、时间和 revision 类型；纯 DTO 不引用 Console Session 模型，只保存不透明 session_id。
2. 定义 `ObjectiveStatus` 八态：CREATED、RUNNING、DRAINING、WAITING_USER、BUDGET_PAUSED、USER_PAUSED、COMPLETED、FAILED。
3. 定义 `AssignmentStatus` 六态：CREATED、RUNNING、PAUSED、COMPLETED、INTERRUPTED、FAILED；定义 kind：intake、work、verification。
4. 定义 `HandoffDecision`，target 封闭为 agent/verifier/user；user 必须显式携带 expects_reply，其他 target 禁止具名 agent、config、pattern 或 executor 字段。
5. 定义 Artifact 与 ArtifactRef（ADR 0045）。Artifact 只索引 `{agent_workspace}/sessions/<session_id>/artifacts/` 下的独立文件（图片、PDF、日志、大文本、外置用户输入），字段含 artifact_id、相对 path、summary、可选小体积 inline content、可选 source_input_id/content_hash；超过大小阈值不保存 inline content。普通文本 report、故障说明和聊天回复不是 Artifact，也不含路由字段。
6. 定义 AssignmentOutcome：唯一 assignment_id、root run_id、终态、reason、普通文本 report、可选 ArtifactRefs、新建 Contributions、contribution annotations、可选 Decision、objective_update、carry_forward 和 provenance。
7. 定义 `ObjectiveUserInput`、输入外置授权与 `ObjectiveContribution`。ObjectiveUserInput 保存稳定 input_id、完整真实 UserMessage，以及可选关联字段。Contribution：`contribution_id`（系统分配）、不可变 `content`、可选 `summary`、annotations 列表。annotation：`{contribution_id, annotation, deactivate?, from, time}`，`from`/`time` 由系统填充。`objective_update` 字段见 ADR 0006。外置授权把原文物化到 artifacts 目录；pause/resume/budget/externalize 不形成 ObjectiveUserInput。
8. 定义 current_goal 的不可变基础与带 revision 的可变分析；可变字段只允许意图、范围、成功标准、假设和来源，不允许 RunPlan、状态、预算或 handoff。
9. 定义 ObjectiveBudget 四维：handoffs、verification_attempts、llm_cost_usd、active_seconds；每维只有 limit/used，不定义 reserved。模型只提供读取视图，没有修改配额的方法。
10. 定义 Session/Objective/Assignment 基数不变量：一个 Session 最多一个非终态 Objective；一个 Objective 最多一个非终态 Assignment；并行只存在于 Assignment 的 root/child Run 树中。
11. 定义合法状态转换与不可重开规则。COMPLETED/FAILED Objective 和 COMPLETED/INTERRUPTED/FAILED Assignment 均不可恢复；只有 PAUSED 类状态可恢复。
12. 公开的领域错误使用类型化 code 和必要字段，不用错误字符串驱动控制流。

## 主要改动位置

- `agiwo/objective/`（新建，暂不公开内部模型）
- `tests/objective/test_models.py`（新建）
- `tests/objective/test_invariants.py`（新建）

## 测试计划

- 每个枚举的合法/非法转换表测试。
- 一个 Session 两个活动 Objective、一个 Objective 两个活动 Assignment 必须失败。
- terminal reopen 必须失败；PAUSED resume 合法。
- HandoffDecision 三种 target 的字段组合校验。
- Outcome 缺少 root run、终态或必要 report 时失败。
- ObjectiveUserInput 必须引用完整真实 UserMessage；系统控制输入无法创建该事实，agent 贡献也无法晋升为该事实。
- 输入外置授权必须引用已存在 input_id，并对应 artifacts 目录中的稳定文件与匹配 hash；模型侧以 path+summary 表示，按需读文件；授权不删除 ObjectiveUserInput。
- current_goal 可变分析不能覆盖用户事实或携带计划/预算字段。
- 普通文本 report 不得建模为 Artifact；Artifact 不得携带 Decision/路由字段。

## 完成标准

- Objective 类型不导入 Scheduler 内部或 Console 模型。
- Scheduler 的 `TaskGuard/TaskLimits` 没有被重命名或包装成 ObjectiveBudget。
- `AgentState.task` 不被当作 Objective 字段使用；其改名在后续 Scheduler 接入任务处理。
- 全部领域不变量都有纯单元测试。
- `uv run python scripts/lint.py ci` 通过。

## 风险与回退

最大风险是把实现细节提前固化成大量公开类型。除 ObjectiveService 所需稳定 DTO 外，模型先保持包内可见；回退时删除整个新包即可，不影响现有 Agent/Scheduler 路径。
