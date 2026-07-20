# P1-02：建立 ObjectiveLog 与投影

状态：done（2026-07-18）

## 目标

让 Objective 的所有领域变化成为 append-only fact，并能仅凭这些 facts 重建 ObjectiveView。任何只修改内存对象、AgentState snapshot 或 UI 状态而没有写 ObjectiveLog 的变化，都不具有领域效力。

## 对应决定

- ADR 0018：ObjectiveLog 是真相源。
- ADR 0024：每个终态 Assignment 都必须有 Outcome。
- ADR 0030、0031：ObjectiveStatus 与 AssignmentStatus。
- ADR 0037、0039：current_goal 和边界驱动更新。

## 依赖

- P1-01 已完成。

## 范围

包含：fact 模型、sequence、稳定 fact identity、纯 projector、领域校验和查询视图；命令幂等契约由 P1-03 的 command receipt 承担。

不包含：物理存储和事务；P1-03 实现。

## 实施步骤

1. 定义 `ObjectiveLogEntry` 公共头：fact_id、objective_id、严格递增 sequence、kind、occurred_at 和类型化 payload。命令幂等由 P1-03 的持久化 command receipt 负责，不能只在 fact 上放一个无法覆盖 create 的 key。
2. 定义最小完整 fact 集：Objective 创建/状态、ObjectiveUserInput、ObjectiveUserInputExternalized、ContextCapacityExceeded、Contribution/annotation、current_goal revision、Assignment 生命周期、Run 关联、Artifact、Decision、Outcome、Budget 调整/实际用量、活动窗口、等待区间、checkpoint、drain、最终交付和系统故障。
3. 将事实按领域含义命名，不复制 Run step。ObjectiveLog 只保存 assignment_id/run_id/artifact_id 引用。
4. 实现纯 projector：输入按 sequence 排序的 facts，输出当前 ObjectiveView、AssignmentView、BudgetView 和 Timeline node。
5. projector 对 sequence 缺口、重复 sequence、未知引用、非法状态转换和重复 Outcome fast-fail，不猜测修复。
6. 强制 Assignment 终态与唯一 Outcome 同一领域提交；PAUSED 不接受 Outcome，终态不接受第二个 Outcome。
7. 每条 `is_user_provided=true` 的语义输入只追加一条 ObjectiveUserInput fact，保存 input_id、完整 UserMessage 和可选关联；程序不从自然语言中判断或抽取片段。pause/resume/budget 等控制命令不是语义输入，不追加该事实。
8. agent 提问后的用户回复形成新的 ObjectiveUserInput，并可引用问题消息或关联 Outcome；ObjectiveLog 不创建候选、确认或贡献晋升事实。
9. `ObjectiveUserInputExternalized` 只能由引用 input_id 的用户控制命令产生，并必须引用已物化到 `sessions/<session_id>/artifacts/`、source/hash 匹配的文件 Artifact。projector 同时保留原始输入和当前 path/summary 表示；外置不是删除或 Objective 修订。
10. ContextCapacityExceeded 保存模型上下文上限、未外置输入 token 估算和可外置 input ids；它令 Objective 进入 WAITING_USER，但不伪造 AssignmentOutcome。
11. Assignment finalization 的 objective_update、Contribution 和 annotations 以 Outcome 为来源写新 fact；旧版本和原文不覆盖。
12. current_goal projector 始终同时保留不可变 ObjectiveUserInput 与最新可变分析，并保存 revision 链与来源引用。
13. ObjectiveDelivered 明确引用 final outcome、普通文本主 report、其他文件 artifact ids 和时间；不能从最后 Run response 推断，也不能把主 report 建成 Artifact。
14. 为 Timeline 提供 sequence 分页所需的稳定 view，不提前引入 HTTP/SSE 类型。

## 主要改动位置

- `agiwo/objective/models.py`
- `agiwo/objective/log.py`
- `agiwo/objective/projection.py`
- `agiwo/objective/errors.py`
- `tests/objective/test_log.py`
- `tests/objective/test_projection.py`

## 测试计划

- 从空日志依次重建 CREATED、RUNNING、WAITING、PAUSED、COMPLETED 路径。
- 每个非法状态转换和终态重开均失败。
- 同一 command receipt 重放不创建第二条语义变化；fact 自身的稳定 fact_id 只用于重复 append 防护。
- Assignment 终态无 Outcome、两个 Outcome、PAUSED 带 Outcome 均失败。
- ObjectiveUserInput/Contribution/current_goal revision 的权威边界与历史保留。
- ContextCapacityExceeded -> 外置授权 -> 容量通过的投影路径；原始 ObjectiveUserInput 始终可查。
- ObjectiveDelivered 只能引用已存在、可交付的 Outcome 与文件 Artifact；主 report 是普通文本。
- 投影结果与增量应用结果一致。

## 完成标准

- 删除所有可变快照后，仅用 ObjectiveLog 能重建同一 ObjectiveView。
- ObjectiveLog 不包含 Run message、tool call 或 token delta 副本。
- projector 不读取 Scheduler store、RunLog 或 Console 模型。
- 所有状态和唯一性不变量具有测试。
- lint 与 objective 单元测试通过。

## 风险与回退

fact 集过早膨胀会使 codec 和存储复杂化。只创建 ADR 明确需要的一等事实；UI 派生字段保持投影，不另写 fact。回退时可以删除 Objective 包，不改现有 RunLog。
