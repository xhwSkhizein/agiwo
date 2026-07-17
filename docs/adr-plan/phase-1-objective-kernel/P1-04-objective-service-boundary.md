# P1-04：建立 ObjectiveService 深模块边界

状态：planned

## 目标

把跨 Assignment 生命周期的确定性规则收口到一个对外门面 `ObjectiveService`。调用者只提交用户命令和读取稳定 DTO，不需要知道 ObjectiveLog replay、outbox、投影或模板实现；Scheduler 仍是下层机械执行者，不能反向读取 Objective。

## 对应决定

- ADR 0001：语义决定由 agent 形成，执行控制由系统实施。
- ADR 0007：ObjectiveUserInput 外置必须由用户明确授权。
- ADR 0034：Objective 是 Scheduler 之上的深模块。
- ADR 0035：一个 Session 依次包含多个 Objective。
- ADR 0044：ObjectiveService 拥有 Objective 控制与 Store 写入。

## 依赖

- P1-02、P1-03 已完成。

## 范围

包含：public facade、稳定请求/响应 DTO、create/load/list/command 骨架、事务调用和导入护栏。

不包含：模板渲染、实际 Scheduler 派发、Assignment finalization 和 HTTP API；后续任务通过这个 facade 增量实现。

## 实施步骤

1. `agiwo/objective/__init__.py` 只导出 `ObjectiveService` 与调用者真正需要的 DTO；aggregate、projector、store codec 和 outbox record 保持内部。
2. ObjectiveService 构造时接收 ObjectiveStore，以及后续可注入的 Scheduler facade/默认 agent provider；不要创建 `ObjectiveEngine`、`CommandHandler`、`AssignmentRunner` 等平行 public 层。
3. 提供稳定用例方法骨架：create objective、get view、list by session、submit user input、externalize user input、pause、resume、adjust budget；未接通执行的命令明确返回不可用状态，不做隐藏 side effect。
4. create 校验预算有限有效、输入为真实用户来源，并通过 P1-03 的单一 store transaction 原子取得 Session slot、写 command receipt、ObjectiveCreated、完整 ObjectiveUserInput、初始领域事实和 outbox；不能先查活动 Objective 再写。
5. 所有命令使用明确 scope/idempotency_key/canonical request hash。相同 key/hash 重放持久化 response，不追加事实；相同 key/不同 hash 返回 `IdempotencyConflict`。
6. Service 只消费结构化 Decision/Outcome，不解析自然语言或 Artifact 推断下一步。每条真实用户语义消息只保存为一条完整 ObjectiveUserInput；不存在候选确认、贡献晋升或第二份用户侧模型。
7. externalize command 必须引用既有 input_id、使用独立 command scope，并在一个事务中写 command receipt、source/hash Artifact 元数据与 ObjectiveUserInputExternalized；相同命令重放不得复制 Artifact。Artifact 内容始终从原始 ObjectiveUserInput 解析。
8. “同一 Session 最多一个活动 Objective”由 Store slot 唯一约束保证，Service 只表达命令意图；终态不重开的状态前置条件仍在同一事务校验。
9. 为 Objective 包增加 import-linter：`agiwo.scheduler`、`agiwo.agent` 不得依赖 `agiwo.objective`；Objective 可以依赖 Scheduler public facade；Console 只能依赖 Objective public facade。
10. 更新 `AGENTS.md` 的目录职责，保持包级描述，不列逐文件索引。
11. 写 facade contract tests，证明调用者无需访问 store/projection 内部类型。

## 主要改动位置

- `agiwo/objective/__init__.py`
- `agiwo/objective/service.py`
- `agiwo/objective/models.py`
- `lint/importlinter_agiwo.ini`
- `tests/agent/test_definition_contracts.py` 或新的 architecture contract test
- `tests/objective/test_service.py`
- `AGENTS.md`

## 测试计划

- create 的预算、真实用户输入、Session 活动 Objective 前置条件。
- command receipt 的同 key/同 hash replay、同 key/不同 hash conflict，以及 create response 跨重启重放。
- 两个不同 objective_id 的并发 create 只允许一个取得同一 Session slot。
- terminal Objective 收到普通输入时要求新建 Objective，不重开旧对象。
- externalize command 的用户权限、input/hash 校验、幂等 Artifact/fact，以及终态拒绝。
- Service 不从 Artifact/free text 推断 Decision。
- import-linter 正向和禁止反向依赖测试。
- public import surface snapshot。

## 完成标准

- 外部用例只需导入 `from agiwo.objective import ...`。
- `scheduler -> objective` 和 `agent -> objective` 导入被机器护栏拒绝。
- Service 没有直接读取 AgentStateStorage、Scheduler engine/runner 或 RunLog 内部表。
- Objective 包内部拆分没有成为调用者必须理解的 public 架构层。
- objective service tests 与 lint 通过。

## 风险与回退

最常见风险是为了“边界清楚”过度创建 port/service/facade。当前只有一种 Scheduler 执行实现，先使用一个 ObjectiveService 和内部函数；只有出现真实第二种实现时再提取 seam。
