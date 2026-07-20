# P3-06：实现重启恢复与 Outbox 对账

状态：done

## 目标

在进程崩溃后，根据 ObjectiveLog、outbox、RunLog 和 Scheduler 可重建状态判断每个 Assignment 应继续派发、连接既有执行、恢复 checkpoint、补写 Outcome，还是进入结构化故障处理；不得凭缺失信息擅自重跑。

## 对应决定

- ADR 0019：outbox lease 与稳定 ID 重放。
- ADR 0033：RunLog + checkpoint 恢复。
- ADR 0040：ObjectiveStore 与 RunLog 同物理配置但独立事务。

## 依赖

- P2-03、P3-04、P3-05 已完成。

## 范围

包含：startup reconciler、outbox/RunLog/runtime 状态矩阵、孤儿 in-flight 动作分类、checkpoint resume、Outcome 补写、Session slot/command receipt 对账和故障注入测试。

不包含：fault 的语义分流；P4 使用本任务提供的结构化恢复结果。

## 实施步骤

1. 在 ObjectiveService 启动时扫描 pending/claimed/dispatched 未完成 outbox、Session slots、未完成 command receipts，以及 RunLog 已记录但尚未进入 ObjectiveBudget 的实际 LLM 成本，按稳定 ID 建立对账队列。不扫描或恢复 ObjectiveActionLease（该概念已废弃）。
2. 对每个 record 只通过 public query 获取 RunView、RunLog action facts、checkpoint、runtime presence 和 ObjectiveView，不读取其他模块内部表。
3. 第一层 Run 矩阵固定为：无 RunStarted -> 重新派发；RUNNING + runtime 存在 -> 连接/等待；RUNNING + runtime 不存在 -> 进入孤儿动作矩阵；PAUSED + checkpoint -> 按 Objective 状态等待或两阶段恢复；终态 + 无 Outcome -> 幂等补写；Outcome 已存在 -> 完成 outbox。
4. RUNNING + runtime 不存在且没有未完成 logical call / 未提交 tool 副作用证据时，从最后 committed safe boundary 为同 run_id 构造 recovery cursor，重建 runtime；不能新建 Run。
5. 孤儿动作按 kind 与 committed 证据分类：
   - LLM 没有任何 committed response/cost 证据：记零成本并结束该 attempt；是否创建新 attempt 由 RetryCoordinator、RunLimitPolicy 和当前 `used + ceiling` 检查共同决定。
   - LLM 已有 response/usage/cost fact：按实际请求 token 与已接收输出 token 补写唯一成本，并按 attempt identity 幂等追加 Objective 用量；不得再次调用同一 attempt。
   - 可能产生外部业务副作用的 tool：adapter 调用前必须先提交 `ExternalEffectMayHaveStarted`。没有该 fact 可以按未开始处理；已有该 fact但没有 committed result 时，只有明确幂等证明才允许重试，否则转 outcome_unknown/user。
   - tool、spawn 或 dispatch 已有 committed result/stable identity：不重复副作用，从结果后的 phase 继续或幂等对账。
6. “先成功派发、后未标 dispatched”通过 run_id 识别已有执行，不再启动；若 `AssignmentExecutionStarted` 缺失，则与 outbox dispatched 状态在同一 ObjectiveStore 事务幂等补写。
7. Outcome 补写只消费已提交 finalization/system fault facts；模型结果缺失时使用带 provenance 的最小 Outcome，不从自由文本猜 Decision。
8. assignment_id/run_id/outcome/fact stable ids 保证补写最多一次；命令重放从 command receipt 返回原 response，不重新执行 handler。
9. 启动时验证每个非终态 Objective 都有且只有一个匹配 Session slot；缺失时从 ObjectiveLog 重建，冲突时停止相关 Session 并报告不变量故障，不能猜选一个。
10. RunLog 状态缺口、重复终态、checkpoint hash 不符或无法确认副作用时，保留 outbox claim 并记录 RecoveryFault；不能自动重跑。
11. ObjectiveLog/Store 不可恢复损坏才允许 Objective FAILED；普通 runtime 丢失优先按上述矩阵重建或暂停给用户。
12. dispatcher/reconciler shutdown 等待当前事务结束并释放 outbox claim lease；不维护 ObjectiveActionLease。
13. 为每个“事务提交后、下一步前”边界加入故障注入测试：receipt/slot、outbox claim、RunStarted、AssignmentExecutionStarted、LLM preflight passed、LLM response observed、actual cost recorded/applied、external effect may have started、result committed、RunPaused、resume prepared/committed/released、RunFinished、Outcome、outbox complete。

## 主要改动位置

- `agiwo/objective/dispatch.py`
- `agiwo/objective/recovery.py`
- `agiwo/objective/service.py`
- `agiwo/objective/store/`
- Scheduler facade **P2-02 已定义的**查询 API（不得另扩平行面）
- RunLog query/service
- `tests/objective/test_recovery.py`

## 测试计划

- 上述每个 crash boundary 的 SQLite 关闭重开测试。
- outbox claim lease owner 崩溃和另一实例接管。
- terminal Run 补 Outcome 与重复补写幂等。
- PAUSED Run 在 BUDGET/USER pause 下不自动恢复；显式 resume 后继续。
- running runtime 存在时不重复启动。
- RunLog=RUNNING/runtime 不存在且无开放动作时恢复同 run_id。
- LLM 无响应零成本、响应已记录但 Objective 用量缺失、tool 外部副作用未知、稳定 result 已提交四类孤儿动作。
- RunStarted 后缺 AssignmentExecutionStarted 的幂等补写。
- command receipt response 和 Session slot 的重启对账。
- 状态无法确认时保留 outbox 并生成 fault，不猜测。

## 完成标准

- 任一已提交 Assignment 最终都可证明处于待派发、活动、暂停或有 Outcome 的一种状态。
- 任一 RUNNING/no-runtime Run 都能按动作 kind 与 committed 证据分类为可恢复、outcome_unknown 或结构化故障，不能停留在未定义状态。
- outbox completed 必然对应唯一 Outcome。
- restart tests 真正重建 store/service/scheduler 对象。
- ObjectiveLog 与 RunLog 无跨库事务依赖。
- recovery tests 与 lint 通过。

## 风险与回退

Reconciler 不是“失败就重试派发”。如果 Run 状态未知，它必须停下并保留证据。尤其涉及外部副作用时，误重跑比暂时无法推进更严重。
