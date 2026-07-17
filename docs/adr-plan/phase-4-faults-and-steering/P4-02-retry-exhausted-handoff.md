# P4-02：实现重试耗尽接力

状态：planned

## 目标

当幂等、retryable 且阻断当前 Run 的操作耗尽自动重试时，由系统从 committed facts 构造 report 和机械的 target=agent Decision，结束当前责任并创建 fresh Assignment。故障模型或工具不再被要求总结自己。

## 对应决定

- ADR 0011：重试耗尽不令 Objective 失败。
- ADR 0015：接力给全新 Assignment/Run。
- ADR 0024：中断 Assignment 仍必须有 Outcome。

## 依赖

- P2-06、P3-01、P4-01 已完成。

## 范围

包含：系统 report、carry_forward、机械 Decision、handoff quota、INTERRUPTED 状态和 fresh dispatch。

## 实施步骤

1. 只处理 run_blocking=true 的 retryable fault exhaustion；已形成普通 ToolResult 且 Run 可继续的 failure 不进入本路径。
2. 从结构化 operation/fault、attempt 时间线、最后 checkpoint、已有 Artifact 和当前 root RunPlan 构造系统 report，不调用 LLM。
3. report 明确 provenance=system_retry_exhausted，不伪装成 assistant summary。
4. 生成固定 HandoffDecision(target=agent)，模型不能改成继续旧 Run或 user/verifier。
5. Outcome 包含 pending/active RunPlan 项及 carry_forward，root Run/Assignment 目标状态为 INTERRUPTED。
6. 在提交终态 Outcome 前，原子检查/消费 handoff quota。额度可用时，Outcome、计数、fresh Assignment 和 outbox 一次提交。
7. 额度不足时先进入 DRAINING(reason=budget)，checkpoint phase=retry_exhausted_handoff；此时旧 Assignment 尚不写终态 Outcome。用户提高额度后从该 phase 提交机械 Outcome 和新 Assignment，不重试原操作。
8. 新 Assignment 复用 Session root agent identity，使用新 assignment_id/run_id 和最近 Outcome，自主选择不同工具或策略。
9. outbox/recovery 使用幂等键，崩溃后不能产生多个中断 Outcome 或多个接棒 Assignment。

## 主要改动位置

- `agiwo/objective/faults.py`
- `agiwo/objective/service.py`
- `agiwo/objective/budget.py`
- `agiwo/agent/retry/`
- `agiwo/agent/models/log.py`
- `tests/objective/test_retry_exhaustion.py`

## 测试计划

- blocking exhaustion -> system report -> INTERRUPTED -> fresh work Assignment。
- nonblocking tool failure 不触发此路径。
- report 包含 attempts/checkpoint/artifacts/carry_forward 且无额外模型调用。
- handoff quota 可用和不足两条路径。
- budget resume 不重复原操作，只完成 Outcome/handoff phase。
- crash 在 Outcome/next Assignment 事务边界不产生重复。

## 完成标准

- 故障模型不可用时路径仍能完整结束。
- 旧 Run 不复活、不继续 retry；新 Run 使用相同 Session agent identity。
- handoff 计数正确，额度不足时可恢复而非 Objective FAILED。
- 每个终态 Assignment 只有一个 Outcome。
- objective/retry integration tests 与 lint 通过。

## 风险与回退

先结束 Assignment、后单独检查 handoff quota 会留下无法继续的终态空洞。终态 Outcome、quota 和下一 Assignment 必须在同一事务，或在 quota 不足时保持 checkpoint 待收口。

