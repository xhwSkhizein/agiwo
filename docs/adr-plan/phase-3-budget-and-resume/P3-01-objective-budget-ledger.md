# P3-01：实现 ObjectiveBudget 账本与状态配额

状态：planned

## 目标

让每个 Objective 必须携带有限、持久化、不可由 agent 提高的四维预算，并先实现 handoff 与 verification_attempts 两个状态迁移配额的原子检查。预算变化以 facts 累积，不覆盖历史。

## 对应决定

- ADR 0010：ObjectiveBudget 是唯一 Objective 级硬边界。
- ADR 0008、0009：agent/verifier 接力的计数语义。
- ADR 0030：额度不足进入可恢复预算暂停，而非 Objective FAILED。

## 依赖

- P1-02、P2-06 已完成。

## 范围

包含：预算创建/调整/用量 facts、handoff/verification 原子消费、BudgetView 和 agent 只读投影。

不包含：LLM 实际成本检查、active time、DRAINING barrier；后续三个任务实现。

## 实施步骤

1. Objective 创建请求必须提供四个明确、有限、合法的 limit；缺失或无限值不能开始执行。
2. 定义 BudgetConfigured、BudgetAdjusted、BudgetUsageConsumed 等 facts，保留每次用户调整的来源、旧值、新值和时间。
3. max_handoffs 只统计 target=agent/verifier；target=user 和用户回复后创建的新 Assignment 不计数。
4. target=verifier 在同一事务中同时消费一次 handoff 和一次 verification attempt；任一不足时两者都不消费。
5. Decision 接受、预算消费、旧 Assignment Outcome、下一 Assignment/outbox 在同一 ObjectiveStore 事务中校验和提交。
6. agent 只能从 ObjectiveView 读取 limit/used/remaining；模型输出和 Run tool 没有预算修改入口。
7. 用户预算调整使用独立幂等 command；只能由已认证用户边界调用，不从普通 agent Contribution 自动提升。
8. 配额不足返回类型化 BudgetBoundaryHit，记录触发维度、checked_at、used/limit 和待执行动作；P3-05 消费它进入 DRAINING。
9. 不增加 max_runs/max_steps。max_steps_per_run 继续来自 AgentOptions。
10. Console/API DTO 使用明确字段名 max_llm_cost_usd、max_active_seconds、max_handoffs、max_verification_attempts。

## 主要改动位置

- `agiwo/objective/models.py`
- `agiwo/objective/budget.py`
- `agiwo/objective/log.py`
- `agiwo/objective/projection.py`
- `agiwo/objective/service.py`
- `tests/objective/test_budget.py`

## 测试计划

- 创建时四维限制的缺失、负数、零值和合法值。
- agent/verifier/user 三种 target 的精确计数。
- verifier 任一维不足时事务无部分消费。
- 并发接受两个 handoff 时只有预算允许的数量成功。
- 用户调整 append-only、幂等且不能低于已用值。
- agent 侧不存在修改预算 API/tool。

## 完成标准

- ObjectiveBudget 不包装 Scheduler TaskLimits/TaskGuard。
- 所有 handoff/verification 计数只由 ObjectiveService 更新。
- 预算不足不创建下一 Assignment/outbox。
- max_steps_per_run 不进入 ObjectiveBudget。
- budget tests 与 lint 通过。

## 风险与回退

预算检查若与 Assignment 创建分成两个事务会产生超额接力。该用例必须用并发测试证明；无法原子保证时不得合并本任务。
