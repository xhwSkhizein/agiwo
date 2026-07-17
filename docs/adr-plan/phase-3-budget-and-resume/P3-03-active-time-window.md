# P3-03：实现 Objective 活动窗口

状态：planned

## 目标

以 `checked_at - current_active_started_at` 限制当前 Objective 活动窗口。首次执行和每次 checkpoint 恢复都开启新窗口并从零计时；等待时间和旧窗口用量保留审计，但不带入新窗口。

## 对应决定

- ADR 0010：每次模型调用与 handoff 前检查 active time。
- ADR 0013：checkpoint 恢复时重新计算，不覆盖首次启动时间。

## 依赖

- P3-01 已完成。

## 范围

包含：活动/等待 facts、投影、检查点、恢复重置和并行共享窗口。

不包含：进程崩溃 heartbeat、基础设施停机时间扣除等 ADR 已排除的极端情况。

## 实施步骤

1. Objective 首次实际启动写 ActiveWindowStarted，并设置不可变 first_started_at 与投影 current_active_started_at。
2. 模型调用前和 ObjectiveService 提交 handoff 前，以注入 clock 的 checked_at 计算当前窗口 used seconds。
3. 达到或超过 limit 时不开始调用、不提交 handoff，记录 BudgetBoundaryHit(active_seconds)。
4. 多个并行 Run 读取同一个 current_active_started_at；重叠时间不累加，各分支不拥有独立窗口。
5. 只有 Objective 整体停止推进时关闭窗口。某个 child 等待而其他分支运行时不关闭。
6. 进入 WAITING_USER、BUDGET_PAUSED、USER_PAUSED 或终态时写 ActiveWindowEnded；适用时写 WaitingStarted。
7. checkpoint resume 先写 WaitingEnded，再写新的 ActiveWindowStarted；新窗口 used 从零，first_started_at 不变。
8. active_seconds 触发后，用户确认继续即可用原 limit 开新窗口，不要求提高额度。
9. 不使用常驻 timer；恢复后下一次 LLM/handoff 边界再检查。
10. Timeline 同时显示每个窗口和等待区间，BudgetView 区分当前窗口 used 与历史审计总时长。

## 主要改动位置

- `agiwo/objective/budget.py`
- `agiwo/objective/log.py`
- `agiwo/objective/projection.py`
- `agiwo/objective/service.py`
- Objective LLM hook
- `tests/objective/test_active_time.py`

## 测试计划

- fake clock 下首次窗口、等待、恢复新窗口的精确计算。
- first_started_at 始终不变。
- 并行 Run 不重复累加时间。
- 某 child 等待但 root/其他 child 活动时窗口保持打开。
- LLM 与 handoff 两个检查边界。
- 恰好等于 limit 时拒绝新动作。

## 完成标准

- 计算不使用“首次时间减累计等待”或各 Run duration 之和。
- pause/resume 后当前窗口用量归零，历史窗口仍可审计。
- 不存在 agent 修改 active-time limit 的路径。
- active time tests 与 lint 通过。

## 风险与回退

时间逻辑必须使用可注入 clock，不能在领域代码散落 `datetime.now()`。否则边界和并发测试会不稳定，也无法证明恢复重置语义。

