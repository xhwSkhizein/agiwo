# P3：预算、暂停与恢复

本阶段把 ObjectiveBudget 从“数据字段”变成真正不可绕过的执行边界，并增加同 Run 的 checkpoint pause/resume。所有并行 child Run 共享同一 Objective 预算和 DRAINING 屏障。

## 入口条件

- P2 正常主链可以完成并重放。
- 所有 LLM attempt 已有统一调用 phase 和计数。
- Objective root/child Run 都带有一等 objective_id 与 assignment_id。

## 任务顺序

| ID | 任务 | 依赖 |
| --- | --- | --- |
| P3-01 | ObjectiveBudget 账本与状态配额 | P1-02、P2-06 |
| P3-02 | LLM 实际成本检查与记账 | P0-04、P3-01 |
| P3-03 | Objective 活动窗口 | P3-01 |
| P3-04 | Run checkpoint 与可恢复中断 resume | P0-04、P2-02 |
| P3-05 | DRAINING 与全局暂停屏障 | P2-03、P3-01 至 P3-04 |
| P3-06 | 重启恢复与 Outbox 对账 | P2-03、P3-04、P3-05 |

P3-02、P3-03、P3-04 可以在 P3-01 完成后并行；P3-05 汇合三者，P3-06 最后完成持久化恢复闭环。

## 阶段出口

- handoff、verification、LLM cost、active time 四个预算维度都有 append-only 用量事实。
- 模型调用开始前检查 `used_llm_cost_usd + call_cost_ceiling <= max_llm_cost_usd`；收到有效响应后按请求 token 与实际接收 token 幂等记账，无响应错误为零成本。不预留、不结算、不返还。
- 顺序执行在 ceiling 正确时不超支；并行执行最坏超支不超过 ceiling × 当时通过检查的并发调用数；此后拒绝新的模型调用并进入 DRAINING。
- 所有 objective-managed LLM/tool/spawn/dispatch 在真正开始前确认 Objective 仍可推进；DRAINING 后拒绝新动作，已经开始的在途操作允许完成。barrier 是活动 Run 的 PAUSED/checkpoint，不使用 ObjectiveActionLease。direct Agent 不启用该门禁。
- 任一分支触发预算或 pause 后，Objective 先进入 DRAINING，再在所有活动 Run PAUSED 后进入稳定暂停状态。运行中用户输入不走 DRAINING。
- Budget Pause 和 User Pause 不结束 Assignment/Run，不产生 Outcome 或 termination reason；普通停止不走 cancel→FAILED。
- resume 使用相同 run_id、Assignment、agent identity、消息和已提交 step，并按消息末项形态继续。
- 多 Run resume 先 prepare/validate 全部 runtime，再提交 Objective resume 并释放单一 barrier；任一 prepare 失败时全部保持 PAUSED。
- 进程在 outbox claim、Run 启动、暂停或 Outcome 提交之间崩溃后，重启能对账而不重复副作用。
