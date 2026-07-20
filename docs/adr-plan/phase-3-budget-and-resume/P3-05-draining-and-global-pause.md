# P3-05：实现 DRAINING 与可恢复中断屏障

状态：done

## 目标

当预算触顶或用户 pause/停止时，先把整个 Objective 原子置为 DRAINING，立即关闭新工作入口，允许已经开始的在途操作完成，再等待全部活动 Run 进入 PAUSED，最后投影为 `BUDGET_PAUSED` 或 `USER_PAUSED`。不使用 ObjectiveActionLease。运行中用户输入**不**走本路径（见 P4-04）。

## 对应决定

- ADR 0010：任一分支触发预算时暂停整个 Objective。
- ADR 0028：DRAINING 只服务可恢复中断收敛。
- ADR 0029：用户暂停/停止与预算暂停统一为可恢复中断。
- ADR 0030、0031：Objective/Assignment 暂停状态。
- ADR 0044：ObjectiveService 拥有预算与 DRAINING 控制。

## 依赖

- P2-03、P3-01、P3-02、P3-03、P3-04 已完成。

## 范围

包含：drain fact、状态门禁、活动分支快照、Run PAUSED barrier、Budget/User Pause、两阶段 resume 和状态投影。

不包含：运行中用户输入（P4-04）；Session 归档的 UI/持久化由 P5-06 消费本任务已定义的 `user_archive` drain，不在此实现 archive 字段。不包含任何 ActionLease。

## 实施步骤

1. 定义 ObjectiveDrainStarted：reason 封闭枚举至少包含 `budget`、`user_pause`、`user_archive`（三者本任务一起落地，不得留待 P5 再扩枚举）、source、trigger sequence、当时 active assignment/run ids、idempotency key。barrier 成员是这些 run ids，不是租约集合。
2. 为 objective-managed root/child Run 提供窄 admission/progress gate：LLM、tool execute、child spawn、dispatch 在真正开始前确认 Objective 仍允许推进。
3. 删除并禁止重新引入 `ObjectiveActionLease`。不把只读 preflight 结果当作执行权。
4. 任一 BudgetBoundaryHit、用户 pause/停止、或 archive 触发的 drain command 在同一 ObjectiveStore 串行点转为 DRAINING，并把当时全部 active Runs 固定为 barrier；同一 reason/目标的重复触发合并到同一 drain。
5. 已经开始的在途操作允许完成并提交；DRAINING 之后新动作启动检查失败。接受极窄残余竞态（每并发分支至多一个已越过检查点的额外动作）。
6. 通过 **P2-02 已定义的** Scheduler facade 可恢复中断 API 向活动 Run 发送 pause；本任务只实现消费与 barrier，不新增平行 facade 方法。ObjectiveService 不直读 AgentStateStorage。
7. barrier 等待所有指定 Runs 均 PAUSED（且已写 checkpoint）。满足前保持 DRAINING；之后：
   - `budget` → `BUDGET_PAUSED`
   - `user_pause` → `USER_PAUSED`
   - `user_archive` → 同样进入 `USER_PAUSED`（与用户暂停共用 checkpoint barrier；归档标记由 P5-06 在稳定 pause 之后写入）
   Assignment PAUSED 且无 Outcome。
8. **不**为运行中用户输入预留 steering Outcome barrier；P4-04 不调用本 drain。
9. Budget resume 校验配额已提高或 active_seconds 新窗口已确认；User resume（含曾因 archive 而 pause 后用户恢复 Session 再 resume）不要求配额变化。
10. Resume prepare → 共享 release barrier → RunResumed；任一 prepare 失败则全部保持 PAUSED。
11. DRAINING 投影显示 reason 与未收敛 Run；不展示 lease。
12. COMPLETED/FAILED 不接受 drain/resume；WAITING_USER 用用户输入创建新 Assignment，不恢复已结束 Run。
13. 为本任务三种 reason 各写至少一条 barrier 集成测试（含 `user_archive` → USER_PAUSED，且无 Outcome）；P5-06 不得再扩展 reason 枚举。

## 主要改动位置

- `agiwo/objective/service.py`
- `agiwo/objective/budget.py`
- `agiwo/objective/projection.py`
- `agiwo/objective/dispatch.py`
- `agiwo/agent/run_loop.py`
- `agiwo/scheduler/engine.py`
- `agiwo/scheduler/_tree_ops.py`
- `agiwo/scheduler/runner.py`

## 测试计划

- root+多个 child 中任一触发预算，全树停止创建新调用/tool/child/handoff。
- 在途完成、未开始被拒；execute 前并发 DRAINING 则 tool 不执行。
- barrier 未齐时保持 DRAINING。
- pause 不产生 Outcome/termination；resume 使用原 IDs。
- 运行中用户输入路径不进入 DRAINING（与 P4-04 交叉测）。
- `user_archive` 与 `user_pause` 共用 PAUSED barrier，投影均为 USER_PAUSED；reason 在 DrainStarted fact 可区分。
- 两阶段 resume 失败/成功矩阵。
- 代码与测试中无 ObjectiveActionLease、无 user_steering drain reason；reason 枚举含且仅含本任务定义的集合（含 user_archive）。

## 完成标准

- 任何稳定暂停状态都能证明所有活动 Run 已 PAUSED + checkpoint。
- 任意分支无法在 DRAINING 后绕过 gate 产生新工作事实。
- ObjectiveService 不调用 Scheduler cancel 实现 pause。
- Budget/User Pause 均可重启后恢复同一 Run。
- objective/scheduler/agent integration tests 与 lint 通过。

## 风险与回退

全局 gate 不能只放在 Scheduler tick；模型 hook、tool execute、outbox 和 handoff 都需要二次检查。pause 信号必须 objective-agnostic 且与 force_fail 分离，避免 Run 被标 CANCELLED/FAILED。
