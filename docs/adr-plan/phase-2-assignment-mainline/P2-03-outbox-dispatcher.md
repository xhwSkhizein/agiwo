# P2-03：实现 Outbox dispatcher 与幂等派发

状态：**done**（2026-07-18）

## 目标

实现 Objective 包内的机械 dispatcher：领取已提交的 DispatchRequested，使用预分配 run_id 通过 Scheduler facade 启动 Assignment root Run，并在崩溃或重复投递时识别既有执行。dispatcher 不读取自然语言，也不选择 agent、pattern 或 workflow。

## 对应决定

- ADR 0001：执行控制集中且不做语义判断。
- ADR 0019：Transactional outbox 与稳定 ID。
- ADR 0034：ObjectiveService 只调用 Scheduler facade。
- ADR 0044：ObjectiveService 拥有 dispatcher 与 Objective 状态写入。

## 依赖

- P1-03、P1-04、P2-02 已完成。

## 范围

包含：claim loop、Scheduler 调用、状态检查、幂等派发、outbox 交付状态和基础恢复分支。

不包含：完整 restart reconciler 和 checkpoint 恢复；P3-06 完成。

## 实施步骤

1. 在 ObjectiveService 内部启动/停止 dispatcher lifecycle；它不是第二个 public service。
2. dispatcher 原子 claim pending/lease-expired record，记录 owner、attempt 和 lease deadline。
3. claim 后、真正启动前重新读取 ObjectiveView：只有允许推进的状态才能派发；DRAINING/暂停/终态保持或取消 record，不启动新工作。
4. 根据 session_id 获取系统默认 persistent root agent。不得按 Assignment kind 切换 config 或 agent identity。
5. 构造 Scheduler 的类型化 root dispatch request，传入稳定 state/session/agent/run identity 和 Assignment Input 引用。
6. Scheduler root runtime 严格复用 Session agent identity；`AgentState.task` 在本任务中改名为表达调度输入的 `input` 或 `assignment_input`，避免与 Objective 混淆。
7. 重复投递前先通过 Scheduler facade/RunLog query 检查 run_id：已有 RUNNING 时连接/等待，已有终态时交给 Outcome 对账，尚无 RunStarted 才启动。
8. dispatcher 观察到 committed RunStarted 后，在一个 ObjectiveStore 事务中幂等追加 `AssignmentExecutionStarted(assignment_id, run_id, started_at)` 并把 outbox 标为 dispatched。Objective/Assignment 的 CREATED -> RUNNING 只由该 fact 投影，不能读取 outbox 或跨存储推断 RunStarted。
9. 如果进程在 RunStarted 与 Objective fact 之间崩溃，reconciler 用稳定 objective_id/assignment_id/run_id 补写同一 fact；幂等键保证只发生一次状态迁移。
10. outbox 区分 pending、claimed、dispatched、completed。AssignmentExecutionStarted 只令其 dispatched；AssignmentOutcome 已提交后才 completed。
11. 派发失败记录结构化 attempt/error 并释放或等待 lease；不得直接把 Objective 标为 FAILED。
12. 慢或卡住的 dispatcher 不阻塞 ObjectiveLog 事务和其他 Objective；并发度与 shutdown 有明确边界。
13. 通过 **P2-02 已定义的** Scheduler facade 查询/派发 API 完成 claim 后启动与重复投递识别；Objective 包不得直读 `scheduler.store`，本任务不得再扩展平行查询面。

## 主要改动位置

- `agiwo/objective/service.py`
- `agiwo/objective/dispatch.py`
- `agiwo/objective/outbox.py`
- `agiwo/scheduler/commands.py`
- `agiwo/scheduler/engine.py`
- `agiwo/scheduler/models.py`
- `agiwo/scheduler/runner.py`
- `tests/objective/test_dispatch.py`

## 测试计划

- 正常 claim -> RunStarted -> dispatched。
- RunStarted -> AssignmentExecutionStarted -> Objective/Assignment RUNNING；outbox 状态本身不改变领域投影。
- RunStarted 后崩溃，reconciler 补写一次 AssignmentExecutionStarted。
- 同 dispatch_id、assignment_id、run_id 重放只出现一个逻辑 Run。
- claim 后崩溃、lease 过期后由另一 owner 接管。
- Objective 在 claim 后进入 DRAINING，启动前二次检查阻止派发。
- Scheduler state 已存在、RunLog 已终态、完全未启动三种分支。
- RunStarted 后 outbox 不提前 completed；Outcome 后才完成。
- Objective 模块不读取 scheduler store 的架构测试。

## 完成标准

- 不存在 Objective facts 已提交但只能依赖一次内存回调才能启动的 Assignment。
- 重复派发不创建第二个 RunStarted 或第二个 Assignment。
- dispatcher 的决定只依赖结构化状态和稳定 ID。
- Scheduler PendingEvent 未被复用。
- objective/scheduler tests 与 lint 通过。

## 风险与回退

`dispatched` 不等于 `completed`。若实现把 RunStarted 当成 outbox 完成，进程可能在 Outcome 前崩溃并永久丢失交付闭环；这条断言必须有故障注入测试。
