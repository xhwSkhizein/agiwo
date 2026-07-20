# P2-02：增加稳定 Run identity 与内部执行请求

状态：**done**（2026-07-18）

## 目标

允许 ObjectiveService 在提交 Assignment 与 outbox 时预先分配 root run_id，并把 objective_id、assignment_id 和 Assignment root/child 身份作为一等字段传入 Run。公开 `Agent.start/run/run_stream` 参数保持不变，直接调用仍自动生成 run_id。

## 对应决定

- ADR 0019：outbox 派发前预分配稳定 run_id。
- ADR 0032、0033：Run identity 必须支持后续同 ID pause/resume。
- ADR 0036：Assignment 复用 Session agent identity，但使用新 run_id。
- ADR 0006：Assignment root 身份必须类型安全，不能从松散 metadata 猜测。

## 依赖

- P0-04、P1-01 已完成。

## 当前源码现状

- `Agent.start()` 和 `run_child()` 直接调用 `uuid4()` 创建 run_id。
- `RunIdentity.metadata` 可以塞任意值，但 RunStarted 没有 Objective/Assignment 一等字段。
- Scheduler runner 调用 Agent public run path，没有传递预分配 identity 的内部协议。
- `RunStatus` 仍包含 STARTING/CANCELLED，缺少 PAUSED/INTERRUPTED。

## 范围

包含：Agent 内部执行请求、RunIdentity 字段、RunStarted/facts/serialization、**Objective 所需 Scheduler facade 机械契约的一次定义**（含后续 P3 才实现的 pause/query 方法签名）、root/child 继承和 RunStatus 枚举。

不包含：pause/resume 实际控制逻辑、outbox claim、Assignment finalization（由后续任务按本契约实现）。

## 实施步骤

1. 在 `agiwo.agent` public facade 可用但不作为普通用户 API 的位置定义 `RunExecutionRequest` 或等价窄 DTO：run_id、objective_id、assignment_id、responsibility_scope / assignment_role、config/template hash 和必要恢复标记。**不包含 FinalizationSpec**；root 收尾由 `assignment_role=root` 触发 Agent 内置协议（ADR 0006 / P2-05）。
2. `RunIdentity` 增加可选 objective_id、assignment_id 和 assignment_role（root/child/none）一等字段；不得只写 metadata。
3. 保持 `Agent.start/run/run_stream` 签名不变。它们内部创建默认 request 并自动分配 run_id。
4. 为 Scheduler runner 增加内部调用入口，接受已校验 request 并复用 Agent 的同一执行 owner；不要复制第二套 run loop。
5. 扩展 Scheduler 的公开机械派发 DTO，使 ObjectiveService 可以传递预分配 identity。Scheduler 只校验 ID/状态并透传，不导入 Objective 模型或解释 Assignment kind。
6. **一次性定义并冻结 Objective→Scheduler facade 扩展契约**（方法可先 stub / NotImplemented，但签名与语义本任务写死；后续任务不得另起平行 API）。至少包括：
   - `dispatch_execution(request) -> handle/state`：按预分配 run_id 启动或连接既有 Run；
   - `get_run_view(run_id)` / `get_run_status(run_id)`：只读机械状态；
   - `list_execution_tree(root_run_id | assignment_scope)`：当前 Assignment 下 root+children 的 run ids 与状态；
   - `request_recoverable_pause(run_ids, reason)`：向指定 Runs 发送 pause 信号（P3-04/P3-05 实现）；
   - `prepare_resume(run_ids)` / `release_resume_barrier(...)`：两阶段恢复（P3-04/P3-05 实现）；
   - `inject_user_message(root_run_id, message)`：运行中注入 false user（P4-04 实现）；
   - 只读通知/订阅边界若 SSE 需要，给出窄订阅句柄类型（P5-02 消费，不在此发明第二套协议）。
   全部方法均为 objective-agnostic 机械面：无 ObjectiveView、无 Decision、不写 ObjectiveStore。
7. `RunStarted`、RunView、Trace 和 storage codec 使用一等字段保存 objective_id/assignment_id/role。
8. child Run 由 parent 的 RunIdentity 自动继承 objective_id 和 assignment_id，但生成自己的 run_id、agent identity 和 parent_run_id；child role 固定为 child。
9. RunStatus 统一为 RUNNING/PAUSED/COMPLETED/INTERRUPTED/FAILED。现有强制 cancel 映射为失败执行事实与 CANCELLED termination reason，不把 CANCELLED 保留为可恢复状态。
10. 对已存在 run_id 的派发提供确定性结果：尚未开始可启动，已运行返回已有 handle/state，终态不重复启动。
11. 更新 Agent/Scheduler contract tests，证明 public API 无新参数、objective-managed 内部路径可指定 ID，且 facade 扩展契约有类型/文档锚点（即使部分方法尚未实现）。

## 主要改动位置

- `agiwo/agent/models/run.py`
- `agiwo/agent/models/log.py`
- `agiwo/agent/agent.py`
- `agiwo/agent/runtime/context.py`
- `agiwo/agent/runtime/state_writer.py`
- `agiwo/agent/storage/serialization.py`
- `agiwo/scheduler/commands.py`
- `agiwo/scheduler/engine.py`
- `agiwo/scheduler/runner.py`
- `agiwo/scheduler/scheduler.py`（facade 契约文档与 stub）

## 测试计划

- `inspect.signature` 证明 Agent 三个公开方法无变化。
- direct Agent 自动 ID 与 Objective internal request 指定 ID 两条路径。
- RunStarted memory/SQLite round-trip 保留三类关联字段。
- child 继承 objective/assignment，但拥有独立 run_id/agent_id。
- 重复派发同 run_id 不产生第二个 RunStarted。
- RunStatus 投影和旧 cancel 路径回归。
- facade 契约：Objective 包测试只依赖公开方法名/类型，不导入 `scheduler.store`；未实现方法有明确错误而非静默 no-op。

## 完成标准

- Objective 归属不依赖 prompt、UserMessage 或 metadata 解析。
- ObjectiveService 不调用 Agent 内部模块，只通过 Scheduler facade 使用执行请求与后续查询/pause/注入面。
- Scheduler 不理解 Objective 状态、Decision 或模板。
- 同一 Session 的 peer Assignment 使用相同 root agent_id、不同 assignment_id/run_id。
- P2-03 / P3-04 / P3-05 / P3-06 / P4-04 / P5-02 只实现或消费本任务定义的 facade 方法，不得再“增加所需查询能力”式发散定义。
- agent/scheduler/storage tests 与 lint 通过。

## 风险与回退

不要把可恢复执行入口直接暴露为 Agent 新 public 参数，否则调用者会绕过 Objective/Outbox 不变量。内部请求应由 Scheduler facade 接受并在 Agent facade 内落地，run loop 仍只有一个 owner。facade 契约若在后续任务被平行分叉，视为回归并收回本任务完成状态。

