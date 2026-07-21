# Objective 派发使用 Transactional Outbox

ObjectiveService 接受 Decision 后，在同一个 ObjectiveStore 事务中追加 Decision 接受、预算结算、Assignment 创建等 Objective facts，并写入 `DispatchRequested` outbox record。ObjectiveService 在这个事务中为 Assignment 的 root Run 预先分配稳定 `run_id`；独立的机械 dispatcher 幂等消费 outbox，把该 ID 经 Scheduler 透传给 Agent 后启动 Run。Scheduler 不接受 Decision 或写 ObjectiveStore，其所有权见 ADR 0044。ObjectiveLog 与 AgentStateStorage 不要求跨存储事务；进程在任意边界崩溃后，未完成的 outbox record 都可以安全重放。

## Status

superseded by ADR-0047

## Considered Options

- 写完 ObjectiveLog 后直接调用 Scheduler runtime：正常路径最短，但进程若在两步之间崩溃，会留下永远没有执行者的 Assignment。
- 让 ObjectiveStore 与 AgentStateStorage 参加分布式事务：一致性强，但显著增加 memory、SQLite 等后端的实现和运维复杂度。
- 复用 PendingEvent：已有队列入口方便，但 PendingEvent 面向 agent mailbox、消费后删除，不能证明一个已提交 Assignment 最终被派发。

## Consequences

- ObjectiveStore 必须在一次原子事务中同时追加相关 ObjectiveLog facts 与 outbox record；任一写入失败则整个 Decision 接受操作不生效。
- outbox record 至少包含稳定的 `dispatch_id`、`objective_id`、`assignment_id`、`run_id`、目标职责、创建时间、状态、attempt 和 lease 信息。
- Objective 管理的 root Run 不在 `Agent.start()` 内临时生成 `run_id`。ObjectiveService 创建 Assignment 时预先分配 ID，Scheduler 的派发链只负责透传；直接使用 Agent 或 Scheduler 的低层调用仍可省略该参数并沿用自动生成行为。
- `RunStarted` 以一等字段记录 `objective_id` 与 `assignment_id`。恢复逻辑不依赖提示词、普通消息或松散 metadata 推断 Run 属于哪个 Objective 和 Assignment。
- dispatcher 观察到 committed RunStarted 后，必须幂等追加 `AssignmentExecutionStarted` Objective fact，并在同一 ObjectiveStore 事务中把 outbox 标为 dispatched。Objective 与 Assignment 的 CREATED -> RUNNING 投影只消费该 fact，不能消费 outbox 状态或跨存储推断 RunStarted。
- dispatcher 只消费已经确定的派发指令，不读取自然语言重新选择 agent 或 pattern；它仍是纯执行骨骼。
- `assignment_id` 与 `run_id` 是幂等键。重复投递可以重复检查和修复快照，但不能创建第二个 Assignment 或第二个逻辑 Run。
- dispatcher 使用有期限的 claim lease；进程在 claim 后崩溃时，其他实例可在 lease 到期后重新取得并继续。
- AgentStateStorage 仍是可重建调度快照。先成功派发、后未及时更新 outbox 状态时，重放必须通过稳定 id 识别既有执行，而不是重复启动。
- outbox 在对应 AssignmentOutcome 成功追加到 ObjectiveLog 前不能标记完成；Run 启动成功只表示派发完成，不表示这项可靠交付已经闭环。
- 恢复或重放时，dispatcher/reconciler 按稳定 `run_id` 查询 RunLog：已有终态时补写 AssignmentOutcome；存在未终结 checkpoint 时恢复同一 Run；Run 正在活动时连接或等待既有执行；尚无 RunStarted 时重新派发。
- 如果 RunStarted 已提交而 `AssignmentExecutionStarted` 缺失，reconciler 依据 objective_id、assignment_id 与 run_id 幂等补写该 Objective fact；重复补写不能产生第二次状态迁移。
- 若 RunLog 状态无法确认，保留 outbox 并进入结构化故障处理，不能擅自重跑或假定结果。重复补写 Outcome 使用幂等键保证 ObjectiveLog 只接受一次。
- ObjectiveLog 记录领域事实，outbox 记录可靠投递进度；两者可以位于同一物理 ObjectiveStore，但不得把 outbox 状态当作 Objective 语义状态。
- 现有 PendingEvent 继续服务 scheduler mailbox 与唤醒，不承担 Objective Assignment 的可靠派发职责。
