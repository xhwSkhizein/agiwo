# RunStatus 支持一等 Pause 与 Resume

RunStatus 固定为 `RUNNING`、`PAUSED`、`COMPLETED`、`INTERRUPTED`、`FAILED`。Run 在 Budget Pause 或 User Pause 时写入 RunPaused 与 checkpoint，保持同一 run_id、Assignment、agent identity、消息和已提交 step；恢复时追加 RunResumed 并继续原 Run。Pause/Resume 不写 termination reason，也不通过创建第二个 Run 模拟。

## Status

accepted

## Considered Options

- Pause 时写 CANCELLED，恢复时新建 Run：可以复用现有终态路径，但丢失原 Run 连续性，并让取消与暂停无法区分。
- 只把 pause 记录在 ObjectiveLog：Objective 状态可见，但 RunLog 无法解释为什么同一个 Run 长时间没有结束，也不能独立重建待执行阶段。
- 序列化并冻结进程内 coroutine：理论上最接近原执行，但无法可靠跨进程或版本恢复，也不适合持久化后端。

## Consequences

- RunStarted 创建 RUNNING 投影；RunPaused 转为 PAUSED；RunResumed 回到 RUNNING；RunFinished、RunInterrupted、RunFailed 分别产生三个终态。
- RunPaused fact 至少记录 pause reason、checkpoint reference、最后 committed sequence、Objective/Assignment ids 和发生时间。
- checkpoint 是薄控制游标；恢复下一步由消息列表末项形态决定（ADR 0029/0033）；消息、step、ledger、tool result 和 Artifact 从 RunLog 重建。
- Pause 不写 RunFinished、RunInterrupted、RunFailed、TerminationDecided 或 AssignmentOutcome；Assignment 同步进入 PAUSED，但仍是同一次责任。
- Budget/User resume 追加 RunResumed，使用相同 run_id 和 agent instance identity；实际 Python Agent 对象可以重建，但必须加载相同配置快照和 RunLog 上下文。
- `RunResumePrepared` 只记录某个 Run 已通过恢复校验并在共享 barrier 上等待，RunStatus 仍是 PAUSED。只有 Objective 的全部活动 Run prepare 成功并提交 resume 后，统一 release 才允许各 Run 追加 RunResumed 和开始新动作。
- 运行中用户输入不结束 Run（ADR 0023）。max_steps_per_run 与 retry exhaustion 结束当前 Run 时使用 INTERRUPTED；正常 Assignment 收尾调用使用 COMPLETED。
- FAILED 只表达真正不可恢复的执行契约破坏；普通用户停止不得写成 FAILED/CANCELLED。Objective 层对真正结束的 Assignment 仍须从 committed facts 形成 Outcome。
- 现有 TerminationReason.CANCELLED 与 Scheduler.cancel_subtree 不承担可恢复中断语义。
