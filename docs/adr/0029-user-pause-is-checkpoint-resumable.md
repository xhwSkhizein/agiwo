# 用户暂停、预算暂停与用户停止统一为可恢复中断

用户暂停、Console「停止」、预算触顶使用同一套可恢复中断机制：经 DRAINING 收敛后，活动 Run 进入 `PAUSED`，Assignment 不结束、不写 Outcome。恢复时沿用同一 Assignment、agent、run_id，并根据当前 llm context（RunLog/StepView 重建的消息列表）最后一项决定下一步，而不是依赖复杂的独立 resume_phase 状态机。现有 Scheduler.cancel 的失败/CANCELLED 语义不得再表示普通用户停止。

## Status

accepted

## Considered Options

- 复用现有 cancel：实现现成，但会把 Run 标失败，无法恢复。
- pause 与 budget 两套协议：语义标签清楚，但恢复规则重复。
- 统一可恢复中断，按消息末项恢复（本决定）：与 agent loop 自然形态一致，实现更简单。

## Consequences

- Objective 投影可区分 `USER_PAUSED` 与 `BUDGET_PAUSED`（原因不同），但底层都是可恢复中断。
- Gateway 的 pause/停止与预算触顶都进入同一收敛与 PAUSED 路径；提高配额或用户 resume 后继续同一 Run。
- 恢复规则（确定性）：
  - 最后一项为 assistant 且含 tool_call → 执行这些 tool call；
  - 最后一项为 assistant 且无 tool_call → 追加 `is_user_provided=false` 的 user 消息，让模型继续；
  - 最后一项为 tool result → 同样追加 false user，让 loop 继续。
- 不写 RunFinished / RunFailed / CANCELLED termination 表示普通停止。
- 内部进程 teardown、不可恢复基础设施故障仍可走强制终止，并进入 FAILED / 领域失败路径；与用户可恢复停止分离。
- 本 ADR 部分取代「pause 与 cancel 必须永久分叉为两套用户语义」且依赖复杂 checkpoint phase 才能恢复的表述；薄 last_committed_sequence 与**钉住的**配置/模板快照（ADR 0033）用于重建与完整性校验，resume 不改用 live Registry。
- 归档含活动 Objective 的 Session 时，先走可恢复中断再归档（ADR 0041）。
