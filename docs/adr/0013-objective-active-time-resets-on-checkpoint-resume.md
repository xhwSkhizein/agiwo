# Objective 活动时间在 checkpoint 恢复时重新计算

`ObjectiveBudget.max_active_seconds` 使用 `checked_at - current_active_started_at` 计算当前 Objective 活动窗口的时长。Objective 首次启动和每次从 checkpoint 恢复都会开启新的活动窗口，限额从零重新计算；进入 `WAITING_USER`、`BUDGET_PAUSED` 或终态时关闭窗口。Objective 保留不可变的首次启动时间，并以 append-only facts 记录活动窗口与等待区间，不能通过覆盖 `Objective.first_started_at` 表达恢复。

## Status

accepted

## Considered Options

- 从首次启动时间中扣除全部累计等待时间：能够得到 Objective 的累计活动时长，但恢复后仍继承此前消耗，不符合 checkpoint 恢复后重新计时的要求。
- 覆盖 `Objective.started_at` 为恢复时间：计算简单，但会丢失首次启动事实，也无法审计 Objective 经历过哪些执行窗口。
- 累加每个 Run 的执行耗时：适合估算计算资源，但并行 Run 会重复计时，不符合 Objective 级活动窗口。

## Consequences

- Objective 区分不可变的 `first_started_at` 与当前窗口投影 `current_active_started_at`；后者来自最近一次活动窗口开始事实。
- 首次启动和每次 checkpoint 恢复都写入新的活动窗口开始事实；窗口恢复时 `max_active_seconds` 的已用值归零。
- 进入 `WAITING_USER`、`BUDGET_PAUSED` 或终态时写入活动窗口结束事实，并同时开始适用的等待区间。
- 恢复时闭合等待区间并开启新活动窗口。等待时长仍记录到 Objective 供审计，但不从首次启动时间做累计扣减，也不带入新窗口预算。
- 只有 Objective 整体停止推进时才关闭窗口。单个并行分支等待、其他分支仍在工作时，当前活动窗口继续计时。
- 多个分支共享同一个 `current_active_started_at`，不把并行时长相加；任一分支在模型调用或 handoff 前发现窗口超限，都会触发整个 Objective 的可恢复暂停。
- 若暂停仅由 `max_active_seconds` 触发，用户确认继续后可在不提高上限的情况下开启新窗口并恢复同一 checkpoint。
- 当前不为进程崩溃、计划停机或基础设施不可用建立 heartbeat 或专用等待区间；这些极端情况暂不处理。
