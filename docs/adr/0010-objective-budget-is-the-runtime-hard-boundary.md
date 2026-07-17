# ObjectiveBudget 是目标运行的硬边界

每个 Objective 使用自身的 `ObjectiveBudget` 作为动态 workflow 的唯一 Objective 级硬边界，不另建部署级 `SystemCeiling` 领域模型。执行系统负责在运行中检查预算、更新 Objective 用量并阻止超额推进；agent 只能读取预算状态，无权提高配额。任意并行分支达到阈值时，系统在安全 step 边界暂停整个 Objective 的活动 Run，向用户报告触发维度、当前用量和已有产出，并等待用户设置新配额后从原 Run checkpoint 继续。

## Status

accepted

## Considered Options

- 同时设置部署级 SystemCeiling 与 ObjectiveBudget：能够增加一层平台保护，但在 Objective 领域内形成两套重叠的配额来源和恢复规则。
- 让 agent 根据复杂度自行提高预算：灵活性最高，但失去硬边界，错误判断可以直接扩大成本和循环长度。
- 达到预算后直接令 Objective 失败：状态简单，但会丢失继续利用已有产出的机会，也剥夺用户追加配额的选择。

## Consequences

- Objective 创建时必须具有明确、有限且可持久化的 ObjectiveBudget；缺少有效预算不能开始执行。
- ObjectiveBudget 固定包含 `handoffs`、`verification_attempts`、`llm_cost_usd`、`active_seconds` 四个配额维度；每个维度只记录 `limit` 与已经确认的 `used`，不记录预留额度。
- `handoffs` 与 `verification_attempts` 是 Objective 状态迁移配额，由 ObjectiveService 在接受 Decision 并原子提交下一 Assignment 时检查和结算；它们不进入 RunLimitPolicy。Scheduler 不接受 Decision，所有权见 ADR 0044。
- `max_handoffs` 只统计 `target=agent` 与 `target=verifier` 的自动接力；`target=user` 和用户回复后创建的新 Assignment 不计数。`target=verifier` 同时消耗一次 handoff 和一次 verification attempt。
- `max_llm_cost_usd` 是跨所有 Assignment Run 汇总的 LLM 美元成本启动阈值。每个模型 attempt 开始前执行 LLM 调用前成本检查：`used_llm_cost_usd + call_cost_ceiling <= max_llm_cost_usd`；失败则拒绝调用并触发暂停。不预留、不结算、不返还。
- Assignment 内由 Parallel、Pipeline、Agent 或其他委派方式产生的 child Run 继承相同 objective_id、assignment_id 和预算检查；child LLM 调用与 root Run 使用同一个 Objective 成本池。
- 第一版不建立模型与工具共用的 CostEntry；工具执行成本不进入 ObjectiveBudget，也不扩展 ToolResult 的成本契约。
- `max_active_seconds` 按 `checked_at - current_active_started_at` 计算，是当前 Objective 活动窗口的时长，不是各 Run 执行时长之和，也不跨 checkpoint 累计。
- Objective 首次启动时开启第一个活动窗口；每次从 checkpoint 恢复时关闭等待区间并开启新的活动窗口，`max_active_seconds` 从零重新计算。
- `Objective.first_started_at` 必须保持不可变。活动窗口开始、等待开始、等待结束和窗口结束都以 append-only facts 保存，`current_active_started_at` 只是可重建投影，不能靠覆盖首次启动时间表达恢复。
- Objective 进入 `WAITING_USER`、`BUDGET_PAUSED` 或终态时关闭当前活动窗口并记录已经消耗的窗口时长；等待区间仍持久化到 Objective 供审计，但不参与后续窗口的限额计算。
- 只有整个 Objective 都不再推进时才关闭活动窗口；某个并行分支局部等待而其他分支仍在执行时，窗口保持打开。并行分支共享同一个当前窗口，重叠执行时间不重复累加。
- `max_active_seconds` 在每次模型调用前和 ObjectiveService 提交 handoff 前检查。若检查时已经达到阈值，不开始模型调用，也不提交 handoff，而是停止整个 Objective 的继续推进。
- 该限制不依赖常驻计时器；恢复执行后，下一次模型调用或 handoff 使用新活动窗口的起点检查。
- 不设置 `max_runs`；Run 的数量不能可靠代表任务工作量，并且同一 Run 可以跨 checkpoint 暂停和恢复。
- `max_steps_per_run` 是 RunLimitPolicy 的单 Run 正常模型调用阈值，不是 ObjectiveBudget 维度；它在每个 Run 上独立生效，不跨 Assignment 或 Run 累加。触发该阈值本身不会暂停 Objective，而会按照单 Run 收口与 handoff 规则结束当前 Assignment。
- 每次预算用量变化和用户配额调整都以增量事实记录，不能覆盖历史配额与消耗记录。
- 预算耗尽由执行系统处理，不要求 agent 产生 handoff 或收口 Decision。
- termination summary 与 Assignment 收尾调用和普通调用使用相同规则：开始前执行 `used + call_cost_ceiling <= limit`；失败则暂停，成功则开始并在响应后按实际成本记账。
- 若下一次收口调用的前检查已无法通过，而总结或收口尚未开始，系统直接暂停 Objective。checkpoint 必须记录下一项待执行的收口阶段。
- 预算暂停后，系统负责向用户展示预算原因、当前进度和可用 Artifact；用户可以调整配额并恢复同一 Objective。
- 若暂停原因只是 `max_active_seconds`，用户确认继续后可以沿用原上限开启新活动窗口，无需提高该配额；LLM cost、handoff 或 verification 配额不足时仍需先提高对应额度。
- 任意分支触发预算阈值时，整个 Objective 停止创建新的 step、tool call、Run、Assignment 和 handoff，并令所有活动 Run 收敛到暂停状态。
- 预算触发后 Objective 先原子进入 `DRAINING(reason=budget)`，再等待所有活动分支到达 checkpoint；在 barrier 满足前不能直接标记 `BUDGET_PAUSED`。
- 每次 LLM attempt 开始前，ObjectiveService 通过 hook 计算本次 `call_cost_ceiling`（完整请求 token × 输入价 + `max_output_tokens` × 输出价，含适用 cache 计价），并检查 `used_llm_cost_usd + call_cost_ceiling <= max_llm_cost_usd` 且 Objective 仍允许推进。检查失败则不开始调用并触发整个 Objective 暂停；检查成功不写 reserved，也不在调用结束后返还。
- attempt 收到至少一个有效响应数据后，按本次完整请求 token 与实际接收的输出 token 计算成本并幂等追加到 `used_llm_cost_usd`；未收到任何响应的失败记为零成本。Provider retry 是新的 attempt，独立做上界检查和记账。
- 顺序执行在上界正确时不会因该规则使 `used` 越过 limit。并行执行可能基于同一 `used` 同时通过检查，最坏超支被「单次调用成本上界 × 当时通过检查的并发调用数」封顶。任一实际记账令 `used + 下一调用上界` 不再满足启动条件后，Objective 进入 DRAINING 并拒绝新的模型调用，但不取消已经开始的在途调用。
- `max_llm_cost_usd` 是带可文档化超支上界的“新调用启动边界”。界面必须同时显示 limit、actual used、本次 ceiling 与可能的并行超出，不能把超出显示为记账错误，也不得重新引入 ObjectiveActionLease 或 Budget Reservation。
- 预算暂停不结束 Assignment 或 Run，也不写完成、失败或取消终态；恢复沿用原 Assignment、agent、run_id、消息上下文和已提交 step。
- 用户提高配额后，Run 从 checkpoint 中尚未完成的阶段继续：尚未总结则进入总结，已经总结但尚未完成 Assignment 收尾调用则直接继续收尾，不重复执行普通工作或已提交的收尾调用。
- 执行系统必须持久化 Run checkpoint；进程重启后也应能从最后一个安全 step 边界恢复，而不能只依赖内存中的 coroutine。
- Provider、宿主机或组织层面的外部限额仍可独立存在，但不属于 Objective 领域模型。
