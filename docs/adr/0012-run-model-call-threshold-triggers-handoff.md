# 单 Run 模型调用阈值触发系统收口与接力

`max_steps_per_run` 是一个 Run 的正常模型调用阈值，用来截断 agent 的极端循环，而不是 Objective 的总预算或不可超越的物理调用上限。Run 内发生的每次实际模型调用都计数，包括主 assistant turn、上下文压缩、Provider 重试、终止总结和 Assignment 收口。达到阈值后，执行系统停止新的普通工作调用，但仍允许有限的系统收口调用；Assignment Run 完成总结和结构化收口后产生 `target=agent` 的 handoff，由同一 Session agent identity 下的新 Assignment Run 接棒，Objective 不因该阈值本身暂停。

## Status

accepted

## Considered Options

- 把调用次数作为 ObjectiveBudget 总量：能够限制整个 Objective 的模型调用，但会把 Objective 级资源额度与单 Run 防循环保险丝混在一起。
- 达到阈值后暂停 Objective 等待用户：最保守，但普通的局部循环会不必要地打断整个动态 workflow。
- 把阈值作为任何情况下都不能超过的绝对上限：边界直观，但达到上限后无法生成总结、Artifact 和结构化接力结果。

## Consequences

- `AgentOptions.max_steps` 应直接重命名为 `AgentOptions.max_steps_per_run`；RunLimitPolicy 消费该配置执行单 Run 检查，但不另行拥有一份配置来源。该字段不属于 ObjectiveBudget。
- 开发者可以在不同的静态 Agent 配置中设置不同上限；执行系统在创建 Run 时注入最终值。动态 workflow 中的模型和由模型构建的执行方案只能读取该值，不能提高、降低或覆盖它。
- 动态创建的 agent 若没有对应的静态专用配置，使用执行系统给定的默认 `max_steps_per_run`，不能从前一 agent 的输出中接受该字段。
- 每个实际模型请求 attempt 都增加同一 Run 的模型调用计数。现有隐藏在 Provider 内部的重试也必须进入统一计数与 RunLog，不能继续成为不可见调用。
- 模型调用采用两层 identity：`logical_call_id + phase` 表达一次业务目的，`attempt_no + retry_reason` 表达该目的下的实际 Provider 请求。phase 至少区分普通 assistant、compaction、termination summary、Assignment 收尾与收尾纠正；Provider retry 不是独立 phase，沿用原 logical call 的 phase。不保留 `steering_outcome` phase。
- 每个实际 Provider attempt 都增加同一 Run 的模型调用计数、在开始前检查 Objective 已用成本并记录实际响应成本。RunLimitPolicy 按 attempt 计数，同时按不变的 phase 判断它是否属于普通工作或系统收口。
- 达到阈值后，普通 assistant、compaction 和 tool 驱动的继续工作不得再开始；无 tools 的系统总结与 Assignment 收口可以越过阈值执行，并继续增加实际调用计数。
- Run 必须同时保留配置阈值、首次触发阈值时的调用序号和最终实际调用总数，不能把越界收口调用伪装成未计费或未发生。
- 系统收尾阶段最多允许三个越过阈值的模型调用：termination summary 一次，Assignment 收尾调用一次，收尾结构错误时纠正一次；这些调用全部进入同一 Run 的实际调用总数。
- 上述三个越界额度按实际 attempt 消费，而不是按 logical call 名义消费。某个收口 phase 的 Provider retry 仍是额外 attempt；阈值已经触发且该 phase 的唯一越界额度已用完时不得再 retry。
- 若一次收尾纠正后仍无法解析，执行系统不再调用模型，直接生成最小的 `HandoffDecision(target=agent)`，并把解析故障和最后一份可用报告写入 AssignmentOutcome。
- 越过 `max_steps_per_run` 只豁免单 Run 的正常调用阈值，不豁免 ObjectiveBudget。总结、收尾或纠正 attempt 开始前若 Objective 已用成本或活动时间达到阈值，系统仍必须暂停；尚未达到则允许调用并按实际响应记账。
- 若 ObjectiveBudget 在收尾前耗尽，整个 Objective 立即预算暂停，当前 Run 保持未结束；checkpoint 记录 termination summary、Assignment 收尾调用或收尾纠正中下一项尚未完成的阶段。
- 若达到 `max_steps_per_run` 时 root Run 的 `RunPlan` 仍有未完成项，Run 和 Assignment 记为 `INTERRUPTED`，系统 handoff 给后继 Agent；AssignmentOutcome 必须包含剩余计划项、当前状态和 `carry_forward` 信息。
- 用户提高 ObjectiveBudget 后恢复同一个 Run，并从 checkpoint 指定的收口阶段继续；用户不提高配额时不再为不完整结果生成总结。
- Assignment Run 的阈值收口必须保存总结为 普通文本 report，并产生系统保证的 `HandoffDecision(target=agent)`；模型可以补充 reason、Artifact 和 ObjectiveContribution，但不能把机械阈值事件改成继续当前 Run。
- 阈值收口完成后当前 Run 进入 `INTERRUPTED`，不是 FAILED；对应 Assignment 以带 handoff 的 Outcome 结束。
- 该 handoff 正常消耗 ObjectiveBudget 的 `handoffs`。若 handoff、llm_cost_usd 或 active_seconds 的 ObjectiveBudget 已耗尽，ObjectiveBudget 的硬边界优先，整个 Objective 转为预算暂停并向用户报告。
- 接棒方遵循既有 handoff 规则：创建 peer Assignment 和新 Run，继续使用 Session 可见历史并接收 ObjectiveView 与最近一次 AssignmentOutcome，但不复活发生循环的旧 Run。
- 非 Assignment 触发的 Run 没有 Objective handoff 语义；达到阈值后完成有限总结，并把中断结果返回原调用方。
