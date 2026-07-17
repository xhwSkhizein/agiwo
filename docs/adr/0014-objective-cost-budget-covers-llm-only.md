# Objective 成本预算第一版只覆盖 LLM

ObjectiveBudget 使用 `max_llm_cost_usd` 作为一个 Objective 中全部模型调用的新调用启动阈值，不建立模型与工具共用的 CostEntry。普通 assistant、上下文压缩、Provider 重试、终止总结和 Assignment 收尾调用在收到有效响应后按实际 token 计入同一累计值；工具执行成本在第一版中不统计。

## Status

accepted

## Considered Options

- 让模型和工具统一产生 CostEntry：长期表达力最好，但为了在执行前形成硬边界，需要同时扩展 BaseTool 成本预估、ToolResult、工具批处理、RunLog 和序列化契约。
- 在模型调用前按最大输出预留成本并在结束后返还：能够严格封顶，但会引入 reserved 字段、孤儿预留和并发恢复状态。
- 调用前只检查 `used < limit`、调用后按实际响应记账：实现简单，但顺序执行可越过一次完整实际成本，且该超支缺少可事先文档化的上界。
- 调用前检查 `used + call_cost_ceiling <= limit`、调用后按实际响应记账且不预留：上界有限可计算；顺序执行在上界正确时不超支，并行超支被 ceiling × 并发数封顶。
- 完全不提供成本限制：实现最简单，但多 Agent 和重试会失去累计成本保护。

## Consequences

- 字段使用 `max_llm_cost_usd`，不使用含义过宽的 `max_cost`；API、存储和界面必须明确币种与覆盖范围。
- 当前 `StepMetrics.token_cost` 或等价的 attempt cost fact 继续作为单次 LLM attempt 成本，并以稳定 attempt identity 幂等进入 Objective 的跨 Run 聚合。
- 模型的 input、output 或 cache price 未配置时沿用当前 `0.0` 默认值，并按零成本参与计算；系统不区分“明确免费”与“价格未知”，也不因此拒绝模型调用。
- `max_llm_cost_usd` 只保证限制依据当前价格配置能够计算出的 LLM 成本。界面和调试信息必须展示使用的模型价格，避免把零成本投影误解为 Provider 一定不收费。
- 每次实际模型请求 attempt 都单独检查和计费；Provider 重试不能隐藏在一次逻辑调用中，否则 Objective 成本会被低估。
- Provider retry 沿用原 `logical_call_id` 与 phase，只增加 `attempt_no` 并记录 retry_reason；成本、预算和 max_steps_per_run 按每个实际 attempt 计算。
- 模型调用开始前计算 `call_cost_ceiling` 并检查 `used_llm_cost_usd + call_cost_ceiling <= max_llm_cost_usd`。检查成功只允许开始本次调用；不写 reserved，不在结束后结算或返还。不引入 ObjectiveActionLease。
- attempt 收到至少一个有效响应数据后计费：优先使用可归属到该 attempt 的 Provider input/output usage；没有可靠 usage 时，本地计算完整请求 token 与实际接收的输出 token。部分 stream 后报错仍按已接收内容计费，完全没有响应数据的错误记为零成本。
- 每条实际成本记录包含 attempt identity、request_tokens、accepted_output_tokens、call_cost_ceiling、价格快照、cost_usd、计算来源和 response_observed。重放相同 attempt 不能重复增加 Objective 用量。
- 顺序执行在 ceiling 正确时不使 `used` 越过 limit；并行执行可能基于同一 `used` 同时通过检查，最坏超支不超过 ceiling × 当时通过检查的并发调用数。此后系统拒绝新调用并进入 DRAINING，但不取消已经开始的在途调用。
- 进程重启只补写能够从 committed response/usage facts 证明的实际成本；没有任何已提交响应证据的 attempt 记为零成本。成本恢复不再使用 request-send marker、孤儿预留、ObjectiveActionLease 对账或保守全额收费。
- 工具调用不修改 `used_llm_cost_usd`；ToolResult 不为本设计新增成本字段。
- 若未来出现必须受预算保护的付费工具，再通过独立 ADR引入工具报价和实际记账契约，而不是提前扩展所有工具。
