# 自动重试要求显式的安全证明

执行系统只在两个条件同时成立时自动重试：故障被结构化标记为 `retryable`，且相关 LLM 或 tool 操作具有幂等保证。重试在原 Assignment、agent 和 Run 中从最近 checkpoint 继续，不创建 handoff。`non_retryable`、`outcome_unknown`，以及配置、认证、权限或参数错误不得由基础设施自动重试；用户 pause 是独立控制命令，不属于 fault 或 retry 流程。

## Status

accepted

## Considered Options

- 所有异常统一重试：实现简单，但可能重复发送消息、写入数据或执行删除等外部副作用。
- 完全不自动重试：结果最保守，但会把普通限流和短暂网络中断都转嫁给 agent 或用户。
- 根据异常字符串匹配重试：兼容面广，但分类脆弱、难以测试，也无法证明外部副作用是否已经发生。

## Consequences

- Model 与 Tool 故障必须使用结构化 Retry Disposition，执行系统不得从自由文本推断。
- Tool 契约必须显式声明幂等属性；未声明时按不可自动重试处理。
- 每次 LLM 实际尝试都在开始前执行 `used + call_cost_ceiling <= limit` 检查，并按本次实际收到的响应独立记账；LLM 与 tool 的所有重试都写入 attempt、fault、backoff 和最终结果等可观测事实，但工具尝试当前不产生 Objective 成本。
- LLM attempt 收到至少一个有效响应数据时，按完整请求 token 与实际接收 token 计费；未收到任何响应的错误记为零成本。后续 retry 是新的 attempt，必须重新计算 ceiling 并重新做调用前成本检查。
- Assignment 收口解析失败可以安全重试，因为它不会重复业务副作用，但每次 LLM 调用仍消耗预算。
- `outcome_unknown` 必须保留操作输入和已知执行信息，不能假设失败或成功；它禁止自动重试，也禁止 handoff 给另一个 agent 猜测结果。
- 执行系统为 `outcome_unknown` 创建 普通文本 report 和 `HandoffDecision(target=user, expects_reply=true)`，令 Objective 进入 `WAITING_USER`，等待用户核验外部状态或决定后续处理。
- `non_retryable` 只表示同一操作不能自动重试，不自动等同于 Objective 失败。普通 tool failure 已形成 ToolResult 且 Run 仍可继续时，结果返回当前 agent，由它选择替代工具或策略。
- 模型认证、配置、权限等 `non_retryable` 故障若直接阻断 Run，执行系统从 fault facts 创建 普通文本 report 和 `HandoffDecision(target=user, expects_reply=true)`；不 handoff 给大概率共享同一故障的新 agent。
- 幂等且 retryable 的阻断性操作在当前 Run 内耗尽自动重试次数后，不把 Objective 判为失败，也不在原 Run 中继续尝试。
- 执行系统根据结构化 fault、attempt 历史和最后已提交产出直接构造 普通文本 report，并生成机械的 `HandoffDecision(target=agent)`；该路径不调用可能仍在故障中的模型完成总结或 Assignment 收尾调用。
- 重试耗尽 handoff 正常消耗 ObjectiveBudget 的 `handoffs`。额度可用时，ObjectiveService 原子结束当前 Assignment、消费配额并创建 peer Assignment/outbox；Scheduler 随后执行新 Run 派发。额度不足时触发 Budget Pause。
