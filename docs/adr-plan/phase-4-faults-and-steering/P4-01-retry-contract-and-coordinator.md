# P4-01：建立结构化重试契约与协调器

状态：done

## 目标

让执行系统只在 fault 明确为 retryable、且操作具有幂等保证时自动重试。每个真实 attempt 都在同一 Run 内独立记录、在开始前检查预算并按实际响应记账；错误字符串不参与控制决策。

## 对应决定

- ADR 0011：自动重试要求显式安全证明。
- ADR 0014：每个 LLM attempt 独立计费。
- ADR 0010：重试不能越过 ObjectiveBudget。

## 依赖

- P0-04、P3-02 已完成。

## 当前源码现状

- Provider 和工具异常尚未统一为 Retry Disposition。
- `BaseTool` 没有足以证明自动重试安全的幂等声明。
- Scheduler `TaskGuard` 只负责 spawn/wake 限制，不应承担操作重试。

## 范围

包含：fault model、Tool 幂等契约、Agent-owned RetryCoordinator、attempt/backoff/checkpoint 和成本接入。

不包含：重试耗尽后的 fresh Assignment，以及 non-retryable/outcome_unknown 后续路径。

## 实施步骤

1. 定义结构化 ExecutionFault：operation、disposition（retryable/non_retryable/outcome_unknown）、run_blocking、response_observed、external_effect_may_have_started、known response、side_effect risk、provider/tool code 和 provenance。
2. 扩展 Tool contract 的幂等声明，至少区分 guaranteed、conditional、not_idempotent/unknown；conditional 必须提供稳定 idempotency key 或 gate 证明。
3. Model/provider adapter 将限流、暂时网络失败、认证、参数、发送状态未知等映射为结构化 fault，不用上层匹配异常文本。
4. 在 `agiwo.agent` 建立窄 RetryCoordinator，位于模型/tool attempt 执行边界；它不属于 Scheduler，也不替 agent 选择替代业务策略。
5. 自动 retry 条件固定为 disposition=retryable 且幂等证明成立；其他组合立即返回结构化结果。
6. attempt、fault、backoff、checkpoint 和最终结果写 first-class RunLog；backoff 使用可注入 clock/sleeper 便于测试。
7. 每次 LLM retry 沿用原 logical_call_id 与 phase，增加 attempt_no/retry_reason/call ordinal；作为新 attempt 重新计算 call_cost_ceiling、执行 `used + ceiling <= limit` 检查，并按自身实际响应独立记账。
8. RunLimitPolicy 在每个 retry attempt 前重新检查。阈值后的系统收口 phase 没有额外 Provider retry 特权：同一 phase 的唯一越界 attempt 已用完时停止 retry。
9. tool retry 当前不产生 Objective 成本，但必须记录 attempt 和副作用安全依据。
10. 可能产生外部业务副作用的 tool adapter 必须在调用前提交 `ExternalEffectMayHaveStarted`；崩溃产生的保守假阳性可以交给用户核验，但不得在副作用已经发生后缺少该事实。LLM 不用该 marker 推断费用。
11. retry 仍在原 Assignment/agent/run_id 中进行，不创建 handoff，也不消耗 max_handoffs。
12. 用户 pause 或 Objective DRAINING 能在 attempt 间阻止下一次 retry；已经获准开始的 attempt 按安全边界完成。
13. 配置最大 attempt/backoff 属于 Agent 执行配置，不纳入 ObjectiveBudget；agent 模型无权提高。

## 主要改动位置

- `agiwo/agent/retry/`（新建或职责清楚的单模块）
- `agiwo/agent/llm_caller.py`
- `agiwo/agent/tool_executor.py`
- `agiwo/agent/models/log.py`
- `agiwo/llm/` Provider adapters
- `agiwo/tool/base.py`
- builtin/custom tool declarations
- `tests/agent/test_retry.py`
- Provider/tool contract tests

## 测试计划

- retryable+idempotent 自动重试，其余组合不重试。
- conditional idempotency key 缺失时拒绝自动 retry。
- LLM 正常响应、部分 stream 后失败、完全无响应三种 attempt 分别按实际、部分和零成本记账。
- 每个 LLM attempt 独立 attempt_no、ordinal、response evidence 和 cost，但 logical_call_id/phase 保持不变。
- 外部副作用 tool 在 marker 前崩溃可按未开始处理，marker 后无结果则进入幂等重试或 outcome_unknown。
- DRAINING 在 attempt 之间停止后续 retry。
- TaskGuard 测试保持只覆盖 spawn/wake，不新增 retry 逻辑。

## 完成标准

- 上层没有根据错误 message/exception class name 猜 Retry Disposition 的分支。
- 未声明幂等性的工具默认不可自动重试。
- RetryCoordinator 不导入 Objective；预算与可推进门禁通过 hook 契约接入；LLM 使用调用前上界检查，不引入 ObjectiveActionLease。
- 所有 attempt 可重放且不隐藏 Provider retry。
- agent/llm/tool tests 与 lint 通过。

## 风险与回退

不要把“工具调用返回失败”一律变成基础设施 retry。普通 ToolResult failure 仍应交给当前 agent 选择替代策略；RetryCoordinator 只处理同一操作的安全重复。
