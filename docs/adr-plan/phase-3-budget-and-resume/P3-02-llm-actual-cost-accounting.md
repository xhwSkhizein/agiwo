# P3-02：实现 LLM 实际成本检查与记账

状态：planned

## 目标

让 Objective 中每个真实模型 attempt 在开始前用「已用成本 + 本次调用成本上界」做启动检查，并在收到响应后按实际 token 幂等记账。系统不预留、不结算、不返还。顺序执行在上界正确时不超支；并行执行的最坏超支被「单次调用成本上界 × 当时通过检查的并发调用数」封顶。

## 对应决定

- ADR 0010：`max_llm_cost_usd` 是带可文档化超支上界的新调用启动边界。
- ADR 0014：第一版只统计 LLM 的实际响应成本；调用前检查取代调用前预留。
- ADR 0034、0044：通过 Agent hook 接入，ObjectiveService 拥有预算；不引入 ObjectiveActionLease。

## 依赖

- P0-04、P3-01 已完成。

## 当前源码现状

- `StepMetrics.token_cost` 已根据模型价格计算完成调用的成本。
- Agent hook 有 BEFORE_LLM/AFTER_LLM phase，但尚无 Objective 级已用成本检查。
- `max_run_cost` 是单 Run 后检查；direct Agent 仍需要保持现有行为。

## 范围

包含：`call_cost_ceiling` 计算、调用前成本检查、响应 token 证据、实际成本 fact、并行超支语义、失败和重启补账。

不包含：工具成本、Retry Disposition、DRAINING barrier；P4-01 增加重试策略，P3-05 实现状态门禁与 Run barrier。

## 实施步骤

1. 删除 `RunUsageLease`、`ObjectiveActionLease`、reserved_usd、reserved_llm_cost_usd、预留返还和保守全额结算模型。不引入任何动作租约替代品。
2. 在 Provider 请求确定后计算 `call_cost_ceiling`：完整请求 token × 输入价 + 本次 `max_output_tokens` × 输出价（含适用 cache 计价），使用当时价格快照；ceiling 写入调试/attempt 事实，不写入 ObjectiveBudget.reserved。
3. objective-managed Run 的 BEFORE_LLM 检查 Objective 仍允许推进，且 `used_llm_cost_usd + call_cost_ceiling <= max_llm_cost_usd`；任一条件失败都不调用 Provider，成本检查失败触发预算暂停路径。没有 objective_id 的 direct Run 不启用 Objective 成本门禁；带 objective_id 却缺少 admission/budget adapter 时 fail closed。
4. child Run 从 root 执行请求继承 objective_id、assignment_id 和 budget/admission adapter；不能因委派绕过成本检查。
5. 记录 request token 数、model/price snapshot、logical_call_id、phase、attempt_no、call ordinal 与 call_cost_ceiling。优先使用可归属到该 attempt 的 Provider input usage，否则使用现有 tokenizer/estimator。
6. adapter 收到第一个有效 response chunk 时标记 response_observed；累计模型实际返回且被 runtime 接受的输出 token。部分 stream 后报错保留已经接收的内容与计数。
7. attempt 完成或失败后写稳定 `ModelAttemptCostRecorded`（或等价 RunLog fact）：request_tokens、accepted_output_tokens、call_cost_ceiling、cost_usd、source、response_observed 和价格快照。Provider usage 可靠时优先使用；否则根据请求与已接收输出本地计算。
8. 完全没有收到响应数据的错误写零成本记录；不根据“请求可能已经发送”推测收费，也不把 ceiling 记成实际成本。
9. ObjectiveService 按稳定 run_id/logical_call_id/attempt_no 幂等追加 `LlmCostConsumed`。同一 attempt 重放不能重复增加 used；RunLog 已有成本而 Objective fact 缺失时由 reconciler 补写。
10. 当下一次调用前检查不再满足 `used + ceiling <= limit` 时，Objective 进入 DRAINING(reason=budget)；已经开始的在途调用允许完成并继续追加实际成本。
11. 顺序执行在 ceiling 正确时最终 `used` 不超过 limit。并行执行可能基于同一 `used` 同时通过检查，最坏超支不超过 ceiling × 当时通过检查的并发调用数；文档与 Console 必须展示该上界。
12. 每个 Provider retry 保持 logical_call_id/phase，增加 attempt_no/retry_reason，并作为新 attempt 重新计算 ceiling、执行调用前检查和记录自身实际成本。
13. assistant、compaction、termination summary、finalization、correction 的每个实际 attempt 全部进入同一调用边界。单 Run 收口额度不豁免 Objective 成本启动检查。
14. 价格缺失沿用 0.0 并按零成本记账；调试投影仍展示 input/output/cache price snapshot 与 ceiling，避免误解为 Provider 一定免费。
15. 重启只根据 committed response/usage/cost facts 补账。没有 committed response 证据的孤儿 LLM attempt 按零成本结束或重试；具有外部业务副作用的 tool 继续按 P4 的 outcome_unknown 规则处理。
16. `max_run_cost` 的现有单 Run 行为保持 direct Agent 可用；objective-managed Run 还必须经过本任务的 Objective 上界检查，测试明确二者的触发优先级。

## 主要改动位置

- `agiwo/objective/budget.py`
- `agiwo/objective/log.py`
- `agiwo/objective/service.py`
- `agiwo/agent/hooks.py`
- `agiwo/agent/llm_caller.py`
- `agiwo/agent/models/log.py`
- `agiwo/llm/` Provider adapters 与现有 token/cost resolver

## 测试计划

- `used + ceiling <= limit` 时允许调用；超出时 Provider 完全不执行并触发暂停路径。
- ceiling 由 request tokens、max_output_tokens 与价格快照可复现计算。
- 正常 stream、部分 stream 后失败、完全无响应失败三类成本分别按实际响应、已接收部分和零计算；ceiling 永不计入 used。
- 顺序调用在正确 ceiling 下不使 used 越过 limit；下一调用因检查失败被拒绝。
- N 个并发 attempt 基于同一 used 同时通过后均可完成；最终超支 ≤ sum(ceilings of those N)，且触发 DRAINING 后没有第 N+1 个新调用开始。
- 相同 attempt 成本重放不重复累计；RunLog 成本已提交但 Objective fact 缺失时可补写。
- Provider retry 使用相同 logical_call_id/phase 和新 attempt_no，独立检查并计费。
- direct Agent、Objective root、Objective child 三条路径；child 继承 adapter，带 objective_id 但缺 adapter 时 fail closed。
- `rg "ObjectiveActionLease|RunUsageLease|reserved_llm_cost|Budget Reservation" agiwo console tests docs` 无实现命中（文档 Avoid 提及除外）。

## 完成标准

- ObjectiveBudget 只有 limit/used，没有 reserved；所有实际成本都能追溯到唯一模型 attempt 和响应证据。
- 完全无响应错误不增加成本；不存在预留、返还、ActionLease 或未知状态保守收费分支。
- 并行超支上界可由 ceiling × 并发通过检查的调用数解释，并出现在用户/调试文档中。
- 工具执行不修改 LLM cost 字段，ToolResult 无新成本字段。
- agent hook 与 Objective 模块依赖方向合法，相关测试和 lint 通过。

## 风险与回退

若 ceiling 低估真实成本（例如 Provider 忽略 max_output_tokens），顺序路径也可能超支；实现必须使用与实际请求一致的 max_output_tokens，并在调试事实中保留 ceiling 以便审计。不得为“看起来严格封顶”重新引入预留或 ObjectiveActionLease。
