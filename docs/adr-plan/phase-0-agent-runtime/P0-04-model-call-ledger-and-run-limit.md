# P0-04：统一模型调用 attempt 与 max_steps_per_run

状态：planned

## 目标

建立一个覆盖所有实际模型请求的 Run 级调用账本，并把 `AgentOptions.max_steps` 改为 `max_steps_per_run`。阈值限制普通工作调用，但允许执行系统在不越过 ObjectiveBudget 的前提下完成有限收口；最终实际调用数可以高于配置阈值，并且必须如实记录。

## 对应决定

- ADR 0012：单 Run 模型调用阈值触发系统收口与接力。
- ADR 0014：Objective 第一版只统计 LLM 成本。
- ADR 0010：max_steps_per_run 不属于 ObjectiveBudget。

## 依赖

- 无。可以与 P0-01、P0-03 并行开发。

## 当前源码现状

- `AgentOptions.max_steps` 默认 50。
- `check_non_recoverable_limits()` 使用 `ledger.steps.current` 在普通 assistant turn 前检查。
- compaction 和 termination summary 有各自调用路径，尚无统一调用 ordinal/phase。
- Provider 内部 retry 不能完整投影为独立 RunLog attempt。
- `LLMCallStarted/Completed` 未形成所有调用阶段一致的上层契约。

## 范围

包含：配置改名、logical call/phase/attempt identity、统一调用入口、Provider retry 可见性、普通阈值与系统收口额度、RunMetrics/RunLog/Console 投影。

不包含：fault 的 retryable 分类和 backoff 策略；它们由 P4-01 实现。本任务只保证每个真实请求都能被单独观察和计数。

## 实施步骤

1. 将 `AgentOptions.max_steps` 直接改为 `max_steps_per_run`，同步 Console 配置、表单、环境解析和测试，不保留旧字段兼容。
2. 定义封闭 `ModelCallPhase`，至少包含 assistant、compaction、termination_summary、assignment_finalization、finalization_correction。不包含 `steering_outcome`。`provider_retry` 不属于 phase。
3. 每次业务目的分配稳定 `logical_call_id` 与不变 phase；每个真实 Provider 请求记录 `attempt_no`、可选 retry_reason 和全 Run 单调递增 call ordinal。
4. 在 Run ledger 中记录配置阈值、实际 attempt 总数、首次触发阈值的 call ordinal，以及每个 phase/attempt 的统计。
5. 让所有模型调用经同一个 agent-owned 调用边界产生 Started/Completed/Failed facts；compaction 和 summary 不再各自遗漏计数。
6. 拆开 Provider 隐藏 retry：retry 沿用 logical_call_id/phase，只增加 attempt_no、retry_reason、response_observed、usage 和错误事实。部分 stream 后失败必须保留已接收内容/token 证据；暂不改变是否重试的业务策略。
7. `RunLimitPolicy` 在每个新 attempt 前检查实际计数。达到阈值后拒绝 assistant、compaction 和 tool 驱动继续工作。
8. 为系统收口建立三个按实际 attempt 消费的明确额度：termination summary、Assignment finalization、finalization correction 各一次。阈值触发后，同一 phase 的 Provider retry 没有第二份越界额度。
9. 尚未实现 Assignment finalization 时，先提供通用 phase/policy 能力和单元测试；P2-05 接入实际收尾路径。
10. 非 Assignment Run 达到阈值时继续沿用返回调用方的中断语义；Objective handoff 由 P2/P3 接入。
11. 更新 RunMetrics、RunView、Trace 和 Console 调试视图，分别显示 logical_call_id、phase、attempt_no、retry_reason、阈值、触发 ordinal 和最终实际 attempt 总数。

## 主要改动位置

- `agiwo/agent/models/config.py`
- `agiwo/agent/models/run.py`
- `agiwo/agent/models/log.py`
- `agiwo/agent/llm_caller.py`
- `agiwo/agent/run_loop.py`
- `agiwo/agent/compaction.py`
- `agiwo/agent/termination/limits.py`
- `agiwo/agent/termination/summarizer.py`
- `agiwo/llm/` Provider adapters
- `console/server/models/agent_config.py`
- `console/web/src/components/agent-form.tsx`

## 测试计划

- 配置改名和默认值测试；旧 `max_steps` 输入应 fast-fail 或按模型 extra 规则被明确拒绝，不能静默使用。
- assistant、compaction、summary 使用各自 phase；Provider retry 保持原 phase、相同 logical_call_id，并增加 attempt_no/retry_reason/call ordinal。
- 达到阈值后普通调用被拒绝，三个允许的系统收口 phase 各最多一次。
- 最终实际调用数大于阈值时 metrics 与 RunLog 如实显示。
- Provider 完全无响应、部分 stream 后失败、正常完成三类 attempt 都有完整 response evidence 与 token 事实。
- direct Agent 与 Scheduler persistent root 的 max-step 既有行为不回归。

## 完成标准

- `rg "max_steps\b" agiwo console tests` 不再命中旧配置字段。
- 所有真实 Provider 请求都能关联 run_id、logical_call_id、phase、attempt_no、call ordinal 和 attempt 结果。
- `max_steps_per_run` 不出现在 ObjectiveBudget 类型中。
- 系统收口豁免只针对单 Run 阈值，不提供成本或活动时间豁免。
- SDK、Provider、Scheduler、Console 配置测试和 lint 通过。

## 风险与回退

Provider retry 展开会触及多个适配器，是本阶段风险最高的改动。应先用统一的 provider contract test 固定行为，再逐个迁移适配器。若某 Provider 无法在一个提交中完成，应延迟整个任务合并，不能让部分 Provider 的真实调用继续不可见。
