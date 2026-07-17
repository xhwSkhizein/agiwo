# P4-03：实现 non-retryable 与 outcome_unknown 分流

状态：planned

## 目标

对不能自动重试的 fault 按“当前 Run 是否还能继续”和“外部副作用结果是否未知”确定处理路径。可继续的 tool failure 留给当前 agent；阻断故障和 outcome_unknown 由系统安全交给用户，不 handoff 给另一个没有新事实的 agent。

## 对应决定

- ADR 0011：non_retryable/outcome_unknown 禁止基础设施自动重试。
- ADR 0016：Outcome Unknown 必须交给用户。
- ADR 0017：Non-retryable 按 Run 可继续性分流。
- ADR 0024：故障终态仍有 Outcome。

## 依赖

- P2-06、P3-05、P4-01 已完成。

## 实施步骤

1. non_retryable + run_blocking=false 的 tool fault 形成正常 ToolResult 并提交当前 Run；不结束 Assignment、不自动 handoff。
2. 当前 agent 可以选择新参数、不同工具或放弃该计划项，但不得通过 RetryCoordinator 重复同一不可重试操作。
3. non_retryable + run_blocking=true 的模型认证/配置/权限故障由系统从 fault/committed facts 生成 report，不调用故障模型总结。
4. outcome_unknown report 必须包含操作、幂等性、输入或安全引用、attempt 时间线、已知响应、未知部分、可能副作用和核验建议。
5. 两种阻断路径都机械产生 HandoffDecision(target=user, expects_reply=true)，当前 Run/Assignment 以 INTERRUPTED Outcome 结束，Objective 进入 WAITING_USER。
6. target=user 不消费 max_handoffs，即使自动额度耗尽也必须可达。
7. 进入 WAITING_USER 时关闭 active window；不存在需要恢复的旧 Run checkpoint，因为旧 Run 已因 fault 终结。
8. 用户的语义回复完整成为新的 ObjectiveUserInput fact，并通过 related_outcome_id 关联 outcome_unknown；程序不判断其中哪些片段属于核验结果、目标或约束，也不创建额外的用户侧对象。
9. 回复后创建 fresh Assignment/new run，由后继 agent 基于 outcome report 与新增用户输入判断如何继续；不恢复旧 Run，也不让新 agent猜测未核实的外部状态。给模型时，问题与回复按普通文本渲染，不暴露内部关联结构。
10. outcome_unknown 的输入与敏感数据在 report 中可使用安全引用，但调试权限下保留完整执行证据。

## 主要改动位置

- `agiwo/objective/faults.py`
- `agiwo/objective/service.py`
- `agiwo/objective/projection.py`
- `agiwo/agent/tool_executor.py`
- Provider error adapters
- `tests/objective/test_fault_boundaries.py`

## 测试计划

- nonblocking tool fault 留在同 Run 并允许下一 assistant turn。
- blocking auth/config fault 无额外模型调用，进入 WAITING_USER。
- outcome_unknown 绝不自动 retry 或 target=agent。
- max_handoffs=0 时 user boundary 仍成功。
- 用户核验回复完整形成一条 ObjectiveUserInput，且不会产生第二份用户侧事实。
- 只有简短回复时，下一 Assignment 的模型上下文同时包含所关联的 outcome 问题与原始用户回复，但不包含 ObjectiveUserInput JSON 或内部关联字段。
- fresh Assignment/new run，旧 Run 不恢复。

## 完成标准

- `run_blocking` 是结构化字段，不由错误文本推断。
- outcome_unknown 没有假定成功/失败的默认分支。
- 普通 tool failure 不被过早升级给用户。
- user 安全出口不受自动接力额度限制。
- fault boundary tests 与 lint 通过。

## 风险与回退

如果 Tool adapter 无法判断请求是否发出，必须保守使用 outcome_unknown；不能为了减少用户中断而假定失败并重复有副作用的操作。
