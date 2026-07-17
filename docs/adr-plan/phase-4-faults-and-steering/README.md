# P4：故障与运行中用户输入

本阶段补齐正常主链之外的分流。核心原则是：基础设施只依据结构化 fault 和幂等证明行动；已经花费模型或工具调用得到的信息必须通过 Outcome 离开**真正结束**的 Assignment。运行中用户输入不结束 Assignment，而是注入同一 root Run。

## 入口条件

- P3 的实际成本检查、DRAINING、checkpoint 和可恢复中断恢复已经可用（P4-04 注入路径可不依赖 DRAINING，但仍建议 P3 门禁已存在以免状态歧义）。
- 系统能够在不依赖故障模型的情况下，从 committed facts 构造普通文本 report 和机械 Decision。

## 任务顺序

| ID | 任务 | 依赖 |
| --- | --- | --- |
| P4-01 | 重试契约与协调器 | P0-04、P3-02 |
| P4-02 | 重试耗尽接力 | P2-06、P3-01、P4-01 |
| P4-03 | non-retryable 与 outcome_unknown | P2-06、P3-05、P4-01 |
| P4-04 | 运行中用户输入注入 | P2-04、P2-02 |

P4-02、P4-03、P4-04 可以在共同依赖满足后并行，最后用状态矩阵测试汇合。

## 阶段出口

- 自动重试只发生在 `retryable + idempotent`，每个 attempt 独立计数、检查预算、按实际响应记账并记录 backoff。
- retry exhaustion 创建系统 report 和 fresh agent Assignment，不调用故障模型总结。
- run-blocking non-retryable 与 outcome_unknown 始终能到达用户边界，不受 max_handoffs 阻断。
- 运行中用户输入：写 ObjectiveUserInput → 向同一 root Run 注入 `is_user_provided=false` 系统提示型消息；无 DRAINING、无 Outcome、无新 Assignment。
- 用户暂停/停止与预算暂停走可恢复中断（P3），不与运行中输入混淆；普通停止不得复用 cancel→FAILED。
