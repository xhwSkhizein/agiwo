# 验收不通过时创建全新工作 Assignment

Verifier 判断候选结果不可交付时，不重新打开原 Assignment 或旧 Run，而是以 `target=agent` 提交 handoff。Verifier 的报告进入最新 AssignmentOutcome，执行系统据此创建同一 Objective 下的全新 peer Assignment；Session persistent agent 使用新的 Run、ObjectiveView 和该验收 Outcome 自主决定返工方式。

## Status

accepted

## Considered Options

- 恢复原 Run 继续修改：可以少建一次 Run，但会抹平验收与返工的责任边界，并使旧 Run 的终态失真。
- 回滚原 Assignment 后重跑：状态看似简单，但会抹去已经发生的验收与返工事实，使复盘和预算计数失真。
- 由 Verifier 指定具体返工 agent 或 pattern：减少一次判断，但让 Verifier 承担超出验收职责的实现选择。

## Consequences

- 原工作 Assignment 与 Verifier Assignment 都保持完成；返工通过新的 Assignment 追加到 Objective 历史。
- 每次验收不通过同时增加 Objective 的 verification attempt 与 handoff 计数。
- Verifier 报告必须明确用户原始输入中尚未满足的目标或约束、证据缺口和不可交付理由，并作为返工 Run 的最近一次完整 AssignmentOutcome。
- 验收失败本身不是执行故障；它消耗语义返工预算，而不是基础设施重试预算。
