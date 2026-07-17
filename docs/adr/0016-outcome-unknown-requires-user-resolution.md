# Outcome Unknown 必须交给用户处理

当一个可能产生外部副作用的操作已经发出，但系统无法确认结果时，执行系统将其标记为 `outcome_unknown`。该状态禁止自动重试，也不 handoff 给新 agent 推测成功或失败。系统从持久化执行事实构造 普通文本 report，并机械地产生 `HandoffDecision(target=user, expects_reply=true)`；Objective 进入 `WAITING_USER`，直到用户核验外部状态或明确决定后续动作。

## Status

accepted

## Considered Options

- 假定失败并自动重试：可提高表面成功率，但可能重复付款、发送消息、写入或删除数据。
- 假定成功并继续：避免重复副作用，但可能在实际失败时基于不存在的结果继续执行。
- handoff 给全新 agent 判断：能够获得新上下文，但新 agent 没有额外的外部事实，语义推理不能消除执行结果的不确定性。

## Consequences

- Outcome report 至少包含操作名称、幂等性声明、完整输入或安全引用、attempt 时间线、已知响应、未知部分、可能副作用和核验建议。
- 当前 Run 和 Assignment 以 interrupted outcome 结束；系统不调用故障模型或工具生成总结，也不伪造普通 Assignment Decision。
- `HandoffDecision(target=user, expects_reply=true)` 是执行安全策略产生的机械决定，Objective 进入 `WAITING_USER` 并关闭当前活动窗口。
- 该用户 handoff 不消耗 `max_handoffs`；即使自动接力额度已经耗尽，系统仍必须能够把安全问题交给用户。
- 用户的语义回复原样记录为新的 ObjectiveUserInput，并通过 `related_outcome_id` 关联这次 outcome_unknown；程序不解析其中哪些片段属于目标、约束或核验结果。
- 用户回复后由新的 Assignment 根据 ObjectiveView、outcome report 和新增 ObjectiveUserInput 决定如何继续，不恢复已经产生未知副作用的旧 Run。模型上下文把问题与回复渲染为普通文本，不暴露内部关联对象。
