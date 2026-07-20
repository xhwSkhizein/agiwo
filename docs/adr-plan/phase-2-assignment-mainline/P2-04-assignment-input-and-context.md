# P2-04：装配 Assignment Input 与前缀安全上下文

状态：**done**（2026-07-18）

## 目标

把 ObjectiveView、最近 Outcome 和本次触发事实渲染为 Assignment Input，并与同一 Session 的候选历史共同装配为新 root Run 上下文。选择历史时以前缀缓存命中率为首要原则，不按 Assignment 边界默认隔离，也不把聊天历史当作 Objective 权威状态。

## 对应决定

- ADR 0026：Assignment Input 是持久化输入快照。
- ADR 0035、0036：Session 历史连续、相关性选择和稳定 agent identity。
- ADR 0037：ObjectiveView 同时提供 current_goal 与权威依据。
- ADR 0042：系统输入 provenance。

## 依赖

- P0-03、P2-01、P2-02 已完成。

## 范围

包含：ObjectiveView input、trigger 选择、模板渲染、Run 边界 notice、历史装配、Context Optimization 接入和调试 facts。

不包含：finalization（P2-05）与运行中用户输入注入（P4-04）。

## 实施步骤

1. 构建类型化 ObjectiveView：current_goal、按 ObjectiveLog 顺序排列的全部 ObjectiveUserInput 及外置授权、活动 Contributions 及 annotations、Budget、Artifact refs、最近完整 Outcome 和触发信息。结构化 input id 与关联只供系统存储、投影和装配使用。
2. 不把全量 RunLog、收尾调用 debug payload 或 AgentState snapshot 复制进 ObjectiveView。
3. 根据 Assignment kind 选择 trigger：intake 以最新用户输入为锚点；work/verification 以 Decision、最近 Outcome 或候选交付为锚点。
4. 校验活动 Objective 的每条未外置/已外置用户输入都能在候选 Session 历史中按 `input_id` 找到对应表示（未外置为 canonical `UserMessage`，已外置为 path/summary）。全部命中则原样复用稳定前缀；**禁止**因缺口向消息中段插入用户原文。若存在缺口，提交结构化不变量故障并 fail closed，不派发不可执行 Assignment。随后渲染系统生成的 Assignment Input，标记 `is_user_provided=false`，**只追加在历史尾部**。模板不承载用户原文或内部外置对象。
5. 在新 Run 历史尾部追加短边界说明：当前 run_id 的 RunPlan 尚未建立，旧 `update_plan` 仅是已结束 Run 的历史，需要延续的责任来自 Outcome.carry_forward。
6. 在上下文未超限且没有实质冲突时，保持已提交历史的原顺序和内容，直接追加新输入；不得为了“干净”重排或重述旧消息。
7. 只有长度、冲突或噪声确实要求改变输入视图时，才调用现有 Agent Context Optimization/正式 compaction。它优先压缩 assistant/tool 历史，不得摘要、改写或删除 ObjectiveUserInput；输入、输出、依据、usage 和 MessagesRebuilt 全部持久化。
8. 历史相关性以 Objective/trigger 为锚点，但程序只执行必保字段、上下文上限和结构校验，不用关键词规则假装语义判断。
9. 必保内容：全部未外置 ObjectiveUserInput 原文、已外置输入的 path 与 summary、current_goal 与来源、当前 Assignment 职责、最近 Outcome。若这些内容仍超过物理上限，在创建/派发下一 Assignment 前提交 ContextCapacityExceeded 并进入 WAITING_USER；只有用户 externalize command 可以改变输入表示，禁止静默总结。
10. Assignment finalization prompt/structured output 不进入普通历史；RunLog 可完整查询。
11. 若 Session 来自 fork，首个 Objective root Run 在尾部追加一次 `fork_context_summary` system notice，使用后保留 provenance，不复制源 Objective facts。

## 主要改动位置

- `agiwo/objective/service.py`
- `agiwo/objective/input.py`
- `agiwo/objective/templates.py`
- `agiwo/agent/run_bootstrap.py`
- `agiwo/agent/prompt.py`
- `agiwo/agent/compaction.py`
- `agiwo/agent/models/log.py`
- Console fork/session runtime adapter

## 测试计划

- 短历史下两个 Assignment 的模型输入共享完全相同的最长前缀，只在尾部追加边界与新输入。
- 历史旧 update_plan 保留，但当前 RunPlan 为空且 guard 只读当前 run_id。
- 超限时 formal compaction 可重建，原 RunLog 不删除，调试 facts 完整。
- 三种 kind 的 ObjectiveView/trigger/模板输入快照。
- false Assignment Input 不创建 ObjectiveUserInput 或普通用户气泡。
- 同一 input_id 已在历史时不重复注入；历史缺少某条 ObjectiveUserInput 对应消息时 fail closed，不中段补回。
- 运行中注入产生的 false user 不在新 Assignment 边界被误当成新的用户气泡来源；权威用户事实仍是 ObjectiveUserInput。
- compaction 优先处理 assistant/tool；用户输入表示单独超限时进入可见的 WAITING_USER/外置授权路径。
- ContextCapacityExceeded 不创建不可执行 outbox；用户授权外置后重新检查，通过才派发，未通过继续 WAITING_USER。
- 外置后：模型上下文仅 path + summary；agent 用既有读文件工具（或等价 workspace 读）能打开该相对 path 并读回与 ObjectiveUserInput 一致的原文；禁止只留无法解析的空引用或错误根路径。
- fork summary 只在新 Session 第一个 Objective 使用一次，不复制源 Objective。

## 完成标准

- Assignment 边界本身不触发默认历史隔离或重写。
- Objective 权威字段不依赖解析聊天历史。
- Context Optimization 失败时不丢必保字段，并产生结构化失败事实。
- 收尾 debug payload 不进入普通模型上下文。
- 已外置输入的 Artifact path 对 objective-managed root Run 可按需读取（ADR 0045 / R4）。
- prefix fixture、agent bootstrap 和 objective input tests 通过。

## 风险与回退

“相关性选择”容易演变为每个 Assignment 都重新总结历史，这会破坏前缀缓存。实现默认路径必须是复用原前缀；只有上下文约束真正触发时才重建，并由测试比较精确 message 序列。
