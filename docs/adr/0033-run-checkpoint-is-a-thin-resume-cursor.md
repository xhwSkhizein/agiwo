# RunCheckpoint 以消息边界恢复，并钉住配置快照

可恢复中断后的恢复，以 RunLog 重建的消息列表最后一项形态决定下一步（ADR 0029），而不是维护完整的 resume_phase 状态机。系统保存 `last_committed_sequence`，以及该 Run/Assignment 创建或进入暂停时所钉住的 AgentConfig 与 Assignment 模板快照（或其可重建引用与 content hash）。消息、step、tool result 继续以 RunLog 为真相。

## Status

accepted

## Considered Options

- 序列化完整 RunContext：恢复快，但脆弱且难跨进程。
- 厚 checkpoint 含独立 phase 状态机：表达力强，但与 loop 自然边界重复，MVP 过重。
- 以消息末项恢复，并用**钉住的配置/模板快照**重建 runtime（本决定）：与 loop 自然边界一致，且暂停期间的后台编辑不影响恢复契约。
- hash 与**当前** Registry 不一致则拒绝 resume：能防止「悄悄换 prompt」，但会把合法的后台模板编辑变成不可恢复，违背「编辑只影响新 Assignment」。

## Consequences

- checkpoint 不复制完整 messages；从 RunLog 重放到 last_committed_sequence。
- 恢复下一步由 ADR 0029 的三条消息末项规则决定。
- resume **必须**使用 checkpoint / Assignment 已钉住的 AgentConfig 与模板快照重建，不得改用 resume 时刻 Registry 中的最新默认配置。
- 暂停或运行期间编辑默认 AgentConfig / 模板只影响之后**新创建**的 Assignment；已 PAUSED 的 Run 恢复不受影响。
- content hash 用于校验「钉住的快照内容未被损坏或换错」；与当前 live Registry 的 hash 不同是预期现象，不得因此拒绝 resume。
- 若钉住的快照无法加载（存储缺失、hash 自检失败），产生结构化恢复故障并保持 PAUSED，不静默用新契约继续。
- ObjectiveLog 只存 checkpoint 引用，不复制 Run 内容。
- 本 ADR 简化此前「必须持久化精细 resume_phase」的表述，但不禁止实现层保留最小 phase 提示作为优化。
