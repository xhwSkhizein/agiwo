# P0-03：贯通 UserMessage 来源与前缀稳定性

状态：planned

## 目标

在不修改 `Agent.start/run/run_stream` 公开参数的前提下，让每条规范化 UserMessage 明确记录它是否由真实用户提供。后续 Assignment Input、计划提醒和 fork notice 都可以继续使用 `role=user` 驱动模型，却不会被 Objective 规则或 Console 误认成 ObjectiveUserInput。

## 对应决定

- ADR 0042：UserInput 记录真实用户来源。
- ADR 0036：Assignment 复用 Session agent identity，并保持前缀稳定。
- ADR 0026：模板 notice 是系统输入，不是领域真相。

## 依赖

- 无。提交 `6222212` 已作为跨 Run compaction 回归基线。

## 当前源码现状

- `UserMessage` 只有 `content` 与 `context`。
- string 和 ContentPart list 会在 `UserMessage.from_value()` 中规范化。
- StepView、RunLog serialization、Scheduler pending event 和 Console transport 都会传递 UserInput。
- `6222212` 已修复跨 Run 使用 `MessagesRebuilt` 的恢复 bug；本任务只补回归测试，不重复修改该逻辑。

## 范围

包含：canonical model、序列化、StepView/RunLog、Scheduler mailbox、Console/API DTO、普通聊天展示判定和前缀回归测试。

不包含：ObjectiveUserInput 的领域存储、投影和上下文去重规则，它们由 P1/P2 实现。

## 实施步骤

1. 为 `UserMessage` 增加 `is_user_provided: bool = True`，并在 `to_dict/from_dict/serialize/deserialize/storage/transport` 全路径保留。
2. string、ContentPart list 和现有 Console/渠道输入默认规范化为 true，保持已有调用方行为。
3. 提供明确的内部构造方式生成 false 消息，但不向外部用户输入 API 暴露可伪造来源的字段。
4. 让 `StepView.user()`、CommittedStep、RunLog codec 和 Run replay 保留该值；旧开发数据缺少字段时按 true 读取。
5. Scheduler `PendingEvent.USER_HINT`、mailbox message、fork notice 等结构化路径不得通过 `extract_text()` 丢失来源。
6. Console API/视图 DTO 携带来源；普通聊天组件只把 true 显示为用户气泡，false 只在开发 trace 或相应系统视图中显示。
7. compaction、MessagesRebuilt 和 context assembly 保持原 message 顺序与内容，不因为来源标记重排消息。
8. 增加跨 Run 回归：第一次 Run 产生 compaction，第二次 Run 使用同一 agent/session，确认恢复摘要、历史顺序和来源标记都正确。
9. 增加精确 prompt fixture，证明同一历史前缀不会因新 Run 而发生无关改写。

## 主要改动位置

- `agiwo/agent/models/input.py`
- `agiwo/agent/models/step.py`
- `agiwo/agent/models/log.py`
- `agiwo/agent/storage/serialization.py`
- `agiwo/agent/prompt.py`
- `agiwo/agent/run_bootstrap.py`
- `agiwo/scheduler/models.py`
- `agiwo/scheduler/formatting.py`
- `console/server/response_serialization.py`
- `console/server/models/view.py`
- Console session conversation components

## 测试计划

- `UserMessage` 的 string、parts、structured、storage 和 transport round-trip。
- StepView -> RunLog -> SQLite -> StepView 来源不丢失。
- Scheduler steer/mailbox round-trip 保留 ContentPart、ChannelContext 与来源。
- 普通用户输入固定为 true；内部 notice 为 false。
- 普通聊天不显示 false 输入为用户气泡，开发视图仍可查看。
- `test_prepare_run_context_restores_latest_rebuilt_messages_after_compaction` 保持通过并扩充跨 Run 来源断言。

## 完成标准

- Agent 公开执行 API 的签名测试无变化。
- 任意持久化/重放路径都能区分 true 与 false。
- 真实用户入口不能提交 `is_user_provided=false`。
- 消息来源不参与相关性或内容判断，只提供 provenance。
- SDK、Scheduler、Console 受影响测试和 lint 通过。

## 风险与回退

最主要风险是某个 codec 把 false 静默恢复为默认 true。测试必须覆盖 memory 与 SQLite。若失败，回退整个字段贯通提交，不允许只在 Console 或 Objective 层维护第二份来源标记。
