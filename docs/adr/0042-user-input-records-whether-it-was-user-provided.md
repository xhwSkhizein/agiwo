# UserInput 记录是否由真实用户提供

Objective 集成不为 `Agent.start/run` 增加 Assignment 专用参数。真实用户输入、模板渲染的 Assignment Input 和 agent loop 内的计划提醒都继续通过现有 `user_input` 进入 Agent；规范化后的 `UserMessage` 增加 `is_user_provided: bool = true`，用它区分真实用户消息与系统生成的 `role=user` 输入。

## Status

accepted

## Considered Options

- 给 `Agent.start/run` 增加 `system_notice` 或 `assignment_input` 参数：来源最显式，但扩大底层执行 API，并让 Agent 层理解 Objective 专用调用方式。
- 把系统 Assignment Input 当普通用户消息：无需改模型，但前端和后续逻辑会误认为模板内容是用户亲自提供的原始输入。
- 在 UserMessage 上保存来源布尔值：保持执行 API 不变，也让存储、上下文和前端可以使用同一份来源事实。

## Consequences

- `UserMessage.is_user_provided` 默认 `true`，现有 string、ContentPart list、Console 和渠道输入的规范化行为保持兼容；外部用户入口不能自行伪造为 `false`。
- ObjectiveService 将用户原始输入先原样保存为 Objective fact。直接使用用户原文作为 Assignment Input 时保持 `true`；只要内容经过系统模板生成或包裹，传给 Agent 的 UserMessage 就标记为 `false`，原始用户事实仍可独立展示和追溯。
- work、verification 及其他由 Objective/Outcome 渲染的 Assignment Input 使用 `is_user_provided=false`。计划门禁（plan guard）生成的固定提醒同样使用 `false`，但模型协议中的 message role 仍可为 `user`。
- `Agent.start/run/run_stream` 的公开参数保持不变；Agent runtime、RunLog serialization 和 StepView 投影必须完整保留该布尔值。
- `is_user_provided=false` 的输入不能创建 ObjectiveUserInput、改变用户身份或在普通聊天界面冒充用户气泡。它可以在 Objective 时间线或开发 Run Trace 中按 Assignment/系统提醒样式展示。运行中用户输入的注入消息属于此类；权威用户事实仍是 ObjectiveUserInput。
- 该布尔值只表达消息来源，不决定内容相关性。后续 Assignment 是否使用这条历史，仍由 Objective 当前目标与触发信息的相关性判断。
