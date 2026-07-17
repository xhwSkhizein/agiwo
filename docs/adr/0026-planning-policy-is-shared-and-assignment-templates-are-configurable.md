# Planning Policy 共享，Assignment 模板可配置

系统默认 AgentConfig 的 system prompt 包含共享 planning policy，说明 agent 在收到 Assignment Input 后如何设计行动，并在直接执行、Agent、Parallel、Pipeline 与已有 pattern 之间动态选择。Assignment kind 不使用不同 AgentConfig，而是选择对应的固定输入模板；`intake`、`work`、`verification` 模板都可以在 Console 后台配置。模板把首次用户输入，或当前 Objective 与上一 AssignmentOutcome 等触发事实，渲染成驱动本次 root `agent.run` 的 Assignment Input；模型侧以当前 Run 临时使用的 `<system-notice>` 看到其中的系统职责，不复制 planning policy。运行中用户输入走注入路径，不占用独立模板 kind。

## Status

accepted

## Considered Options

- 把完整 planning 指令复制到每个 kind 模板：单个模板自包含，但四份规则会独立漂移，修改 planning 时难以保持一致。
- 让 Scheduler 根据 Objective 选择 pattern：执行路径直观，但 Scheduler 会重新承担语义规划，违背机械调度边界。
- 为每种执行计划注册专用 agent config：便于人工预设，但会重新引入具名 agent 选择和有限能力列表。

## Consequences

- 共享 planning policy 属于系统默认 AgentConfig.system_prompt；它描述规划方法、可用组合原语和权限边界，但不为具体 Objective 预先生成 workflow。
- 共享 planning policy 必须把 `RunPlan` 限定为当前 Assignment root 在下一次 Decision 前承担的责任。计划项可以由 child 执行，但必须由 root 接收并负责；后继 Agent、Verifier 或 User 才能完成的未来工作不能进入当前 `RunPlan`，否则完成门禁会在 handoff 前形成循环依赖。
- Agent system prompt 按实际内建工具装配：`update_plan` 可用时渲染 RunPlan 规则，`enable_trajectory_review=true` 时渲染轨迹复盘规则；不得展示当前 Run 不具备的工具指令。`enable_trajectory_review` 默认保持 `true`，显式关闭后不装配 `review_trajectory`，但不影响 RunPlan、完成门禁或 carry_forward。相同 AgentConfig 的内建工具 schema 保持稳定，以维护 LLM 前缀缓存。
- Agent、Parallel、Pipeline 与已有 pattern 是 agent 形成执行计划时可以采用的组合原语，不是 HandoffDecision target，也不是 Scheduler 的语义路由项。
- Assignment kind 固定为 `intake`、`work`、`verification`，分别对应 Console 可编辑的输入模板。
- Assignment 是一张持久化工作单；Assignment Input 是工作单中保存并通过现有 `user_input` 实际交给 root `agent.run` 的 UserMessage 快照。Assignment 本身不“接收上下文”或执行工作，输入快照才是模型本次执行的直接输入。
- 模板只能从类型化 Assignment 与 ObjectiveView 字段装配系统生成的内容，至少包括 current_goal、适用 Contributions、ObjectiveBudget、相关 Outcomes 和本次职责。ObjectiveUserInput 不属于模板字段；上下文装配器按 `input_id` 去重复用 Session 历史中已有的 canonical user message，新 Assignment 内容只追加尾部，不因「缺失」向中段补回用户原文。
- 每次 Assignment 创建时记录 template id、version/hash 和最终渲染输入；RunLog 保存实际模型输入，使模板修改后仍能重放和解释历史执行。
- `<system-notice>` 只是 Assignment 对模型的提示载体，不是领域真相源；Objective 状态只能从结构化 Assignment facts 重建，不能解析 notice 文本得到。
- Assignment Input 若作为 `user_input` 进入 Run，则可以持久化为带 `is_user_provided=false` 的内部 User Step；它不属于普通用户对话、不产生 ObjectiveUserInput，也不会自动进入后续 Session 的用户历史。开发模式仍可完整查看它。
- 若模板渲染结果直接作为 `user_input` 提交，则对应 UserMessage 必须使用 `is_user_provided=false`；前端与 Objective 规则据此区分系统工作单和真实用户输入，Agent API 不增加新参数。
- `ContextAssembled` 与 `LLMCallStarted` 保存包含 notice 的完整实际输入，Console 开发模式据此展示和调试；普通聊天界面不展示该 notice。
- planning policy 与 kind 模板的修改只影响修改后创建的 Agent/Assignment；已经运行或正在收口的 Run 使用创建时的快照。
- 模板和 planning policy 的配置、校验与预览由 Console 控制面提供；模型无权在运行中修改这些系统配置。
- 四种 kind 模板作为系统默认 AgentConfigRecord 的配置字段，随对应 AgentRegistryStore 持久化；不建立独立模板表或模板版本仓库。
- Console 保存时先通过 AgentRegistry 持久化新的 config record，成功后再更新内存中的有效模板配置。进程重启时优先加载 Registry 中的默认 agent record，只有未保存记录时才回退到环境或代码默认值。
- 模板只支持固定占位符 `current_goal`、`objective_contributions`、`objective_budget`、`assignment_outcomes`、`assignment_kind`，不提供 Jinja 循环、条件或代码执行。模板不再渲染“全部用户输入”或“最新用户输入”；结构化 input id、去重集合与关联只服务上下文装配，不暴露为模型协议。
- Console 在保存前拒绝未知或缺失的必需占位符，并提供基于示例 ObjectiveView 的渲染预览。
- template revision 使用 AgentConfigRecord.updated_at 与内容 hash 表达；Assignment 继续保存模板 hash、最终渲染输入和 RunLog 模型输入，以解释历史执行。
