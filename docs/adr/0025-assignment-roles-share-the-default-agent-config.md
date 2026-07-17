# Assignment 职责共享系统默认 AgentConfig

Objective 中的入口、工作与验收 Assignment 都由同一个系统默认 AgentConfig 构建的 Session persistent agent 执行。`worker` 与 `verifier` 不是两种 agent 类型或两条 registry config；它们只表示 Assignment 的职责、ObjectiveView 内容和系统控制的输入模板不同。Handoff target 决定下一 Assignment 的职责，不选择具名 agent、pattern 或 AgentConfig。运行中用户输入不创建新 Assignment（ADR 0023）。

## Status

accepted

## Considered Options

- 为 worker 与 verifier 分别注册 AgentConfig：可以独立调优模型和工具，但把领域职责固化为部署配置，并重新引入 handoff 到具名执行者的路由问题。
- 让前一个 agent 从 AgentRegistry 选择 config：灵活，但要求模型读取完整注册表，带来信息噪声，也让受旧轨迹影响的上下文承担重大执行选择。
- 让 Scheduler 根据语义选择 registry agent：集中可控，但把语义判断泄漏到机械调度层。

## Consequences

- AgentRegistry 继续管理系统已有 AgentConfig records；现有 runtime factory 负责从 record 构造 Agent 实例。Registry 不是 Objective 的语义路由表。
- Objective 系统配置只引用一个默认 AgentConfig。一个 Session 使用稳定 agent identity；每个 Assignment 创建新的 Run，并从同一 Session 的 RunLog 重建可见消息历史。
- `target=agent` 与 `target=verifier` 映射为不同的 Assignment kind 和输入，而不是不同的 config id。
- Assignment kind 使用封闭集合 `intake`、`work`、`verification`；它只选择职责输入模板，不改变默认 AgentConfig。
- Assignment 的最终模型上下文必须显式包含 kind、按出现顺序且不重复的用户输入表示、适用的 ObjectiveContributions、ObjectiveBudget、相关 Outcomes 与本次职责。用户输入表示来自已提交 Session 历史（按 `input_id` 去重复用），未外置沿用 canonical UserMessage，已授权外置使用 Artifact 的 path 与 summary；系统生成的 Assignment Input 不复制用户原文，默认 AgentConfig 本身不因职责而修改。
- `max_steps_per_run`、模型、工具和 skill 权限来自同一个默认 AgentConfig，并继续受系统控制；模型不能借 Assignment kind 覆盖这些配置。
- 默认 AgentConfig 的共享 system prompt 提供 planning policy；具体采用直接执行、Agent、Parallel、Pipeline 或 pattern，由新 Assignment Run 依据 ObjectiveView、最近 Outcome 与 Session 可见历史规划，不由前一个 Run 或 Scheduler 预先选择。
