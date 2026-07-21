# Agiwo Domain Language

Agiwo 是一个以 Session 为唯一用户交互边界、以 Run 为执行单位、以 Scheduler waitset 承载 child 委派的 agent 运行时。本文件只定义**现行核心术语**；旧 Objective / Assignment / Turn 聚合等作废词见 ADR 0048，不得再指导实现。

## Language

**会话（Session）**：
用户与 agent 系统之间一段可持续、多轮且可分叉的对话容器，也是唯一用户交互主路径。Session 拥有稳定的 persistent agent identity 与唯一的用户可见消息历史；用户消息先进入该历史，再启动 root Run。
_Avoid_：Objective、Turn 聚合、为每个 root Run 更换 agent identity、Assignment

**运行（Run）**：
一个 agent 的一次实际执行。Session 启动的是 **root Run**；Parallel、Pipeline 或工具委派产生的是 **child Run**，各自有独立 identity，但仍属于该 root 的执行树。同一 Session 上用户再提交一条消息时，若当前 root 仍在 `RUNNING`，则注入/steer 同一 root；否则启动新的 root Run。
_Avoid_：Turn、Assignment、把每个 child Run 建模成 handoff、并行多个无关的活动 root Run（一棵树内 child 并行除外）

**运行日志（RunLog）**：
Run 内 append-only 真相源。Session 用户可见历史与 Trace/stream 视图均由其投影；用户原文以 `is_user_provided=true` 写入，不得另建第二份用户账本。
_Avoid_：ObjectiveLog、TurnLog、覆盖式快照当唯一真相

**运行状态（RunStatus）**：
Run 的执行生命周期投影，封闭为 `RUNNING / PAUSED / COMPLETED / INTERRUPTED / FAILED`。`PAUSED` 表示可恢复中断；`FAILED` 仅表示不可恢复故障。
_Avoid_：ObjectiveStatus、WAITING_USER、AssignmentStatus、用 CANCELLED 表示普通暂停

**机械调度者（Scheduler）**：
Agent 之上的机械编排层：维护 AgentState、启动和恢复 Run、管理 child 委派树与 waitset，不拥有跨 Run 任务账本。对 Session 暴露启动 root、steer/注入运行中输入、取消、以及执行树查询；内部可保有 WAITING / IDLE / QUEUED 与 mailbox。
_Avoid_：中心编排器、主控 agent、ObjectiveService、Objective 专用派发账本

**委派（Delegation）**：
一个执行者把局部工作交给另一个执行者（child），仍保留对该工作结果的责任，并通过 waitset 等待结果返回。
_Avoid_：接力、移交、跨 root 的自动下一跳账本

**会话入口（Session Gateway）**：
Console Session 与渠道消息的统一用户入口。把用户消息写入 Session 历史并经 Scheduler 启动或注入 root Run；不创建或升级任何跨 Run 任务账本。
_Avoid_：每条消息创建 Objective、升级为 Objective、普通用户入口绕过 Session 直接编排任务状态机

**运行中用户输入（Running User Input）**：
活动 root Run 仍在 `RUNNING` 时收到的自然语言用户输入。完整原文以 `is_user_provided=true` 写入 Session 历史，并注入同一 root Run；不结束当前 Run，不另建任务账本。
_Avoid_：用 system-notice 再包装一份用户原文、为运行中输入更换 agent identity、DRAINING 后新建 steering Run

**会话分叉（Session Fork）**：
依据用户提供的 `fork_context_summary` 创建全新 Session。不复制源 Session 的活动 root Run 或 checkpoint；summary 在新 Session 的首个 root Run 中作为临时 `<system-notice>` 使用。
_Avoid_：复制活动 Objective、两个 Session 恢复同一 checkpoint、只保存但不消费 fork summary

**会话归档（Session Archive）**：
把 Session 从普通用户的默认列表隐藏，但保留 RunLog、`sessions/<session_id>/artifacts/` 与恢复可见性的能力。恢复只恢复可见性，不自动调用模型。
_Avoid_：普通用户物理删除、留下活动 Scheduler 的删行操作、恢复即自动调用模型

**运行计划（RunPlan）**：
root Run 通过 `update_plan` 按需声明的工作计划。计划项状态与规范化规则由 Agent 拥有；新的 root Run 从空计划开始。Scheduler 不拥有或解释 RunPlan。
_Avoid_：Objective 层第二份清单、把计划当作跨 Run 任务账本、物理删除计划项

**轨迹复盘（Trajectory Review）**：
Agent 对最近一段工具调用是否推进当前 active milestone 的可选自我检查，默认启用。复盘只追加纠偏经验与低置信度有用性评分，不删除或改写历史消息。
_Avoid_：在模型协议中暴露 RunLog sequence、根据评分确定性删除内容

**产出（Artifact）**：
对 agent 工作目录中独立文件型产出的索引与按需读取句柄。Artifact 不驱动执行控制；普通文本报告与聊天回复不是 Artifact。
_Avoid_：把 report 建模成 Artifact、把任意字符串当文件

**默认 Session Agent**：
系统从默认 AgentConfig 构建 Session 的 persistent agent；同一 Session 内相继 root Run 复用稳定 agent identity，以新的 `run_id` 区分执行。
_Avoid_：每个 Run 一个新 agent_id、WorkerAgentConfig / VerifierAgentConfig

**单次运行循环上限（max_steps_per_run）**：
`AgentOptions` 中的单 Run 正常模型调用阈值。达到阈值后只允许有限的总结与收口 attempt。
_Avoid_：跨 Run 的 step 总额度、ObjectiveBudget.max_steps

**模型逻辑调用（Logical Model Call）**：
一次具有稳定业务目的的模型调用，拥有 logical_call_id 与不变 phase。Provider retry 不创建新 phase，而是增加 attempt。
_Avoid_：用 logical call 掩盖真实请求次数

**执行故障（Execution Fault）**：
LLM、tool 或执行基础设施未能按契约完成一次操作的事实。
_Avoid_：把一切失败写成跨 Run 任务终态

**重试判定（Retry Disposition）**：
执行系统对故障作出的 `retryable`、`non_retryable` 或 `outcome_unknown` 分类。只有操作可安全重复且故障为 retryable 时才能自动重试。
_Avoid_：根据错误文本猜测、所有失败一律重试

**用户输入来源（is_user_provided）**：
标记一条 user 侧消息是否为真实用户原文。用户可见历史与模型上下文都必须区分用户原文与系统注入的 notice。
_Avoid_：把控制命令伪装成用户原文

## Retired language (do not use)

下列概念已由 ADR 0048 作废，不得出现在核心公开 API、现行定义或新测试的领域断言中：Objective、ObjectiveService、ObjectiveLog、ObjectiveStatus、RootRunRequested、Outbox、Run Role、verification latch、HandoffDecision（作跨 Run 控制面）、ObjectiveBudget、DRAINING、Assignment、Turn（作聚合/API）、Session Gateway「升级为 Objective」。已被取代的 ADR 正文在 `trash/adr-superseded-by-0048-*`（及 git 历史），不是规格。
