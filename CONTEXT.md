# Agiwo Domain Language

Agiwo 是以 **Session** 为用户交互边界、以 **MainAgent** 为会话级执行者、以 **Run** 为单轮主执行、以 **RunLog** 为唯一执行账本、以瘦 **Scheduler**（waitset / 取消子树）承载 **Worker** 委派的 agent 运行时。本文件只定义**现行核心术语**；目标分层见 ADR 0049，Objective 等作废词见 ADR 0048，不得再指导实现。

## Language

**会话（Session）**：
用户与系统之间一段可持续、多轮且可分叉的对话容器（类似产品中的会话 tab），也是唯一用户交互主路径。Session 拥有稳定的 MainAgent、SessionIntent，以及由 RunLog 投影出的用户可见消息历史（无独立消息表）。
_Avoid_：Objective、Turn 聚合、为每个 Run 更换主执行者身份、把 SessionStore 元数据当成消息账本

**代理规格（AgentSpec）**：
可复用的主 agent 配置模板（模型、工具、skills、选项等），不绑定某一 Session，不含活队列或当前 Run。
_Avoid_：把活着的 MainAgent 叫作 spec、在 spec 上保存 Session 状态

**主执行者（MainAgent）**：
由 AgentSpec 绑定到某一 Session 后的长期活体：Run 结束只进入空闲，不销毁。外部用户输入经 `accept`；用户插话与 Worker report 共用 MainAgent 的 Session 级 staging 队列与同一 enqueue/drain 原语（活 Run 则转投 Loop 消息队列，空闲则开新 Run 或续 Run）；结束门禁反馈只发生在 Run 存活期间，由 Loop 直接写入活 Run 的消息队列，不经 Session 级队列。不再另设 steer/inject。
_Avoid_：ExecutionHandle、Run 结束即丢弃句柄、steer/inject 双队列、SessionRuntime（作领域对外名）、与 MainAgent 并行的第二套 Agent.run 入口

**运行（Run）**：
Session 内主执行者的一轮实际执行，拥有本轮 RunLog 与可选 RunPlan。同一 Session 上用户再提交时，若 Main Loop 仍在跑则注入同一 Run；否则 MainAgent 开启新 Run。
_Avoid_：Turn 聚合、Assignment、并行多个无关的活动主 Run

**运行日志（RunLog）**：
按 Session 追加的唯一执行事实账本（行上含 `session_id` / `run_id` / `agent_id`）。用户可见历史、Trace/stream、以及传给模型的 messages 均由其投影。主上下文按 **`session_id + 主 agent_id`** 加载以保留多轮主历史；Worker 用独立 `agent_id` 隔离。不得用「只按 run_id」作为主多轮上下文边界。用户原文以 `is_user_provided=true` 写入。不得另建第二份「Session 消息表」替代本账本。
_Avoid_：ObjectiveLog、TurnLog、覆盖式快照当唯一真相、与 RunLog 平行的第二套用户消息库、按 run_id 单独作为主多轮上下文键

**会话意图（SessionIntent）**：
Session 级对齐视图（指南针，不是发动机）：全量保留用户输入原文，Run 结束时追加 report 摘要；可在 Run 结束时附带上一轮 RunPlan 快照。**独立持久化**（自有表/文档，按 `session_id`），与 RunLog 并存；`accept` 时对用户全文做 D1 双写。用于跨 Run 对齐与（在启用时）语义结束门禁；第一版不做 LLM 提炼的 core_goal / 路线 / 决策库。不是跨 Run 任务状态机。
_Avoid_：Objective、ObjectiveStatus、把 Intent 当派发账本、对用户输入做 summary、默认用模型改写「真实目标」、把 Intent 仅做成每次扫描 RunLog 的临时投影、塞进 Session 元数据行当大 blob

**运行计划（RunPlan）**：
当前 Run 的结构化工作计划 / milestone，与 RunLog 并列管理，不靠解析工具输出文本还原。新 Run 从空计划开始；Run 成功结束后冻结。Scheduler 不解释 RunPlan。
_Avoid_：挂在 Session 上的活计划表、Objective 层第二份清单、仅从 tool 日志反序列化当权威

**循环（Loop）**：
单次 Run 内模型调用与工具执行的循环引擎；自带消息队列，在轮次间隙取出 `accept`/门禁反馈/Worker 完成等注入。
_Avoid_：把 Loop 与 Session 级 MainAgent 混名为同一个对象

**工人（Worker）**：
仅由 MainAgent 派生的一次性委派执行：隔离上下文，只产出最终 report，不可复用、不可再启动、不可再派生 Worker。同步则主 Loop 经 Scheduler 等待；异步则主继续，完成后将 report 写入主上下文并 **enqueue** 进与用户输入相同的主队列；若主 Loop 已停则在同一 Run 内 resume。取消 Main 时取消其 Worker。
_Avoid_：把 Agent-as-Tool 叫做 Worker、可复用的 child agent、子 Worker 再派生、把 Worker 全程轨迹写入主上下文、跨 Run 自动接力账本

**代理工具（Agent-as-Tool）**：
把一个 AgentSpec（或等价配置）暴露为普通功能工具；在一次 tool call 内嵌套跑模型与工具。它不是 Worker，不进入 Scheduler 委派树 / waitset，不走 Worker report 协议；与 Worker **并存**，各管各的语义。
_Avoid_：子 agent、Worker、第二套委派协议、用 waitset 描述 Agent-as-Tool

**委派（Delegation）**：
MainAgent 将局部工作交给 Worker，仍保留对结果的责任，并通过 Scheduler waitset 等待或异步收回 report。不含 Agent-as-Tool。
_Avoid_：接力、移交、把工具嵌套当成委派、深度大于一的 Worker 树、跨 root 的自动下一跳账本

**机械调度者（Scheduler）**：
不可替代的核是 waitset 与取消子树（登记等待、到点或完成时唤醒、主取消则取消 Worker）。不拥有用户历史语义，不解释 RunPlan，不作「跑 Session」的全能门面。与 Agent-as-Tool 无关。
_Avoid_：中心编排器、ObjectiveService、把 steer/开 Run/写历史收口成 Scheduler 唯一 API、跨 Run 任务账本、用 Scheduler 描述 Agent-as-Tool

**结束门禁（Completion Gates）**：
Main 宣称本轮结束前的检验：机械门禁始终生效（未完成 Worker、仍 open 的 milestone 等）；语义门禁默认关闭，按复杂度指标开启，开启时以 SessionIntent 为重要检验标准。输出仅为继续（带反馈入队）或允许结束，不维护平行进度状态机。**第一期只交付机械门禁**（G-v1a）；语义门禁留缝默认不开。
_Avoid_：Objective 验收闩锁、把语义门禁做成跨 Run 任务大脑、解析错误文本当唯一依据、第一期强行上语义门禁阻塞 M3 主路径

**会话入口（Session Gateway）**：
Console Session 与渠道消息的统一用户入口。将用户消息写入 RunLog（Session 历史投影）并交给 MainAgent.accept（或现行等价路径）；不创建或升级任何跨 Run 任务账本。
_Avoid_：每条消息创建 Objective、升级为 Objective、普通用户入口绕过 Session 直接编排任务状态机

**运行中用户输入（Running User Input）**：
Main Loop 仍在运行时收到的用户自然语言输入。全文写入 RunLog（及 SessionIntent），并经 MainAgent 注入同一 Run 的消息队列；不结束当前 Run，不另建任务账本。
_Avoid_：用 system-notice 再包装一份用户原文、为运行中输入更换 MainAgent 身份、DRAINING 后为插话新建第二 Run

**会话分叉（Session Fork）**：
依据用户提供的 `fork_context_summary` 创建全新 Session。不复制源 Session 的活动 Run 或 checkpoint；summary 在新 Session 的首个 Run 中作为临时 `<system-notice>` 使用。
_Avoid_：复制活动 Objective、两个 Session 恢复同一 checkpoint、只保存但不消费 fork summary

**会话归档（Session Archive）**：
把 Session 从普通用户的默认列表隐藏，但保留 RunLog、`sessions/<session_id>/artifacts/` 与恢复可见性的能力。恢复只恢复可见性，不自动调用模型。
_Avoid_：普通用户物理删除、留下活动调度状态的删行操作、恢复即自动调用模型

**轨迹复盘（Trajectory Review）**：
Agent 对最近一段工具调用是否推进当前 active milestone 的可选自我检查。复盘只追加纠偏经验与低置信度有用性评分，不删除或改写历史消息；基于 RunLog 投影，不解析 RunLog sequence 进模型协议。
_Avoid_：在模型协议中暴露 RunLog sequence、根据评分确定性删除内容

**产出（Artifact）**：
对 agent 工作目录中独立文件型产出的索引与按需读取句柄。Artifact 不驱动执行控制；普通文本报告与聊天回复不是 Artifact。
_Avoid_：把 report 建模成 Artifact、把任意字符串当文件

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

SessionIntent 不是 Objective 的换皮；禁止用 Intent 复活跨 Run 任务状态机或派发账本（ADR 0049）。
