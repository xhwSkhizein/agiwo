# Agiwo Domain Language

Agiwo 用 agent 的语义判断驱动任务推进，并用确定性的执行控制保证任务可追踪、可约束地运行。本文件统一描述这套协作模型中的核心术语。

## Language

**语义决策（Semantic Decision）**：
依据目标、上下文和产出含义，判断任务下一步应当执行什么、是否需要协作，以及是否可以提议结束。
_Avoid_：智能调度、中心编排

**执行控制（Execution Control）**：
对已经形成的结构化决定进行校验、记录和执行，不判断决定在语义上是否正确。
_Avoid_：语义路由、智能匹配

**机械调度者（Mechanical Dispatcher）**：
Scheduler 所承担的 Run/agent 级机械执行职责：维护 AgentState、启动和恢复 Run、管理委派树与等待，但不读取 ObjectiveLog、不解释 Decision，也不拥有 ObjectiveBudget。Objective 级控制的唯一 owner 是 ObjectiveService。Objective 对 Scheduler 的依赖面收窄为稳定 facade：派发预分配 root Run、wait、运行中注入、可恢复暂停/恢复、执行树查询；Objective 路径不使用 `route_root_input` / mailbox / WAITING / QUEUED 作为工作推进入口。Scheduler 内部仍可保有 nested agent 的等待与队列实现。
_Avoid_：中心编排器、主控 agent、Objective 直接依赖 Scheduler mailbox 状态机

**委派（Delegation）**：
一个执行者把局部工作交给另一个执行者，但仍保留对该工作结果的责任，并等待结果返回。
_Avoid_：接力、移交

**接力（Handoff）**：
一个执行者结束自己的责任并把任务控制权交给下一位执行者；前后执行者是相继关系，不是父子关系。
_Avoid_：委派、派生 child

**自动接力额度（max_handoffs）**：
Objective 中自动执行链可以提交的 `target=agent` 与 `target=verifier` 接力次数。`target=user` 是始终可达的用户边界，不消耗该额度；用户回复后创建新的 peer root Run 也不计数。
_Avoid_：限制用户安全出口、把用户回复算作自动接力

**目标（Objective）**：
一个 Session 中从某个用户意图开始，到结果被接受、目标失败或交还用户决定为止的一次完整问题解决过程。它既是全局目标、预算、协作关系和最终结果的共同边界，也是非终态期间随用户输入持续演进的语义模型；终态 Objective 不重新打开，后续输入在同一 Session 中创建新 Objective。
_Avoid_：ObjectiveRun、WorkflowRun、单个 agent 的 task、重新打开终态 Objective

**当前目标（Current Goal）**：
Objective 对“现在需要完成什么”的简洁、带版本投影。程序在每次用户输入时原样追加目标事实，并在 root Run 收尾时根据结构化 Objective 修订更新非权威的意图、范围、成功标准理解与假设；这些分析不能覆盖、删除或取代用户事实。
_Avoid_：作为真相源的可变 summary、让 agent 改写用户原始输入

**目标修订（Objective Revision）**：
Objective 全局语义模型的一次 append-only 版本更新。用户输入立即追加权威目标事实；运行收尾调用可以提出精简、结构化且带来源的非权威分析修订，由 ObjectiveService 校验后提交。Run 内不存在修改 Objective 的 runtime tool，RunPlan 由独立的 `update_plan` 管理。
_Avoid_：运行中任意改写 Objective、伪造用户输入、把 RunPlan 复制进 Objective、无来源的覆盖式 summary

**会话（Session）**：
用户与 agent 系统之间一段可持续、多轮且可分叉的对话容器。Session 保存多个 Objective 的交互归属和历史连续性，并拥有一个稳定的 persistent agent identity；同一 Session 的相继 root Run 继续使用该 identity 和可见消息历史。
_Avoid_：Objective、为每个 root Run 更换 agent identity、重新引入 Assignment

**会话分叉（Session Fork）**：
依据用户提供的 `fork_context_summary` 创建全新 Session 的操作。它不复制源 Session 的活动 Objective、预算、活动 root Run、checkpoint 或 Outcome，也不影响源 Objective；summary 在新 Session 第一个 Objective 的 root Run 中作为临时 `<system-notice>` 使用。
_Avoid_：复制活动 Objective、两个 Session 恢复同一 checkpoint、只保存但不消费 fork summary

**会话归档（Session Archive）**：
把 Session 从普通用户的默认列表隐藏、但保留其 ObjectiveLog、RunLog、`sessions/<session_id>/artifacts/` 文件与恢复能力的操作。存在活动 Objective 时必须先完成用户暂停并保存 checkpoint；恢复 Session 只恢复可见性，不自动恢复 Objective 执行。
_Avoid_：普通用户物理删除、留下活动 Scheduler 的删行操作、恢复即自动调用模型

**目标服务（ObjectiveService）**：
Scheduler 之上的 Objective 深模块唯一公开门面，也是 Objective 级执行控制的唯一 owner。它消费 Run Outcome/Decision，写 ObjectiveLog 与 ObjectiveStore，检查并结算 ObjectiveBudget，维护 Session 的活动 Objective 占用，并创建待派发 root Run / outbox；Scheduler 只执行已经确定的 Run 调度。ObjectiveLog 重放、outbox、预算 hook 和输入装配都是包内实现细节；稳定执行关系只有 `ObjectiveService -> Scheduler -> Agent.run`。公开语言是 `Session → Objective → Run`，不再暴露 Assignment 聚合。
_Avoid_：ObjectiveEngine 语义编排器、直接操作 AgentState、为每项内部职责建立公开 service/port、公开 Assignment 生命周期 API

**目标状态（ObjectiveStatus）**：
Objective 当前生命周期投影，封闭为 `CREATED / RUNNING / DRAINING / WAITING_USER / BUDGET_PAUSED / USER_PAUSED / COMPLETED / FAILED`。`COMPLETED / FAILED` 是不可重新打开的终态；FAILED 只表示不可恢复的存储、ObjectiveLog 或领域不变量故障，不承载普通 agent、模型或工具失败。
_Avoid_：CANCELLED 用户暂停、用 FAILED 表达执行返工、从终态恢复

**目标日志（ObjectiveLog）**：
Objective 跨多次 root Run 的 append-only 真相源，记录用户输入、贡献、预算、状态、Decision、handoff、Run Outcome 与 checkpoint 等事实。`Objective` 是从 ObjectiveLog 重建的投影视图；ObjectiveLog 跟随 RunLog 的存储配置，MVP 在相同物理数据库中使用独立表（memory / SQLite），以 `run_id`（及 objective_id）关联，不复制或伪装成 RunLog/StepView。其他 backend fail-closed。
_Avoid_：覆盖式 Objective 快照作为唯一真相、特殊 RunLog kind、特殊 StepView、半成品 Mongo ObjectiveStore、配置漂移时 silent 降级为 memory、以 assignment_id 作为领域主键、为旧 Assignment fact 写 migration 或双读兼容层（切换前清理开发库）

**目标时间线（Objective Timeline）**：
Console 从 ObjectiveLog 投影出的完整任务推进视图，按 sequence 展示用户输入、贡献、root Run、Decision、预算、故障、用户等待与终态。Run 节点通过稳定 `run_id` 下钻到 RunLog；开发模式展示收尾调用的完整输入输出、reasoning、解析和用量。
_Avoid_：从 AgentState 快照猜历史、把 RunLog 全量复制到时间线、时间线一等展示 Assignment

**最终交付视图（Final Delivery View）**：
Objective 完成后以正式交付的普通文本 report 为主体、附带文件型 Artifact 列表的用户视图。Objective 时间线默认折叠但可展开，Run Trace 继续作为时间线节点的开发下钻；前端依据 ObjectiveDelivered fact 中的 report 文本与明确 Artifact 引用渲染，不从最后一条 assistant message 猜测结果。
_Avoid_：把未验收候选结果当交付、把普通文本 report 伪装成 Artifact、从聊天尾部推断最终内容

**目标事件流（Objective Event Stream）**：
把已提交 ObjectiveLog facts 通过 SSE 按 sequence 重放并实时推送的 API 视图。SSE id 使用 ObjectiveLog sequence，客户端以 `Last-Event-ID` 或 `after_sequence` 续传；普通用户只看到已提交进度和最终交付，不接收未验收 Run 的 token delta。
_Avoid_：WebSocket 双向控制、仅内存事件、第二套 Run stream 协议

**目标派发 Outbox（Objective Dispatch Outbox）**：
ObjectiveStore 中持久化的“待派发箱”，与 ObjectiveLog facts 在同一事务写入。ObjectiveService 提交 Objective 变化时，同时留下 `DispatchRequested` 待办；Objective 包内的机械派发逻辑随后使用稳定 `dispatch_id / run_id`（及 `objective_id`）通过 Scheduler 公开接口启动 root Run。进程崩溃不会丢失待办，outbox 也不承载语义决策。
_Avoid_：ObjectiveLog 提交后直接依赖内存调用、复用可删除 PendingEvent、outbox 以 assignment_id 为领域键

**运行职责（Run Role）**：
挂在 Objective 管理的 root Run 上的封闭职责：`work / verification`。Objective 的第一个 root Run 为 `work`。当 `verification_required` 为 false 时，`work` 可经收尾直接交付；为 true 时，`work` 只能提交完成提议并接力新的 `verification` root Run。系统按 role 选择输入模板；role 不改变 AgentConfig。
_Avoid_：Assignment Kind、intake、按 role 切换 agent config

**运行输入（Run Input）**：
系统根据 root Run 的 `Run Role` 与当前 Objective 复杂度档位，把首次用户输入或当前 Objective 与上一 Run Outcome 等触发事实形成的 `UserMessage` 输入快照。它作为现有 `user_input` 驱动 root `agent.run`；真实用户原文使用 `is_user_provided=true`，系统模板或计划提醒使用 `false`。输入分档：当 `verification_required` 仍为 false 且尚无需要承接的贡献/Outcome 历史时，采用薄输入；当 `verification_required` 为 true，或 role=`verification`，或已有需承接历史时，使用完整 role 模板渲染 ObjectiveView。
_Avoid_：Assignment Input、简单路径仍注入完整史诗模板

**运行计划（RunPlan）**：
当前 Objective 管理的 root Run 通过 `agiwo.agent.plan` 内建系统工具 `update_plan(changes=...)` 按需声明和直接管理的工作计划，不是 Objective 层的独立 todo 模型。Agent 只提交本次增量变化，程序原子应用、规范化 active 状态后，以完整 `RunPlanUpdated` RunLog fact 保存当前快照。计划项沿用 `pending / active / completed / abandoned` 状态，不提供物理删除；不再需要的项目进入 `abandoned`，同一 Run 内可以重新打开。只要存在未解决项目，规范化计划必须恰有一个 active；`active_milestone_id` 由 milestone status 派生，不单独持久化。同一份计划同时驱动轨迹自省、完成门禁、`verification_required` Latch 与中断时的 `carry_forward` 快照。计划严格属于当前 root Run，只覆盖当前 root 及其 child 执行树在下一次 Decision 前能够完成和验证的责任；新的 peer root Run 从空计划开始。门禁只对 root Run 生效，child Run 的计划随委派结果消亡。Scheduler 不拥有或解释 RunPlan。
_Avoid_：Assignment Plan、Objective 层第二份清单、第二个计划管理入口、物理删除计划项

**轨迹复盘（Trajectory Review）**：
Agent 对最近一段工具调用是否推进当前 active milestone 的可选自我检查，默认启用。复盘只在消息尾部追加纠偏经验和以 `tool_call_id` 标识的 0–3 有用性评分，不删除或改写历史消息；评分是实验性、低置信度信息，只能作为后续 compaction 的可选参考。
_Avoid_：在模型协议中暴露 RunLog sequence、根据评分确定性删除内容、让复盘成为 RunPlan 或 compaction 的必需依赖

**运行结果（Run Outcome）**：
每个结束的、由 Objective 管理的 root Run 都必须产生的唯一交接封装，无论正常完成、用户中断还是执行故障。Outcome 挂 `run_id`（及 `objective_id`），用普通文本 report、可选的文件 Artifact 引用和中断时的 `carry_forward` 计划项带出已获得的信息，并保留生成来源；原始调用细节继续由 RunLog 保存。`PAUSED` 的 Run 尚未结束，因此暂不产生 Outcome。child Run 不单独写 Objective 级 Outcome。
_Avoid_：AssignmentOutcome、无 Outcome 结束 root Run、把完整 RunLog 复制进 Outcome、把普通文本 report 建模成 Artifact

**运行（Run）**：
一个 agent 的一次实际执行。Objective 管理的工作以 **root Run** 为责任与 Outcome 边界：root Run 在派发前预分配稳定 `run_id`，携带 `Run Role`，并承担最终责任；Parallel、Pipeline 或工具委派产生的 child Run 保留各自独立 identity，但仍属于该 root Run 的执行树。一个 Objective 同时最多一个非终态 root Run；并行只存在于该 root 下的 child 树。公开领域模型不再包含 Assignment。
_Avoid_：Assignment、把每个 child Run 都建模成 handoff、并行多个活动 root Run

**运行状态（RunStatus）**：
Run 的执行生命周期投影，封闭为 `RUNNING / PAUSED / COMPLETED / INTERRUPTED / FAILED`。派发前的待办只存在于 outbox，不单独发明与 RunStatus 平行的 AssignmentStatus。`PAUSED` 表示可恢复中断（用户停止、预算触顶等），保持同一 run_id，不写失败 termination；`INTERRUPTED` 用于单 Run 调用阈值收口后的交接结束；`FAILED` 仅表示不可恢复故障。root Run 进入 `COMPLETED` 前，其 `RunPlan` 中不得存在 `pending / active` 项，且必须已有 Run Outcome。
_Avoid_：AssignmentStatus、用 CANCELLED 表示可恢复暂停、把用户停止写成失败终态、恢复时创建新 run_id、把 outbox claim 当领域状态

**目标网关（Objective Gateway）**：
用户与 Objective 生命周期之间的异步服务边界；它接收目标后立即创建并返回 `objective_id`，提供 Objective 当前投影、按 sequence 的事件流和用户输入入口。Console Session input 与渠道消息都只是它的适配入口：无活动 Objective 时创建 Objective，有活动 Objective 时向该 Objective 提交输入。
_Avoid_：前端 agent、中心编排器、普通用户入口直接调用 Scheduler

**运行中用户输入（Running User Input）**：
Objective 的 root Run 仍在 RUNNING 时收到的自然语言用户输入。系统先把完整输入写入 Session 历史与 ObjectiveLog（ObjectiveUserInput），再向**同一**活动 root Run 注入一条 `is_user_provided=false` 的系统提示 user 消息（可含 `<system-notice>` 包装），并附带用户原文，要求模型检查与当前目标/计划的一致性后再继续。不结束当前 root Run、不写 Run Outcome、不创建新的 peer root Run。
_Avoid_：DRAINING 后新建 steering Run、Steering Outcome Synthesis、为运行中输入更换 agent identity、重新引入 Assignment

**输入复杂度评估（Input Complexity Assessment）**：
Objective 创建并写入权威 `ObjectiveUserInput` 之后、派发首个 root Run 之前，系统发起的一次轻量、非关键路径模型调用：只输出 0–10 的复杂度分数（越大越复杂）并写入 ObjectiveLog。每个 Objective 只评估一次（首次用户输入）。当分数严格大于固定阈值（默认 3，进程级可配置）时，仅在首个 root Run 的输入中附加「建议规划 / 使用 plan 工具」类 `<system_notice>`；是否规划仍由模型决定。评估失败或超时时记空分并按简单路径继续派发，不阻塞 Objective。该评估不设置 `verification_required`，也不在后续用户输入上重复打分或重复附加规划建议。
_Avoid_：创建 Objective 前的同步关键打分、用入口分数强制验收、打分失败则拒绝用户请求、每轮用户输入重复打分、把建议规划写成强制门禁

**验收必需（Verification Required）**：
Objective 上由运行时轨迹机械置位的布尔标志：当任一 root Run 的 `RunPlan` 中出现过至少 1 个 milestone（任意状态，含 `abandoned`）时置为 true，且只增不减（Latch）。入口复杂度预判不设置该标志。权威写入点在 root Run 收尾 → Decision 映射之前：Objective 读取该 Run 的 RunLog 计划事实并提交 `VerificationRequiredSet`（若尚未置位）。收尾时若为 true，执行者不得以 `target=user, expects_reply=false` 直接交付，必须接力独立 verifier；若为 false，允许在系统收尾规则下直接交付。
_Avoid_：入口分数单独强制验收、work kind 无条件强制验收、由模型口头声明是否需要验收、以收尾当下计划是否清空来解除验收、要求 milestone 非 abandoned 才置位、靠 Agent live hook 作为唯一 latch 写入点、在派发 verification 时重复写 Latch fact

**完成提议（Completion Proposal）**：
执行者认为当前产出可以满足 Objective 时提交的候选结果。仅当 Objective 的 `verification_required` 为 true 时，它只能触发验收、不能直接结束 Objective；否则可经系统收尾直接进入交付路径。
_Avoid_：把一切完成都当成必须验收、把未验收候选当最终交付

**验收运行（Verification Run）**：
针对完成提议创建的、`Run Role=verification` 的独立 peer root Run，依据 Objective 的原始目标和硬约束判断候选结果应当通过、返工还是交还用户决定。仅在 `verification_required` 为 true 时进入该路径。
_Avoid_：Completion Review Assignment、入口 agent 自审、全链路重放、对无计划简单问答强制验收

**默认 Session Agent**：
执行系统从同一个默认 AgentConfig 构建 Session 的 persistent agent。该 Session 内相继的 work / verification root Run 复用稳定 agent identity，并通过新的 `run_id`、`Run Role` 与输入表达职责差异。
_Avoid_：WorkerAgentConfig、VerifierAgentConfig、每个 Run 一个新 agent_id、为运行中用户输入新建 agent、Assignment Agent

**Planning Policy**：
系统默认 AgentConfig 的共享 system prompt 中关于行动设计的规则，说明 agent 如何依据当前 root Run 的职责选择直接执行、Agent、Parallel、Pipeline 或已有 pattern。它提供规划方法和边界，不预先生成固定 workflow。
_Avoid_：把 planning 复制进每个 role 模板、Scheduler 预先选择 pattern

**产出（Artifact）**：
对 agent 工作目录中独立文件型产出的索引与按需读取句柄，例如图片、PDF、日志、额外生成的大文本，以及用户授权外置的超长输入文件。Artifact 对象包含稳定 `artifact_id`、相对 path（落在 `{agent_workspace}/sessions/<session_id>/artifacts/`）、短 summary，以及可选的小体积 inline content（超过大小阈值只保留 path，需要时按 path 读取）。Artifact 本身不驱动执行控制；普通文本报告、故障说明和聊天回复都不是 Artifact。
_Avoid_：Decision、把 report/故障说明建模成 Artifact、把任意字符串当文件、路由指令

**决定（Decision）**：
Objective 管理的 root Run 成功结束时提交的唯一结构化终态指令，明确 Objective 接下来应当如何推进（交付、等用户、或派发下一个 peer root Run）。只有 Decision 可以驱动执行控制。
_Avoid_：Artifact、自由文本建议、Scheduler 推断

**运行收尾调用（Run Finalization Call）**：
Objective 管理的 root Run 已经可以停止时，执行系统在历史尾部追加一条 `is_user_provided=false` 的收尾 user 消息，并必然再发起一次模型调用，取得与 trajectory 解耦的结构化调度参考（至少含 `Decision`）。Outcome 的普通文本 `report` **不是**收尾 JSON 字段：系统直接取工作循环最后一条助手正文。收尾模型不得填写或复述 `report`。输出契约分两档：当 Objective 的 `verification_required` 为 false 时仅要求极简字段（`decision`）；当其为 true，或当前 root Run 的 role 为 `verification` 时，才要求完整字段（含 `new_contributions`、`contribution_annotations`、可选 `objective_update` 与文件 Artifact 引用等）。请求必须携带与当前 Run 常规 turn 相同的 tools 列表以保住前缀缓存；指令要求本次只输出结构化 JSON、不要发起 tool call。它不是模型自行选择的 tool call，也不另开第二个 Run；完整输入输出可调试，但不进入后续普通业务对话。在 `verification_required=false` 且 role=`work` 的简单路径上：若 `target=user` 但缺少或写错 `expects_reply`，缺省为 `false` 并交付；若 JSON 解析失败但已有普通文本 report，则以该 report 交付（Decision=`user` / `expects_reply=false`）。简单路径禁止因收尾失败而机械创建新的 work root Run。在 `verification_required=true` 且 role=`work` 时，收尾失败则机械 `target=verifier`，以已有 assistant report 作为完成提议。在 `verification` root Run 上收尾失败则机械 `target=user, expects_reply=true`，交还用户，不自动交付、不自动返工。`verification_required=false` 的简单路径：首次收尾解析失败后**不**再发起 correction LLM，直接机械 fallback。`verification_required=true` 或 role=`verification`：允许至多一次 correction；仍失败再机械 fallback。
_Avoid_：Assignment Finalization、closure、去掉 tools 的收尾请求、隐藏调用、简单路径仍强制完整收尾 schema、简单路径上的 finalization_correction、用收尾解析失败机械打开新的 work Run、验收路径因坏 JSON 而跳过 verifier 或假装交付、让收尾模型复述 report

**目标用户输入（ObjectiveUserInput）**：
真实用户提交给活动 Objective 的 append-only 权威原始事实。每条事实保存稳定 `input_id` 与完整 `UserMessage`，并可保存 `in_reply_to_message_id / related_outcome_id` 等关联；系统不拆分、概括或改写消息。agent 提问后的用户回复仍是一条新的 ObjectiveUserInput，关联信息只服务于存储、投影和确定性校验，给模型时仍按原始顺序渲染为普通文本。pause、resume、预算调整等控制命令不是用户语义消息，不产生 ObjectiveUserInput。`create` / `submit_user_input` 必须在同一命令路径内先使该消息进入 Session 可见历史，再提交 ObjectiveUserInput 与 outbox；禁止留下「日志有、历史无」。派发前若仍检测缺口，记不变量故障并停止对该 dispatch 空转重试，不在 dispatcher 内自愈补写历史。
_Avoid_：第二份用户要求模型、候选晋升、从自然语言抽取片段、把结构化事实对象暴露成模型消息协议、dispatcher 自愈补历史、history_gap 无限 claim

**用户输入外置授权（Objective Input Externalization）**：
用户通过结构化控制命令，明确允许指定超长 ObjectiveUserInput 改为以 Artifact 管理：原文物化到 `sessions/<session_id>/artifacts/` 下的稳定文件，ObjectiveLog 仍保留原始 ObjectiveUserInput；后续模型上下文在原位置只放入该 Artifact 的 path 与 summary，agent 按需读取全文。外置不是摘要、删除或目标修订。
_Avoid_：系统自动外置、静默摘要、覆盖原始用户事实、上下文只留无法读取的空引用

**LLM 调用成本上界（Call Cost Ceiling）**：
一次模型 attempt 在开始前可计算的美元成本上界，由本次完整请求 token 与该次调用配置的 `max_output_tokens`（及适用的 cache 计价）按当时价格快照得出。上界有限、可复现，并写入调试事实；它只用于启动检查，不写成 reserved，也不在调用结束后返还。
_Avoid_：Budget Reservation、按未知输出无限放大、把上界当成已占用额度

**LLM 调用前成本检查（LLM Preflight Cost Check）**：
objective-managed Run 在 BEFORE_LLM 中检查 `used_llm_cost_usd + call_cost_ceiling <= max_llm_cost_usd`，并确认 Objective 仍允许推进。任一条件失败则不调用 Provider，并触发预算暂停路径；成功只表示允许开始本次调用。系统不预留、不结算、不返还。顺序执行在上界正确时不因该规则超支；并行执行可能基于同一 `used` 同时通过检查，最坏超支被「单次调用成本上界 × 当时通过检查的并发调用数」封顶。
_Avoid_：ObjectiveActionLease、RunUsageLease、Budget Reservation、只检查 `used < limit` 却对超支无上界说明

**幂等命令回执（Command Receipt）**：
把命令作用域、idempotency key、规范化请求摘要和首次结果持久绑定的记录。同一作用域与 key、相同请求重放首次结果；同 key 不同请求产生冲突。Objective 创建使用 Session 级作用域，因此在 objective_id 产生前也能查重。
_Avoid_：只在 fact 上保存 key、同 key 不同 payload 静默复用、仅内存去重

**目标贡献（ObjectiveContribution）**：
agent 在 root Run 收尾时提交的、可能帮助后续工作且不具有用户输入权威性的发现。新建贡献只含 `content`（及可选 `summary`）；系统分配稳定 `contribution_id` 后原文不可变。对**已有**贡献的后续判断不改原文，而是另写 `contribution_annotations`：`{contribution_id, annotation, deactivate?, from, time}`，其中 `from`/`time` 由系统填充。`deactivate` 只影响是否进入活动上下文。
_Avoid_：ObjectiveUserInput、把 annotation 写成新贡献、自动生效的硬约束、晋升为用户事实

**目标修订提案（objective_update）**：
收尾 JSON 中可选的非权威全局理解更新，字段限于 `expected_revision`、`intent`、`scope`、`success_criteria`、`assumptions`、`sources`；无新理解时必须为 `null`。不得携带 RunPlan、Decision、预算或状态。由 ObjectiveService 校验后写入 `current_goal` 可变分析的新版本。
_Avoid_：运行中 runtime tool 改 Objective、无来源覆盖式 summary、与 Contribution 混写

**接力目标（Handoff Target）**：
Decision 指定的下一阶段职责，只能是 `agent`、`verifier` 或 `user`。它不指定具体 agent config、agent 名称或 pattern。
_Avoid_：ExecutorRef、具体执行实例、pattern 名称

**目标视图（ObjectiveView）**：
为 Objective 管理的 root Run 提供的当前 Objective 权威视图，包含当前全局目标、按原始顺序排列的全部 ObjectiveUserInput 及其外置授权（已外置项以 Artifact path/summary 表示）、仍有效的贡献、预算状态、文件 Artifact 清单、最近一次完整 Run Outcome 与本次触发信息。结构化视图只服务系统代码。模型上下文中的用户输入表示来自同一 Session 已提交历史（按 `input_id` 去重复用）；装配器只追加尾部系统内容，不重新拼装或中段补回用户话。
_Avoid_：把 Objective 权威状态藏在聊天历史中、按已删除的 Assignment 边界隔离上下文、复制全量 RunLog、把文本 report 塞进 Artifact 清单、因日志有历史无而中段插入用户消息

**验收轮次（Verification Attempt）**：
Verifier 对一个候选结果进行的一次独立 verification root Run。验收不通过时，原 work root Run 保持完成（已有 Run Outcome），并通过 handoff 创建新的 work root Run。
_Avoid_：回滚、重开旧 Run、验收 Assignment

**目标预算（ObjectiveBudget）**：
Objective 可以消耗的接力、验收、LLM 美元成本和活动时间配额及其当前用量。预算由执行系统检查和更新，agent 只能读取，不能修改。单个 Run 的循环上限和工具成本不属于 ObjectiveBudget。LLM 成本在每次 attempt 前做上界检查、在 attempt 后计入进程内 pending；`BudgetUsageRecorded` 仅在 root Run 边界以该 Run 的汇总 Δ 写入 ObjectiveLog（provenance `source=llm_run_total`）。attempt 明细只留在 RunLog/trace。门禁用的 `used` = 已投影汇总 + 当前进程内 pending。
_Avoid_：每个 LLM attempt 一条 BudgetUsageRecorded 作为领域叙事
_Avoid_：SystemCeiling、agent 自行扩容

**LLM 成本预算（max_llm_cost_usd）**：
Objective 中已收到有效响应的模型调用所累计的美元成本阈值，跨相继 root Run 计算。每次调用前执行 LLM 调用成本上界检查；调用后按请求 token 与实际接收 token 追加实际成本。不预留、不结算、不返还。
_Avoid_：Budget Reservation、reserved cost、max_cost、工具成本、ObjectiveActionLease

**目标活动时间（max_active_seconds）**：
在模型调用或 handoff 发生前，以 `checked_at - current_active_started_at` 计算的当前活动窗口时长。Objective 首次启动和每次从 checkpoint 恢复都会开启新的活动窗口，额度从零重新计算；并行分支共享同一个当前窗口，不分别累加。
_Avoid_：Run 执行耗时之和、跨 checkpoint 累计 active time

**目标活动窗口（Objective Active Window）**：
Objective 从首次启动或 checkpoint 恢复开始，到进入 `WAITING_USER`、`BUDGET_PAUSED` 或终态为止的一段持久化执行区间。`Objective.first_started_at` 保留首次启动事实，当前窗口起点由增量事实投影为 `current_active_started_at`，不能通过覆盖首次启动时间实现重置。等待区间仍记录在 Objective 上供审计，但不再参与跨窗口累计预算计算。
_Avoid_：覆盖 Objective 首次启动时间、跨窗口累计 max_active_seconds

**单次运行循环上限（max_steps_per_run）**：
`AgentOptions` 中的单 Run 正常模型调用阈值，用于在 agent 陷入极端循环时停止继续工作。主 assistant turn、上下文压缩、Provider retry attempt、终止总结和 Run 收尾等实际模型请求全部计数；达到阈值后只允许执行系统发起有限的总结与收口 attempt，因此最终调用数可以超过阈值。该值由开发者静态配置并由执行系统注入 Run，动态 workflow 中的模型只能读取、不能覆盖。root Run 收口后可通过 handoff 创建新的 peer root Run，不因此暂停 Objective。
_Avoid_：ObjectiveBudget.max_steps、跨 Run 的 step 总额度

**模型逻辑调用（Logical Model Call）**：
一次具有稳定业务目的的模型调用，拥有 logical_call_id 与不变 phase，例如 assistant、compaction、termination summary 或 run_finalization。Provider retry 不创建新 phase，而是在同一 logical call 下增加 attempt_no、retry_reason 与实际调用序号；每个 attempt 独立计数并按其实际响应记账。
_Avoid_：provider_retry phase、用 logical call 掩盖真实请求次数、retry 后丢失原收口语义

**预算暂停（Budget Pause）**：
Objective 达到预算阈值后进入的可恢复中断（与用户暂停/停止同一机制）。系统经短暂 DRAINING 禁止新动作、允许在途完成，再使活动 Run 进入 PAUSED；不写 Run Outcome、不结束当前 root Run。用户调整配额后按可恢复中断规则从同一 run_id 继续。
_Avoid_：agent Decision、Objective 失败、与用户暂停分成两套恢复协议

**全局收敛（DRAINING）**：
Objective 在进入可恢复中断前的短暂机械过渡：立即禁止新的模型调用、tool、child spawn、handoff、新的 peer root Run 与派发；已经开始的在途操作允许完成。随后活动 Run 进入 PAUSED。运行中用户输入**不**走 DRAINING，而是直接注入当前 root Run。
_Avoid_：语义 Decision、为 steering 使用 DRAINING、ObjectiveActionLease、未收敛就宣称已暂停

**可恢复中断（Recoverable Interrupt）**：
用户暂停/停止、预算触顶等需要停下但以后还能继续的控制。与不可恢复失败相对：当前 root Run 不结束、不产生 Run Outcome；同一 run_id 恢复。恢复时根据当前 llm context（由 RunLog/StepView 重建的消息列表）最后一项决定下一步：assistant 且有 tool_call → 执行这些 tool；assistant 且无 tool_call → 追加 `is_user_provided=false` 的 user 消息让模型继续；最后一项是 tool result → 同样追加 false user 让 loop 继续。用户可见的「停止」与预算暂停共用此机制；现有强杀式 cancel 不得再表示普通暂停。
_Avoid_：用 CANCELLED/FAILED 表示普通停止、为 pause 与 budget 维护两套 resume、恢复时新建 Run

**用户暂停（USER_PAUSED）**：
可恢复中断在 Objective 投影上的用户原因状态（预算原因可投影为 BUDGET_PAUSED）。机制与可恢复中断相同；恢复后开启新的活动时间窗口。
_Avoid_：CANCELLED 终态、Scheduler.cancel_subtree 实现暂停、创建新 Run 恢复

**运行检查点（Run Checkpoint）**：
可恢复中断时的恢复依据：以 RunLog 重建到当前安全边界的消息/step 视图为主；薄控制元数据可保存 last_committed_sequence 与配置/模板 hash。恢复下一步由消息列表最后一项形态决定（见可恢复中断），不依赖独立的复杂 resume_phase 状态机。
_Avoid_：复制完整 RunContext、无必要的新 peer root Run、fork、把 cancel 当 checkpoint

**LLM 实际成本记账（LLM Actual Cost Accounting）**：
模型 attempt 收到至少一个有效响应数据后，按完整请求 token 与实际接收的输出 token 计算并追加成本；未收到任何响应的失败记为零成本。相同 attempt 的记账以稳定 identity 幂等提交。调用前的成本上界只用于启动检查，不写入 reserved，也不在结束后结算或返还。
_Avoid_：Budget Reservation、ObjectiveActionLease、把 max_output_tokens 当成已占用额度、无响应失败收费

**执行故障（Execution Fault）**：
LLM、tool 或执行基础设施未能按契约完成一次操作的事实。它不同于 Verifier 判断结果不可交付的语义返工。
_Avoid_：Verification rejection、agent Decision

**重试判定（Retry Disposition）**：
执行系统对故障作出的 `retryable`、`non_retryable` 或 `outcome_unknown` 分类。只有操作可安全重复且故障为 retryable 时才能自动重试。
_Avoid_：根据错误文本猜测、所有失败一律重试

**重试耗尽接力（Retry Exhausted Handoff）**：
幂等且 retryable 的阻断性操作在当前 Run 内耗尽重试次数后，由执行系统根据结构化 fault 和最后已提交事实生成普通文本 report，并机械地产生 `HandoffDecision(target=agent)`。在同一 Session agent identity 下创建新的 work root Run 寻找其他执行方式，不要求失败的模型或工具再次完成总结。
_Avoid_：Objective 失败、复活旧 Run重试、为接力更换 agent identity、把系统 report 建成 Artifact

**结果未知（Outcome Unknown）**：
操作已经发出，但系统无法确认外部副作用是否发生的执行状态。执行系统禁止自动重试和 agent handoff，保存操作输入、响应证据与核验建议，并产生 `HandoffDecision(target=user, expects_reply=true)`；用户的语义回复仍原样成为新的 ObjectiveUserInput，并关联该 Outcome，后继 peer root Run 根据新增事实继续处理。
_Avoid_：假定成功、假定失败、自动重复有副作用的操作

**不可重试故障（Non-retryable Fault）**：
不应再次执行同一操作的结构化故障。普通 tool failure 若已成为当前 Run 可读取的 ToolResult，由 agent 选择替代方案；认证、配置、权限等阻断 Run 的故障由系统生成 report，并产生 `HandoffDecision(target=user, expects_reply=true)`。
_Avoid_：一律 Objective 失败、一律创建新 agent、重复相同操作
