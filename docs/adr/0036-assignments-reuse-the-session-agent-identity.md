# Assignment 复用 Session Agent Identity

一个 Session 只拥有一个稳定的 persistent root agent identity。该 Session 内不同 Objective 和 Assignment 不再为承担主责任的 root Run 实例化独立 agent identity，而是分别创建新的 Assignment 与 root Run，并以 `assignment_id / run_id` 标识责任和执行边界。Session 历史继续可供新 root Run 使用，但是否原样保留、压缩或排除，必须依据 Objective 当前全局目标与本次 Assignment 触发信息的相关性动态决定，不能由 Assignment 边界固定隔离。Assignment 内受委派的 child agent 不受 identity 复用规则限制，继续使用现有独立 identity。

## Status

accepted（术语：Assignment → Objective 管理的 root Run；identity 复用规则仍有效，见 ADR-0046）

## Considered Options

- 每个 Assignment 使用独立 agent identity：能得到天然隔离的上下文，但会跳过当前按 `session_id + agent_id` 加载的会话历史，使多轮 Session 的连续性丢失。
- 同一 agent identity，但只注入 ObjectiveView：上下文更可控，却需要另造一套历史筛选路径，并改变当前 persistent root 的基本语义。
- 每逢新 Assignment 就排除上一项 Assignment 的执行历史：边界简单，但会机械丢弃与最新用户输入或当前目标高度相关的证据。
- 复用同一个旧 Run：能够完整保留内存状态，但会破坏 Assignment 的责任终态、Outcome 唯一性和 Run 级循环边界。

## Consequences

- Session.id 继续作为 Console objective-managed 路径中 root agent/state 的稳定 identity；Assignment id 和 root Run id 分别承担责任标识与执行标识。
- handoff、verification rejection 和 `max_steps_per_run` 触发时结束旧 Assignment/Run，并在同一 agent identity 下创建新 Assignment/Run；不得恢复已经结束的旧 Run。运行中用户输入不结束 Run（ADR 0023）。
- 相同 `session_id + agent_id` 使 Session 历史成为新 Run 的候选上下文，而不是要求无条件把全部历史发送给模型。
- 跨 Run 上下文装配以前缀缓存命中率为首要约束。已经提交并进入模型历史的消息保持原有顺序与内容，不因 Assignment 边界删除、改写或搬移；新的 Assignment Input、Outcome 和 Run 边界说明只追加在稳定历史之后，使 Provider 能复用尽可能长的相同前缀。
- 每次装配以 Objective 当前全局目标和本次触发信息为相关性锚点。入口 Assignment 的触发信息通常是最新用户输入；handoff 或 verification Assignment 的触发信息通常是 Decision、最近 Outcome 或候选交付结果。运行中用户输入走注入路径，不创建新 Assignment。
- 活动 Objective 的用户输入进入后继 Assignment 上下文的路径是：它们已作为 canonical user message 存在于同一 Session agent 的已提交历史中（首次 intake、WAITING_USER 回复派发、或运行中注入同一 Run）。跨 Assignment 装配**不**重新拼装、不重排、不向消息中段插入「缺失」用户话；按 `input_id` 去重复用稳定前缀，新的 Assignment Input / Outcome / 边界说明只追加尾部。已授权外置的输入在历史中对应位置只保留 Artifact 的 path 与 summary（外置当时改写的是该条表示，不是事后中段插入）。若 ObjectiveLog 中存在某条未外置 `ObjectiveUserInput`，而候选历史中完全找不到对应 `input_id` 的消息，视为不变量破坏并产生结构化故障，禁止静默补回。assistant/tool 历史较短且相关时可原样复用；长度/冲突/噪声时由 Context Optimization 优先压缩 assistant/tool，不得删除或改写用户输入表示。
- 若未外置 ObjectiveUserInput 本身已经超过模型物理上下文上限，系统进入 WAITING_USER，并只允许用户通过结构化命令授权指定输入外置或结束 Objective；不能用静默摘要换取继续执行。授权后必须重新检查容量，仍超限则继续等待。
- Context Optimization 的完整输入、输出、选择依据和模型用量必须持久化以便调试。它改变的是本次模型输入视图，不删除 RunLog 或 Session 历史事实。
- Parallel、Pipeline、Agent 或 tool 委派创建的 child agent 保留当前独立 identity 和上下文语义；它们属于当前 Assignment 的执行树，不形成新的 Assignment 或 handoff。
- Objective 随用户输入持续投影当前全局目标。ObjectiveView 仍必须显式注入，因为聊天历史不是该目标、ObjectiveUserInput、ObjectiveContribution、ObjectiveBudget 或最近 Outcome 的权威投影。
- verifier 与 worker 可以使用同一 Session 中与当前验收相关的历史，不能再声称通过固定上下文隔离消除了自我审查偏差；验收独立性来自单独 Assignment、固定验收模板和不可绕过的 Decision。
- Assignment 收尾调用的系统模板和结构化输出仍按既有决定从普通对话上下文隐藏，避免内部控制消息污染后续 Run。
- 每个新 Assignment 的职责模板由系统渲染为当前 Run 临时使用的 `<system-notice>`；notice 不成为普通 Session message，Assignment 的结构化状态仍保存在 Objective facts 中。
- 新 Assignment Input 必须在消息尾部明确标记新 Run 边界，并说明历史 `update_plan` 调用只属于已结束 Run、不代表当前 `RunPlan`；当前 Run 从空计划开始，需要延续的责任以最近 Outcome 的 `carry_forward` 为准。该尾部声明消除旧计划的语义歧义，但不牺牲历史前缀缓存。
- 当前跨 Run compaction 没有把已持久化的压缩摘要恢复到下一 Run，已经由最小复现确认。因为本决定依赖跨 Run 历史连续性，该 bug 必须在 Objective 集成前修复并增加回归测试。

本决定修正 ADR 0004、0008、0009、0011、0012、0015、0023、0025 和 0035 中关于“全新 agent identity / 干净上下文”的旧描述，也否定按 Assignment 边界默认隔离历史的规则。
