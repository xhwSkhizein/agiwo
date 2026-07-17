# ObjectiveService 拥有 Objective 控制，Scheduler 拥有 Run 执行

ObjectiveService 是 Objective 级执行控制的唯一 owner：它消费 AssignmentOutcome 与 Decision，校验并更新 ObjectiveBudget，在 ObjectiveStore 事务中追加 ObjectiveLog facts、维护 Session 的活动 Objective 占用，并创建 Assignment 与 outbox。Scheduler 是 Run/agent 级机械调度者：它维护 AgentState、执行 root/child Run、spawn、wake、wait、pause 和 shutdown，但不读取 ObjectiveLog、不解释 Decision，也不写 ObjectiveStore。Agent 与 Scheduler 在真正开始 LLM、tool、child spawn 或 dispatch 前，通过 Objective 模块提供的窄 hook/门禁接口确认“仍可推进”并（对 LLM）完成调用前成本检查；它们负责在执行点拒绝不可推进动作，却不因此取得 Objective 状态或预算所有权。

## Status

accepted

## Considered Options

- 让 Scheduler 同时拥有 ObjectiveStore、Decision 与 ObjectiveBudget：全局控制集中在一个类中，但会使 Scheduler 同时理解 Objective 领域状态和 Run 调度状态，破坏 `objective -> scheduler -> agent` 的依赖方向，并与 ObjectiveService 深模块形成两个入口。
- ObjectiveService 只保存数据，Scheduler 独立消费 Decision 和预算：表面分工明确，但同一 Assignment 状态迁移需要两个 owner 协调，无法保证 Outcome、预算、下一 Assignment 和 outbox 原子提交。
- ObjectiveService 独占 Objective 控制，Scheduler 独占 Run 执行：Objective 事务和 Run 调度各有唯一 owner；执行点通过不含 Objective 语义的门禁接口连接，复杂度留在 Objective 深模块内部。
- 用持久化 ObjectiveActionLease 连接执行点与 DRAINING barrier：曾用于消除 check-then-act，但把预算无关的租约生命周期强加给所有动作路径；已废弃，改用状态门禁 + Run checkpoint/Outcome barrier，LLM 成本改用调用前上界检查。

## Consequences

- `ObjectiveService` 是 ObjectiveLog、ObjectiveStore、ObjectiveBudget、Session 活动 Objective 占用、Assignment 生命周期和 Objective command receipt 的唯一写入入口。
- 只有 ObjectiveService 可以接受 Decision，并在同一 ObjectiveStore 事务中提交旧 Assignment Outcome、预算消费、下一 Assignment 和 DispatchRequested。
- Scheduler 不接受或解释 Decision，不直接检查或结算 handoff/verification/LLM cost/active-time 配额，也不写 ObjectiveLog 或 outbox。
- Scheduler 继续拥有 Run tree 与 AgentStateStorage；对 Objective 暴露的机械 facade（派发、树/run 查询、可恢复中断、注入等）在集成前一次性定义（见 adr-plan P2-02），后续只填实现、不平行扩面。`TaskGuard` 继续只保护 max depth、children、wake 等 Scheduler 本地限制，不扩展为 ObjectiveBudget owner。
- Objective 模块通过 Agent/Scheduler 已有 public hook 或窄门禁接口，在外部动作真正开始前回答 allow/deny。接口不把 ObjectiveView、Budget aggregate 或 Store 泄漏给下层，也不签发 ObjectiveActionLease。
- LLM 路径在 BEFORE_LLM 中执行调用前成本检查（`used + call_cost_ceiling <= limit`）并确认 Objective 仍可推进；tool、child spawn 和 dispatch 在真正开始前确认可推进状态。失败则拒绝开始；成功不预留费用。
- 进入 DRAINING 后，门禁拒绝新动作；已经开始的在途操作允许完成。barrier 由当时活动 Run / Assignment 收敛到 checkpoint 或 Outcome，不维护租约集合。
- 稳定依赖仍是 `ObjectiveService -> Scheduler -> Agent`。Objective 模块实现下层定义的 hook/门禁接口；Scheduler 与 Agent 不导入 `agiwo.objective`。
- 本 ADR 部分取代 ADR 0001 中“Scheduler 是全部执行控制与全局硬限制唯一 owner”的表述；ADR 0001 的分布式语义决策原则仍然成立。
- 本 ADR 部分取代 ADR 0010 中“Scheduler 接受 Decision 并结算 handoff/verification 配额”的表述；这些事务归 ObjectiveService，Scheduler/Agent 只在动作发生点执行门禁结果。
- 本 ADR 部分取代 ADR 0019 首段中“Scheduler 接受 Decision 并写 ObjectiveStore”的表述；ObjectiveService 写事务，dispatcher 再通过 Scheduler facade 执行已确定的 outbox record。
- 本 ADR 废弃早期草案中的 ObjectiveActionLease / RunUsageLease / Budget Reservation；LLM 成本边界以 ADR 0010、0014 的调用前上界检查为准，DRAINING 以 ADR 0028 的状态门禁与 Run barrier 为准。
