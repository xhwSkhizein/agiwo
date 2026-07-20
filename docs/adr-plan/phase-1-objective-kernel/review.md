# P1 Objective 内核：讲解结论记录

审查/讲解日期：2026-07-18。

本文记录对 `docs/adr-plan/phase-1-objective-kernel/` 设计方案的讲解结论。它不是新的 ADR，也不替代任务文档；用途是把「为什么有这一层、它管什么、这一阶段刻意不做什么」写成可复查的共识。

---

## 1. 问题从哪来

Agiwo 今天已经能让 Agent 跑一轮对话，也能用 Scheduler 调度父子 Agent、等待、取消。但这些能力回答的是「一次执行怎么跑」，而不是「用户交给系统的一整件工作从开始到交付如何被拥有」。

现有缺口大致是：

- Console 的 Session 可以跨很多轮，却没有「当前这件工作」的正式边界。
- Scheduler 里的 `AgentState.task` 只是调度输入，不是全局目标。
- Run 的状态只有跑起来/结束这类执行态，表达不了「等用户」「预算暂停」「整件工作完成」等跨多次执行的状态。
- 普通用户入口仍直接进 Scheduler；在 Objective 主链未完成前，刻意不切换流量。

P1 要做的，不是马上改用户入口，而是先在 Scheduler 之上建一层可信的领域内核：有自己的语言、自己的日志、自己的存储事务，以及唯一对外门面。

---

## 2. 四层概念（必须分清）

从外到内：

| 概念 | 一句话 | 例子 |
| --- | --- | --- |
| Session | 用户看到的长期对话容器 | 一次控制台会话 |
| Objective | 一次完整工作：从用户目标到接受/失败/交还用户 | 「帮我写完这份报告并验收」 |
| Assignment | Objective 内的一段局部责任 | intake / work / verification |
| Run | Agent 的一次实际执行 | Scheduler 派发后的 root/child Run 树 |

基数约束（P1 出口必须成立）：

1. 同一 Session 最多一个非终态 Objective。
2. 同一 Objective 最多一个非终态 Assignment。
3. 并行只存在于某个 Assignment 的 root/child Run 树里，不出现「两个活动 Objective」或「两个活动 Assignment」。

终态（COMPLETED / FAILED；Assignment 还有 INTERRUPTED）不可重开；只有暂停类状态可恢复。

---

## 3. P1 在总路线中的位置

总方案分 P0–P6。P0 先把 Agent 运行时里和 Objective 冲突的语义清干净（RunPlan、输入来源、模型调用账本等）。P1 在此之上建立内核与存储。

```text
用户命令 → ObjectiveService →（以后）Scheduler → Agent
                ↓
         ObjectiveStore
         （facts + slot + receipt + outbox）
```

稳定依赖方向：`objective -> scheduler -> agent`。Agent 与 Scheduler 不得导入 `agiwo.objective`。

P1 **不**做：模板渲染、真正派发 Scheduler、Assignment 收尾、HTTP/SSE、切换 Console/渠道入口。那些属于 P2 及之后。

---

## 4. 四个任务各自解决什么

### P1-01：领域模型

定义纯语言与不变量，不碰存储、不碰 Scheduler、不让模型参与状态判断。

核心对象包括：

- **ObjectiveStatus**（八态）：CREATED、RUNNING、DRAINING、WAITING_USER、BUDGET_PAUSED、USER_PAUSED、COMPLETED、FAILED。
- **AssignmentStatus**（六态）与 kind：intake / work / verification。
- **HandoffDecision**：target 封闭为 agent / verifier / user；不把具体 agent 配置塞进 Decision。
- **Artifact**：只索引 Session 工作目录下的文件型产出；普通文本 report 不是 Artifact。
- **AssignmentOutcome**：终态 Assignment 的结构化收尾（含 report、可选 ArtifactRefs、Contributions、可选 Decision）。
- **ObjectiveUserInput**：真实用户语义输入的权威事实（完整 UserMessage）；pause/resume/budget 等控制命令不是语义输入。
- **Contribution / annotation**：系统侧累积贡献与批注。
- **current_goal**：不可变用户事实 + 可修订分析；分析不得夹带 RunPlan、预算或 handoff。
- **ObjectiveBudget** 四维：handoffs、verification_attempts、llm_cost_usd、active_seconds；每维只有 limit/used，没有 reserved。

### P1-02：ObjectiveLog 与投影

所有领域变化必须成为 append-only fact。没有写进 ObjectiveLog 的内存改动，没有领域效力。

- fact 带严格递增 sequence；投影器按 sequence 重建 ObjectiveView / AssignmentView / BudgetView / Timeline。
- 投影器对缺口、重复、未知引用、非法转换 fast-fail，不猜测修复。
- Assignment 终态必须与唯一 Outcome 同一领域提交；PAUSED 不接受 Outcome。
- ObjectiveLog 只存引用（assignment_id / run_id / artifact_id），不复制 Run 消息、tool call 或 token delta。
- 命令幂等不靠 fact 上挂一个 key 解决，而由 P1-03 的 command receipt 承担。

### P1-03：ObjectiveStore 与事务 Outbox

一次命令必须在同一事务里提交：facts、Session slot、command receipt、待派发 outbox。避免「日志写了但没人执行」或「两个活动 Objective 抢同一 Session」。

关键机制：

- **Session slot**：`session_id` 唯一占用；create 时取得，终态时释放；不能依赖先查后写。
- **command receipt**：scope + idempotency_key + request hash；同 key 同 hash 重放首次响应，同 key 不同 hash 冲突。
- **DispatchRequested outbox**：带 lease 的待派发记录；P1 只持久化，真正 dispatcher 在 P2。
- **存储配置**：与 RunLog 共用物理配置（memory/SQLite），但独立表、不互相读内部实现；Mongo 等未实现 backend fail-closed，禁止 silent 降级。

### P1-04：ObjectiveService 深模块边界

对外只暴露一个门面：`ObjectiveService` + 稳定请求/响应 DTO。调用者提交命令、读取视图，不必理解日志重放、outbox、投影或 store codec。

- create / get / list / submit / externalize / pause / resume / adjust budget 等用例骨架落在 Service。
- 未接通执行的命令返回明确不可用，不做隐藏副作用。
- import-linter 固化依赖方向；`AGENTS.md` 只写包级职责。
- 刻意不提前拆出 `ObjectiveEngine`、`CommandHandler`、`AssignmentRunner` 等平行公开层——当前只有一种执行实现时，多层抽象会变成浅模块。

---

## 5. 我们达成的共识（讲解结论）

1. **P1 的产品目标是「可信内核」，不是「用户可见功能」。** 成功标准是：只靠 ObjectiveLog 能重建状态；Session 基数与幂等在存储层原子成立；外部只能通过 ObjectiveService 进入。
2. **Objective 与 Run/Scheduler 是不同真相源。** Run 内事实进 RunLog；跨 Assignment 的工作事实进 ObjectiveLog；AgentState 只是可重建的执行快照。
3. **控制权唯一。** 只有 ObjectiveService 拥有 Decision 消费、Budget、Session 活动 slot、command receipt 与 ObjectiveLog 写入。Scheduler 是下层机械执行者，不得反向读 Objective。
4. **先事实、后副作用。** facts + slot + receipt + outbox 同事务；没有「先写日志、后补 outbox」的降级路径。
5. **用户原文不可被系统偷换。** 每条真实用户语义消息对应一条完整 ObjectiveUserInput；外置必须用户显式授权并引用 input_id；控制命令不伪装成用户输入。
6. **Artifact 很窄。** 文件索引而已；主报告与故障说明仍是普通文本，不得建成 Artifact，也不得夹带路由字段。
7. **Budget 不是 TaskGuard。** Scheduler 的 spawn/wake 护栏继续存在；ObjectiveBudget 是跨 Assignment 的全局硬边界，由 Objective 侧拥有。
8. **回退成本可控。** P1 未切换用户流量；若内核不合格，可删整个 `agiwo.objective` 包而不改坏现有 Agent/Scheduler 路径。

---

## 6. 阶段出口检查清单（验收时对照）

验收日期：2026-07-18。结论：P1 出口全部通过，阶段标记 **done**。

- [x] Objective / Assignment / Run / Session 在类型与状态机中不混用
- [x] 当前状态可仅靠 ObjectiveLog 重建（Outcome 派生贡献/目标修订经 `expand_outcome_derived_facts` 落成独立 facts）
- [x] Session ≤1 非终态 Objective；Objective ≤1 非终态 Assignment
- [x] Session slot 在 memory/SQLite 中原子保证，非先查后写；非终态释放被 store 拒绝
- [x] 写命令均有持久化 receipt（scope / key / hash / 首次响应）
- [x] facts 与 outbox 可原子提交；`AssignmentCreated` 同事务要求 `DispatchRequested`
- [x] ObjectiveStore 与 RunLog 同物理配置、无共享表、无互相读实现
- [x] 公开面为 ObjectiveService、稳定 DTO 与接线用 store factory；import-linter 禁止 agent/scheduler → objective

---

## 7. 刻意留给后续阶段的事

| 能力 | 阶段 |
| --- | --- |
| Assignment 模板、预分配 Run identity、真正 outbox dispatcher | P2 |
| intake → work → verification → delivered 闭环 | P2 |
| 预算账本、活动时间、checkpoint、DRAINING、崩溃对账 | P3 |
| 重试、接力、运行中用户输入 | P4 |
| HTTP/SSE、Console/渠道切到 ObjectiveService、时间线 UI | P5 |
| E2E 恢复矩阵与发布门禁 | P6 |

---

## 8. 与总方案 review 的关系

仓库根下还有 `docs/adr-plan/review.md`，那是对照 ADR 全集的缺陷审查（R1–R16）。本文是 **P1 阶段讲解结论**，关注「这一阶段设计意图是否说清楚」，不重复那些跨阶段缺陷编号。若讲解中发现与 ADR 冲突，应先开项到总 review 或直接修订任务文档，而不是在本文件另立第二套决策。
