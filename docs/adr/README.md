# Architecture Decision Records

现行领域语言以仓库根目录 [`CONTEXT.md`](../../CONTEXT.md) 为准。本目录记录**已经做出的架构取舍**；过时决定保留文件，用 Status 标明 `superseded`，不要删改历史正文冒充现行规格。

**真相源顺序**：`CONTEXT.md` → 本目录现行阅读顺序中的 ADR → 源码。冲突时以源码与现行 `CONTEXT.md` 对齐；Status 为 superseded 的 ADR、`trash/`、历史施工计划**不是**规格。

## 现行阅读顺序（先读这些）

1. **[0048](./0048-session-run-waitset-minimal-core.md)** — Session + Run + Waitset 最小核心；Objective / Turn 聚合作废  
2. [0001](./0001-distributed-semantic-decisions-centralized-execution-control.md) — 语义决策 vs 执行控制（仍适用 Agent 内判断 vs 机械调度）  
3. [0002](./0002-separate-delegation-from-handoff.md) — 委派 vs 接力（核心只保留委派 / waitset；跨 root 自动接力账本已作废）  
4. [0042](./0042-user-input-records-whether-it-was-user-provided.md) — 用户输入来源标记  
5. [0043](./0043-trajectory-review-is-append-only-experimental-metadata.md) — 轨迹复盘  
6. [0045](./0045-artifact-indexes-session-files-only.md) — Artifact  
7. [0041](./0041-user-session-deletion-is-archive.md) — Session 归档（不再依赖活动 Objective 暂停前提）  

## 已被取代（勿再当现行规格）

| ADR | 取代者 | 摘要 |
| --- | --- | --- |
| 0003 Objective 是完整工作边界 | **0048** | 核心无跨 Run 任务账本 |
| 0004–0009 / 0015 Assignment 与 intake/验收 | **0048**（此前经 0046/0047） | Assignment / 强制验收链删除 |
| 0005 / 0006 强制 Decision / 收尾 LLM | **0048**（此前经 0047） | 无跨 Run 控制面 Decision |
| 0010–0014 / 0028–0033 Objective 预算与暂停 | **0048** | 预算 / DRAINING / Objective 暂停移出核心 |
| 0018 ObjectiveLog | **0048** | RunLog 为唯一执行账本 |
| 0019 Outbox | 0047 后由 **0048** | 无 RootRunRequested 派发账本 |
| 0020–0022 Objective 时间线 / Gateway API / SSE | **0048** | Session/Run 视图为主路径 |
| 0023–0027 / 0031 / 0038 Assignment 生命周期 | **0048** | Assignment 已删且禁止回潮 |
| 0034 / 0044 Objective 深模块与控制权 | **0048** | 无 ObjectiveService |
| 0035 Session 含相继 Objective | **0048** | Session 只含相继 root Run |
| 0039 Objective 修订时机 | **0048** | 无 Objective 修订 |
| 0040 Objective store 跟随 RunLog 配置 | **0048** | 无 Objective store |
| 0046 Objective–Run 无 Assignment | **0048** | Objective 本身移出核心 |
| 0047 Session 默认 + Objective 升级 | **0048** | 无升级路径；仅 Session→root Run |

Status 行已写 `superseded by …` 的文件仍保留全文，供追溯。实现与评审不得引用它们作为现行行为依据。

## 历史建成计划

`docs/adr-plan/` 已移至 `trash/adr-plan-historical-2026-07-21/adr-plan/`。

现行实现与文档以 `CONTEXT.md` + 上表阅读顺序为准。
