# Architecture Decision Records

现行领域语言以仓库根目录 [`CONTEXT.md`](../../CONTEXT.md) 为准。本目录**只保留仍指导实现的 ADR**。

**真相源顺序**：`CONTEXT.md` → 本目录下列 ADR → 源码。冲突时以源码与现行 `CONTEXT.md` 对齐。

## 现行阅读顺序

1. **[0048](./0048-session-run-waitset-minimal-core.md)** — Session + Run + Waitset 最小核心  
2. [0001](./0001-distributed-semantic-decisions-centralized-execution-control.md) — 语义决策 vs 执行控制  
3. [0002](./0002-separate-delegation-from-handoff.md) — 委派 vs 接力（核心只保留委派 / waitset；跨 root 自动接力账本已作废，见 0048）  
4. [0042](./0042-user-input-records-whether-it-was-user-provided.md) — 用户输入来源标记  
5. [0043](./0043-trajectory-review-is-append-only-experimental-metadata.md) — 轨迹复盘  
6. [0045](./0045-artifact-indexes-session-files-only.md) — Artifact  
7. [0041](./0041-user-session-deletion-is-archive.md) — Session 归档  

## 已归档（勿当现行规格）

Objective / Assignment / 预算账本 / Outbox / Gateway SSE 等已被 **0048** 取代的 ADR，整批移至：

[`trash/adr-superseded-by-0048-2026-07-21/`](../../trash/adr-superseded-by-0048-2026-07-21/)

其中包括原 0003–0040、0044、0046、0047，以及带大量 Objective 表述的 pause/retry/fault 决定（如 0011、0016、0017、0029、0032、0033）。  
Run 级 pause / retry / fault 行为以**源码**为准；若需再成文，应写新的 ADR，不要复活归档正文。

历史建成计划：[`docs/adr-plan/README.md`](../adr-plan/README.md) → `trash/adr-plan-historical-2026-07-21/`。
