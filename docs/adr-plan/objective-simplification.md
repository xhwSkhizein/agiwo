# Objective 简化实施清单（ADR 0046）

冻结日期：2026-07-20。  
依据：grilling 结论、`CONTEXT.md` 更新、[ADR 0046](../adr/0046-objective-run-model-without-assignment.md)。  
**实现前**清空 `console/.agiwo`（及本地 Objective 相关开发状态）；无 migration。

## 进度（2026-07-20）

| PR | 状态 |
| --- | --- |
| S0 清库与护栏 | **done**（开发库已移至 `trash/console-agiwo-backup-2026-07-20/`；AGENTS.md 已更新） |
| S1 领域模型去 Assignment | **done** |
| S2 Store/outbox 挂 run_id | **done** |
| S3 Service 主链 / 收尾分档 / 薄输入 | **done**（复杂度评估可选注入；Console 默认注入 complexity_model） |
| S4 history 契约 | **done**（先写 Session 历史；gap → FAILED 停空转） |
| S5 预算 Run 边界汇总 | **done**（attempt 内存累计；Run 边界 `llm_run_total` 单条 fact） |
| S6 Scheduler 窄 facade | **done**（`dispatch_execution`；root `assignment_id=None`） |
| S7 Console / 模板 / 前端 | **done**（模板无 intake；API/表单已对齐；HTTP DTO `root_runs`） |

回归：`tests/objective/`；`scripts/check.py console-tests`；`console/web` session page tests。

## 已冻结规则（摘要）

1. **删除 Assignment**；公开 `Session → Objective → Run`；Outcome / outbox / Decision 挂 `run_id`。
2. **Run Role**：`work` | `verification`（无 intake）。
3. **`verification_required` Latch**：任一 root `RunPlan` 出现过 ≥1 个 milestone（**任意状态**）→ true，只增不减；入口分数不置位。
4. **入口复杂度评估**：create 后非关键打分 0–10，每 Objective 一次；`>3` 仅首个 Run 加规划建议 notice；失败按简单。
5. **收尾**：每 root Run 必跑；两档 schema；简单失败→交付且无 correction；复杂至多一次 correction 再机械 fallback。
6. **UserInput**：先 Session 历史再 ObjectiveLog；history_gap 停空转。
7. **用量**：Run/Outcome 边界汇总进 ObjectiveLog。
8. **Run Input**：简单薄 / 复杂或 verification 厚。
9. **Scheduler**：Objective 只依赖窄 facade。

## 关键回归场景

- [x] 简单问答路径单测：work 可直接交付
- [x] `verification_required` 强制 verifier
- [x] complexity notice 阈值（mock）
- [x] history co-write / gap → FAILED
- [x] E2E：真实 `update_plan` → Latch → 验收链
- [x] Console 注入 `complexity_model` 后 live 打分
- [x] 去掉 agent `RunExecutionRequest.assignment_id` shim
- [x] Startup 全库 `llm_run_total` reconcile（`list_objective_ids`）
- [x] 派发 verification 不再双写 Latch；权威在 Decision 前读 RunLog
- [x] 收尾 JSON 不含 `report`；Outcome.report = 工作正文
- [x] `AssignmentRole` → `RunTreeRole`；相位 `run_finalization`

## 与旧 adr-plan 关系

`docs/adr-plan/phase-*` 描述的是 Assignment 时代的建成路径，已完成。本文件是 **建成后的简化改造**；执行时以 ADR 0046 + 本清单 + `CONTEXT.md` 为准。
