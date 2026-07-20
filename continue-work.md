# 当前开发进展与交接说明

## 背景与目标

**P1–P6 已完成**：

- Phase 1：Objective Kernel  
- Phase 2：Assignment Mainline  
- Phase 3：预算与可恢复中断  
- Phase 4：故障与运行中用户输入  
- Phase 5：Objective Gateway 与 Console（`docs/adr-plan/phase-5-gateway-and-console`）  
- Phase 6：系统验收与发布收口（`docs/adr-plan/phase-6-release`）

约束仍有效：

- Agent 不得 import `agiwo.objective`；预算门禁经注入的 `LlmBudgetGate`。
- `dispatched ≠ completed`；pause ≠ cancel；运行中用户输入 ≠ DRAINING。
- 普通 Web/渠道入口经 `SessionObjectiveGateway` → `ObjectiveService`，不再直接 `Scheduler.route_root_input`（debug scheduler API 仍可用）。

## P6 交付要点

| 任务 | 状态 | 说明 |
|---|---|---|
| P6-01 | done | E2E 矩阵（slot / WAITING_USER 续派 / crash redispatch & COMPLETE_OUTBOX / pause·幂等）+ 既有 objective/agent/scheduler 专项测试；`tests/objective/e2e/test_coverage_matrix.py` 锁定 owner 模块 |
| P6-02 | done | `project_objective_metrics` + `GET /api/objectives/{id}/metrics`；`docs/objective-observability.md`（review/prefix-cache 走 Trace/RunLog，非第二真相） |
| P6-03 | done | import-linter / repo_guard AGW045–047、API/清理文档、`AGENTS.md` Console 入口更正、发布门禁 |

## 下一步

Objective 重构主线已收口。后续优先：运维观察、prefix-cache/review 成本看板加深、以及未覆盖 crash 边界的增量回归。

## 验证记录（本机）

- `uv run python scripts/lint.py ci`：通过  
- `pytest tests/`：794 passed（需非沙箱，PTY 测试依赖本机权限）  
- `scripts/check.py console-tests`：245 passed  
- `console/web` lint/test/build：通过  
- `uv build` + `console/uv build`：通过  
- `scripts/check.py pre-push`：通过  
- `smoke_console_docker.py`：本机 Docker daemon 未运行，未执行；有 Docker 时请补跑

## 验证命令

```bash
uv run pytest tests/ -q
uv run python scripts/lint.py ci
uv run python scripts/check.py console-tests
(cd console/web && npm run lint && npm test && npm run build)
uv run python scripts/smoke_console_docker.py
uv run python scripts/check.py pre-push
```
