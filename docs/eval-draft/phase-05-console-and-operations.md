# Phase 05：CLI、Console 与运行运维

状态：Planned

## 目标

让评测从“命令行实验脚本”变成可查询、可取消、可追溯的工作流。Console 只做控制面和投影，不直接扫描 Agent 的底层 storage。

## CLI 文件计划

### 创建

- `agiwo/evaluation/cli.py`
- `scripts/run_evaluation.py`
- `scripts/list_evaluations.py`
- `tests/evaluation/test_cli.py`

CLI 至少支持：

- 列出 suite 和 version。
- 按 case/tag/split 选择任务。
- 设置 attempts、seed、并发和预算。
- dry-run 检查环境依赖。
- 启动 batch、查看状态、取消 batch。
- 输出 JSON report 和 case artifact 路径。

## Console 后端文件计划

### 创建

- `console/server/models/evaluation.py`
- `console/server/services/evaluation/registry.py`
- `console/server/services/evaluation/run_service.py`
- `console/server/services/evaluation/query_service.py`
- `console/server/routers/evaluations.py`
- `console/tests/test_evaluations_api.py`

### 修改

- `console/server/app.py`
- `console/server/dependencies.py`
- `console/server/models/__init__.py`
- `console/server/response_serialization.py`

建议 API：

```text
GET  /api/evaluations/suites
POST /api/evaluations/batches
GET  /api/evaluations/batches/{batch_id}
POST /api/evaluations/batches/{batch_id}/cancel
GET  /api/evaluations/batches/{batch_id}/cases
GET  /api/evaluations/cases/{evaluation_id}
GET  /api/evaluations/reports/{batch_id}
```

路由只做 HTTP 装配；批次生命周期、查询和报告组装分别放在 evaluation service 中。Console 不得直读 `scheduler.store`，同样不应直读 evaluation SQLite 表。

## Console 前端文件计划

### 创建

- `console/web/src/app/evaluations/page.tsx`
- `console/web/src/app/evaluations/[batchId]/page.tsx`
- `console/web/src/components/evaluation/batch-form.tsx`
- `console/web/src/components/evaluation/batch-summary.tsx`
- `console/web/src/components/evaluation/case-result-table.tsx`
- `console/web/src/components/evaluation/failure-breakdown.tsx`
- `console/web/src/components/evaluation/trace-link.tsx`
- `console/web/src/lib/evaluation-api.ts`
- 对应组件测试

第一版 UI 只需要：suite/version 选择、启动参数、进度、成功率、`pass^k`、成本/延迟、失败分类和 trace/case 详情链接。不要先做复杂的在线标注系统。

## 运维约束

- batch、attempt、case 和 agent run 使用不同状态枚举，不能混成一个 `status`。
- 取消只影响未完成 attempt；已完成 artifact 不得删除或重写。
- Console 显示 evaluator version、suite version 和 config fingerprint。
- artifact 访问需要脱敏和权限控制。
- 运行中的外部环境必须可查询和可回收。
- 页面显示“部分完成”时必须说明成功、失败、跳过和基础设施错误的数量。

## 测试与退出门槛

- API contract test 覆盖正常启动、失败、取消、空 suite 和分页。
- 查询服务不直接依赖路由 request object。
- 前端测试覆盖 loading、partial failure、empty state、long case id 和报告缺失。
- `cd console/web && npm run lint && npm test && npm run build`
- `uv run python scripts/check.py console-tests`

