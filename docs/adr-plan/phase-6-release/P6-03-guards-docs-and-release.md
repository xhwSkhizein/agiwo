# P6-03：完成架构护栏、文档与发布收口

状态：planned

## 目标

清除过渡命名和临时入口，用机器护栏固定最终依赖方向，更新项目文档和运维说明，并运行完整发布门禁。该任务不增加新行为，只确保已经实现的系统不会在后续维护中退化成多套真相或绕过 ObjectiveService。

## 对应决定

- 覆盖 ADR 0001 至 0045 的最终术语、依赖方向和不可回归约束。
- 以 ADR 0034 的深模块边界、ADR 0038 的唯一 RunPlan、ADR 0040 的存储配置和 ADR 0042 的输入来源作为机器护栏重点。

## 依赖

- P6-01、P6-02 已完成。

## 实施步骤

1. 全库扫描并删除旧领域名和配置：GoalState、GoalUpdate、GoalMilestonesUpdated、declare_milestones、enable_goal_directed_review、AgentOptions.max_steps，以及任何 Objective 语义下的 Task* 旧名。
2. 保留 Scheduler `TaskGuard/TaskLimits`，并在文档中说明它们只保护调度树，不是 ObjectiveBudget。
3. 更新 import-linter：SDK 不依赖 Console；agent 不依赖 scheduler/objective；scheduler 不依赖 objective；objective 只能依赖 scheduler public facade；Console 不导入 Objective 内部 store/aggregate。
4. 增加 repo guard/contract tests：普通 Session/Channel path 不直接 Scheduler route；ObjectiveStore 不复用 RunLog table；计划只有 RunPlan 一份；外部输入不能 false。
5. 更新 `AGENTS.md` 目录职责、public API、标准流程；保持包级说明，不列逐文件镜像。
6. 对照实际源码校正 `CONTEXT.md` 术语；ADR 保留历史决定，不改写为实现手册。若实现与 accepted ADR 冲突，先新 ADR 决定，不在计划文档中偷改。
7. 补充 Objective HTTP/SSE、pause/resume、budget adjustment、Session archive、模板配置和故障处理文档。
8. 写明开发数据清理要求：schema 直接更新，无 migration、无兼容读取；复述 P0 入口条件中的清理范围，并列出发布说明中需删除的本地 SQLite/`.agiwo` 位置与重建命令。
9. 更新公开 repository overview 生成规则，使 `agiwo/objective` 出现在模块职责中。
10. 检查所有日志 event_name、错误 code 和 Console 文案使用统一术语；普通用户不看到 closure、seq、outbox lease 等内部词。
11. 运行完整 SDK、Console、前端、Docker smoke、build/install 检查；保留结果作为发布记录。
12. 发布前再次核对 ADR 0001 至 0045 覆盖矩阵，为每个任务记录提交、测试和最终代码位置。

## 主要改动位置

- `lint/importlinter_agiwo.ini`
- `scripts/repo_guard.py`
- `scripts/generate_repo_overview.py`
- `AGENTS.md`
- `CONTEXT.md`
- `docs/` 下的 API、运行与清理说明
- architecture contract tests

## 测试计划

- 全库旧名称扫描和 public import surface 检查。
- import-linter 正反向依赖 contract。
- repo guard 对普通入口绕过 ObjectiveService、第二份计划数据和错误 provenance 的检测。
- 文档中的命令、路径、API 与当前源码逐项核对。
- 运行下面的完整 SDK、Console、前端、Docker、build 和 pre-push 门禁。

## 必跑命令

```bash
uv run python scripts/lint.py ci
uv run pytest tests/ -v
uv run python scripts/check.py console-tests
(cd console/web && npm run lint)
(cd console/web && npm test)
(cd console/web && npm run build)
uv run python scripts/smoke_console_docker.py
uv build
(cd console && uv build)
uv run python scripts/check.py pre-push
```

若打包元数据、workflow 或发布脚本在实施过程中发生变化，再运行对应 wheel smoke install。

## 完成标准

- import-linter 和 repo guard 能阻止主要边界回归。
- 45 份 ADR（`docs/adr/0001` 至 `0045`）在总方案矩阵中都有实现提交与测试证据。
- `AGENTS.md`、CONTEXT、API/Console 文档与源码当前行为一致。
- 旧开发数据不被静默误读，清理说明明确。
- 所有必跑命令通过，没有被跳过的失败测试。

## 风险与回退

不要为了“兼容旧数据”重新引入旧名称 alias、双读 fact、双写状态或遗留 kind 探测。MVP 约定是清理重建；保持一个当前契约，不承载历史债务。
