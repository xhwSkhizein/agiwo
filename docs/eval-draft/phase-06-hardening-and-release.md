# Phase 06：Hardening、复现与发布

状态：Planned

## 目标

确保 benchmark 结果可以被复核、比较和安全地重复运行。只有这一 phase 结束后，evaluation 才适合成为稳定的公共能力或 CI 门禁。

## 文件计划

### 创建

- `tests/evaluation/test_reproducibility.py`
- `tests/evaluation/test_cancellation_and_cleanup.py`
- `tests/evaluation/test_concurrency_limits.py`
- `tests/evaluation/test_report_schema.py`
- `scripts/check_evaluation_environment.py`
- `docs/guides/evaluation.md`
- `docs/api/evaluation.md`

### 修改

- `AGENTS.md`，补充稳定的 evaluation 目录职责和标准检查命令。
- `docs/README.md`，把已稳定的 guide/API 加入正式文档入口。
- `pyproject.toml`，只在需要 optional dependency group 时修改。
- `.gitignore`、`.dockerignore`，确认 evaluation artifact、缓存和外部环境文件不会误入发布包。

## 复现要求

每个 report 必须带：

- suite id/version/commit。
- evaluator version。
- agent id/config fingerprint。
- model/provider/version。
- tool schema fingerprint。
- environment image/version。
- seed、attempts、并发和预算。
- 运行开始和结束时间。
- 网络和 credential policy。

缺少关键 fingerprint 时，报告应标记为 non-reproducible，而不是正常进入比较榜单。

## 可靠性与资源

- 外部环境使用显式并发上限。
- 每个 attempt 有 hard timeout 和 cleanup timeout。
- provider 重试不得改变 attempt 语义；记录 retry 次数。
- 资源超限应标记为 budget/infra failure，而不是普通任务失败。
- SQLite 写入、artifact 写入和环境 teardown 需要故障注入测试。
- 大型 raw artifact 支持 retention policy 和脱敏策略。

## 防止错误比较

- 报告默认按 suite/category 展开，不生成跨 benchmark 总榜。
- 变更 prompt、工具、模型或环境时，自动生成新的 config fingerprint。
- suite case 变更时递增 suite version；不能静默替换旧 case。
- hidden test、公开 test 和开发 test 必须分开标记。
- 多次运行使用 case-level paired comparison 时，保留相同的 case selection。

## CI 分层

### Pull request

- contract tests
- local deterministic suite smoke
- aggregation and report schema tests
- lint、compileall 和 `git diff --check`

### Nightly

- 多 seed local suite
- provider-backed small suite
- failure taxonomy regression
- artifact redaction scan

### Manual / scheduled external

- BrowserGym、SWE-bench、OSWorld、AndroidWorld 等高成本环境。
- 明确记录运行版本、资源、耗时和成本。

## 最终验收标准

- 新增一个 suite 不需要修改 `EvaluationRunner`。
- 新增一个 grader 不需要修改 Agent 主循环。
- 本地 suite 在无外网和无 provider key 时可以完整测试。
- 任何 case 都能从报告跳到 attempt、run、trace 和原始 grade。
- 失败结果可以从 raw attempt 重新聚合得到。
- 运行取消后不存在未回收的外部资源。
- 文档、AGENTS.md、CLI、Console 和实际代码对 phase 结束状态一致。

