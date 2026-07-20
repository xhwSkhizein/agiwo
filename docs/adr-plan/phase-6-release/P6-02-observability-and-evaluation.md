# P6-02：完成观测、指标与 trajectory review 评估

状态：done

## 目标

让开发者能够解释一次 Objective 为什么形成当前状态、花费多少资源、在哪个边界暂停或接力，并评估默认开启的 trajectory review 是否带来实际收益。观测事实不反向成为领域真相。

**范围边界：** 本任务不是通用 Agent benchmark / evaluation 框架。`docs/eval-draft` 不在本次 Objective 重构范围内，本任务不得实现、依赖或对齐该目录中的方案。

## 对应决定

- ADR 0020：Objective Timeline 与 finalization 调试下钻。
- ADR 0043：trajectory review 是默认开启、低置信度实验元数据。
- ADR 0010、0011：预算、attempt、fault 和 recovery 可观测。

## 依赖

- P0-02、P3、P4、P5-04 已完成。

## 实施步骤

1. 建立 Objective 级指标：总时长、活动窗口、等待时长、Assignment/Run/handoff/verification 数、最终状态和交付次数。
2. 建立预算指标：limit/actual used、调用前阈值拒绝、response_observed、request/accepted output tokens、零价格配置、并发越界数和各 ModelCallPhase 成本。
3. 建立执行指标：outbox lease/attempt、dispatch latency、drain latency、checkpoint/resume、retry/backoff、recovery result 和 fault disposition。
4. finalization 调试视图可关联 RunLog call ordinal、prompt hash、parse/correction、Artifact/Outcome 和 Objective fact sequence。
5. trajectory review 指标包括：启用比例、调用次数、额外 token/cost/latency、aligned 比例、score 分布、unknown 比例和后续 Objective 结果。
6. 为同任务 review on/off 提供可重复的本地观测配置，默认生产配置仍为 true；不引入 eval-draft 式 suite/case/attempt 内核。
7. 分析 review 与完成率、verification reject、handoff、cost、latency 的相关性，但明确标记为相关而非因果。
8. compaction 记录是否读取 experimental score 以及最终判断；不实现固定权重或 score threshold。
9. prefix-cache 观测记录相邻 Assignment 的稳定前缀长度/cache read tokens，以及 compaction/rebuild 原因。
10. 日志继续遵守结构化 event_name；高基数字段如 prompt/content 不进入普通 metrics label，只在受控 Trace/RunLog 下钻。
11. 增加 dashboard/query 文档或 Console 聚合视图，说明每个数字的真相源。

## 主要改动位置

- `agiwo/observability/`
- `agiwo/agent/trace_writer.py`
- `agiwo/objective/` query/projection
- `console/server/services/metrics.py`
- `console/server/services/runtime/runtime_observability.py`
- Console objective/run debug components
- 与 trajectory review 相关的观测测试（非 eval-draft）

## 测试计划

- RunLog/ObjectiveLog -> Trace/metrics 投影一致性。
- 同一事实重放不重复计数。
- 正常响应、部分响应、无响应零成本、重启补账和并发越界 metrics。
- review enabled/disabled 指标与功能隔离。
- score 不触发 deterministic message deletion。
- prompt/content 不泄漏到低权限/普通 metrics API。
- 代码与测试不依赖 `docs/eval-draft` 或未来 evaluation 包路径。

## 完成标准

- 任一 Timeline 状态可下钻到产生它的 fact、Assignment 和 Run。
- 能量化 review 的额外成本与潜在结果关联。
- review score 仍是 optional experimental compaction input。
- prefix-cache 损失能关联到明确 compaction/rebuild，而非静默改写。
- 本任务交付物不包含通用 benchmark eval 框架。
- observability/Console tests 与门禁通过。

## 风险与回退

观测层只能投影 committed facts，不能创建“为了页面方便”的第二份状态。若 Trace 与 ObjectiveView 不一致，应修 projector 或事实写入，而不是在 UI 合并猜测。
