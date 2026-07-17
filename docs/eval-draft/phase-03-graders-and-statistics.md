# Phase 03：Graders、轨迹和统计报告

状态：Planned

## 目标

把“运行完成”变成有解释力的评测结果。结果 grader 判断做成了什么，trajectory grader 判断怎样做的，statistics 模块判断结果是否稳定，report 模块负责以可比较的方式呈现它们。

## 文件计划

### 创建

- `agiwo/evaluation/grading/__init__.py`
- `agiwo/evaluation/grading/context.py`
- `agiwo/evaluation/grading/response.py`
- `agiwo/evaluation/grading/state.py`
- `agiwo/evaluation/grading/trajectory.py`
- `agiwo/evaluation/grading/safety.py`
- `agiwo/evaluation/grading/judge.py`
- `agiwo/evaluation/metrics.py`
- `agiwo/evaluation/aggregation.py`
- `agiwo/evaluation/failures.py`
- `tests/evaluation/test_graders.py`
- `tests/evaluation/test_metrics.py`
- `tests/evaluation/test_aggregation.py`

## Grader 接缝

### ResponseGrader

用于文本、JSON 和结构化答案：

- exact match
- normalized match
- JSON/schema match
- reference answer predicate
- 可选的引用完整性检查

### StateGrader

用于环境状态：

- 目标字段是否达到期望值。
- 禁止字段是否保持不变。
- 中间里程碑是否按要求完成。
- 是否触发危险或不可逆操作。

### TrajectoryGrader

从 `RunLog`/`Trace` 读取：

- tool selection
- argument validity
- call ordering
- repeated calls
- tool errors
- recovery
- steps/calls
- compaction、rollback、termination 等运行事实

它不得解析模型“思考文本”作为权威依据；应优先使用 committed tool steps、LLM call facts 和 runtime facts。

### SafetyGrader

至少支持：

- 不可信工具输出中的 prompt injection。
- 未授权工具调用。
- 违反环境政策的状态变化。
- 恶意请求是否被拒绝。
- 正常任务是否因为安全策略被过度拒绝。

### JudgeGrader

仅处理开放式答案质量，例如研究报告完整性、解释清晰度和用户偏好。必须保存 rubric、judge model/version、原始 judge payload 和校准信息。Judge 不能覆盖 `StateGrader` 或 `SafetyGrader` 的程序事实。

## 统计口径

### 单 case

```text
success@1 = successful attempts / attempts
```

### 多次尝试

对同一个 case 的 k 次独立尝试：

- `pass@k`：至少一次成功。
- `pass^k`：全部成功。
- `mean_score`：连续分数的平均值。
- `variance`：判断 agent 是否高度不稳定。

在报告中明确区分 observed `pass@k` 与基于独立性假设的概率估计，不把两者混写。

### Suite 聚合

- 默认报告 case-level macro average。
- 按 tags、难度、工具类型、失败类型分别切片。
- 同时报告 n、均值、标准误或 bootstrap confidence interval。
- 低样本类别显示样本数，不伪装成稳定结论。
- 成本、延迟和 token 单独报告，必要时增加 `success / USD` 辅助指标。

## 报告模型

建议包含：

- suite identity 和版本。
- agent/model/tool/config fingerprint。
- case 总数、attempt 总数和跳过数。
- success@1、pass@k、pass^k。
- outcome、trajectory、safety 分数。
- token、cost、latency、steps、tool calls。
- failure taxonomy 分布。
- case-level result 链接。
- raw artifact retention policy。

不设默认加权总分。若业务确实需要总分，权重必须成为 suite 配置的一部分，并在报告中公开。

## 测试与退出门槛

- grader 可以单独接收 fixture，不必启动真实 Agent。
- trajectory grader 可以从 RunLog replay 得到和 live trace 一致的结果。
- state grader 能发现 collateral damage。
- judge grader 异常不会覆盖程序 grader。
- `pass@k` 和 `pass^k` 有边界 case：k=1、全成功、全失败、部分成功。
- macro average 不受 case 顺序影响。
- 聚合结果可以从单 case result 重新计算。

