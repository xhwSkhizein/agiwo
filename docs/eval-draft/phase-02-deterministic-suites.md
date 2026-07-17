# Phase 02：本地确定性 Benchmark Suites

状态：Planned

## 目标

建立第一条无需外网、无需 GUI、无需大型容器的可重复价值路径。它应暴露 Agent 在工具选择、参数构造、状态更新、信息不足和副作用控制上的能力。

这不是 BFCL、tau-bench 或 AppWorld 的完整复刻，而是借鉴它们的判分思想，为 Agiwo 提供一个稳定的内部回归套件。

## 文件计划

### 创建

- `agiwo/evaluation/suites/__init__.py`
- `agiwo/evaluation/suites/function_calling.py`
- `agiwo/evaluation/suites/stateful_tools.py`
- `agiwo/evaluation/suites/fixtures.py`
- `tests/evaluation/suites/test_function_calling.py`
- `tests/evaluation/suites/test_stateful_tools.py`
- `tests/evaluation/test_local_suite_smoke.py`
- `scripts/run_evaluation.py`

## Suite A：Function Calling

第一版至少包含：

- 单工具单次调用。
- 多工具选择。
- 必填参数缺失。
- 类型和枚举参数错误。
- 无需工具时不应强行调用工具。
- 两个并行安全工具的并行调用。
- 多轮对话中的工具调用。
- 工具报错后的重试或改道。

判分顺序：先检查结构化工具调用，再检查最终答案。工具调用错误时，即使最终文本碰巧正确，也应保留过程失败标签。

## Suite B：Stateful Tools

构造一个进程内虚拟应用，例如订单、日历或文件工作区。每个 case 提供：

- 初始状态。
- 用户目标。
- 允许的工具。
- 业务政策。
- 目标状态。
- 禁止修改的字段。
- 可选的中间里程碑。

判分使用：

```text
goal state satisfied
AND policy respected
AND forbidden state unchanged
```

这一步要明确区分“任务成功”和“无副作用成功”。AppWorld 的状态测试与 collateral damage 检查是此处的直接参考。

## 环境实现规则

- 所有虚拟工具都实现真实的 `BaseTool` 接口。
- 工具返回值包含足以让 Agent 继续工作的结构化信息。
- environment snapshot 使用稳定 JSON 表示，便于 diff 和测试。
- 每个 case 都从全新初始状态开始。
- 不能用测试专用绕过路径代替 Agent 真实调用。
- 不依赖当前系统时间、随机 UUID 或未固定的文件路径。

## 失败分类

第一版先使用有限的分类集合：

- `wrong_tool`
- `invalid_arguments`
- `missing_information_not_requested`
- `policy_violation`
- `tool_error_not_recovered`
- `goal_state_missing`
- `collateral_damage`
- `timeout_or_budget`
- `final_response_error`

失败分类属于 grader 输出，不应反写成 Agent runtime 的终止原因。

## CLI 草稿

```bash
uv run python scripts/run_evaluation.py \
  --suite local-stateful-tools \
  --suite-version 0.1 \
  --attempts 3 \
  --output ./data/evaluations
```

CLI 第一版只负责运行和输出路径；复杂筛选、Console 展示留给后续 phase。

## 测试与退出门槛

- 在 fake model 下能稳定得到预期工具调用。
- 至少有一个成功 case、一个错误恢复 case、一个越权 case、一个副作用 case。
- 同一 seed 重跑产生相同环境 snapshot 和 grade。
- 不使用真实网络，不依赖 provider API key。
- CLI smoke 能生成 batch、attempt、grade 和 aggregate report。
- `uv run pytest tests/evaluation -q`

