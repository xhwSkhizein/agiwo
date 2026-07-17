# P0-02：改造 append-only trajectory review

状态：planned

## 目标

保留默认启用的轨迹复盘实验，但取消它删除、隐藏或改写历史消息的能力。复盘只在消息尾部留下对齐判断、精简经验和以 `tool_call_id` 标识的有用性评分；这些评分可以供 compaction 参考，却不能成为确定性删除规则。

## 对应决定

- ADR 0043：trajectory review 是 append-only 实验性元数据。
- ADR 0038：review 只读取 RunPlan，不拥有计划。
- ADR 0036：跨 Run 上下文优先维护稳定前缀。

## 依赖

- P0-01 已完成；本任务使用 `RunPlan`、`RunPlanUpdated` 和 `update_plan` 新名称。

## 当前源码现状

- `agiwo/agent/introspect/repair.py` 会生成 step-back repair plan。
- `agiwo/agent/introspect/apply.py` 会改写 tool result、隐藏 review pair 和清理 notice。
- `IntrospectionOutcome` 仍包含 seq boundary、hidden step 和 repair range。
- `ReviewTrajectoryTool` 由 `agiwo/scheduler/runtime_tools.py` 定义。
- 配置名是 `enable_goal_directed_review`，Console 与测试也使用该旧名。

## 范围

包含：复盘模型、工具归属、append-only 提交、有用性评分、配置改名、compaction 可选输入和可观测投影。

不包含：证明该实验能提高任务质量；P6-02 负责收集和评估结果。

## 实施步骤

1. 将配置改为 `enable_trajectory_review: bool = true`，删除旧配置名和兼容读取。
2. 把 `ReviewTrajectoryTool` 移入 `agiwo.agent.introspect`，由 Agent 在配置开启时作为系统工具装配；Scheduler 不再定义或派生它。
3. 定义结构化输出：`aligned`、可选 `experience`、`tool_usefulness[{tool_call_id, score}]`；score 只允许 0、1、2、3。
4. 复盘窗口由 committed facts 确定，但模型输入只包含窗口内的 tool_call_id 与 tool name，不暴露 RunLog sequence、StepView id 或 repair 区间。
5. 对重复 ID、窗口外 ID 和非法分数逐项拒绝；遗漏的工具记为 unknown，不能按 0 分处理。
6. 把完整解析结果写入新的 `IntrospectionOutcomeRecorded` 结构；保留系统内部定位字段用于调试，但不进入模型协议。
7. 删除 trajectory review 触发的 `ContextRepairPlan`、message hiding、tool result rewrite 和 notice cleanup。保留与 review 无关的正式 compaction、context rollback 能力。
8. review tool call、tool result、notice 和纠偏结果按原提交顺序留在历史中。不得通过 `condensed_content` 偷换旧消息内容。
9. compaction 在存在评分时可以把它作为标有 experimental/unverified 的附加信息；不设置固定权重、阈值或程序删除规则。
10. system prompt 只在配置开启且工具已装配时说明 review 规则；关闭 review 不影响 RunPlan、计划门禁或 carry_forward。
11. 更新 RunLog serialization、trace、Console review cycle 和调试视图，显示 tool_call_id、评分、成本与延迟。

## 主要改动位置

- `agiwo/agent/introspect/`
- `agiwo/agent/models/config.py`
- `agiwo/agent/models/log.py`
- `agiwo/agent/run_tool_batch.py`
- `agiwo/agent/prompt.py`
- `agiwo/agent/compaction.py`
- `agiwo/scheduler/runtime_tools.py`
- `agiwo/scheduler/engine.py`
- `console/server/models/agent_config.py`
- `console/server/services/runtime/runtime_observability.py`
- `console/web/src/components/agent-form.tsx`

## 测试计划

- 复盘前后逐项比较已提交 message，证明旧内容、顺序和数量没有被重写或隐藏。
- 覆盖工具评分 0–3、unknown、重复、窗口外和部分非法输入。
- 覆盖 review disabled：无工具、无 notice、无评分，但 update_plan 与 RunPlan 正常。
- 覆盖 compaction 无 review、带 review 两种路径，证明评分只是可选输入。
- 覆盖模型输入不出现 `seq`、step id 或内部区间坐标。
- 更新 Console 和 trace 投影测试。

## 完成标准

- `rg "enable_goal_directed_review|step_back|ContextRepairApplied" agiwo/agent/introspect agiwo/scheduler console` 不再命中旧 review 路径。
- review 不产生任何 `hidden_from_context` 或旧 step 内容更新。
- Scheduler 只保留调度工具。
- 关闭 review 后 RunPlan 相关测试不变。
- Python、Console 后端和受影响前端测试通过。

## 风险与回退

删除 step-back 会改变现有 introspect 测试和 Trace 形态。回退必须以整个任务为单位，不能同时保留 append-only 与 rewrite 两种模式，否则运行历史会出现不可解释的混合语义。

