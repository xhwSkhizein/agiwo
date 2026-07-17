# P0-01：统一 RunPlan 与 update_plan

状态：planned

## 目标

把现有 milestone 能力整理成唯一的 Run 级计划模型 `RunPlan`。计划由 Agent 自己拥有，通过原子的增量工具 `update_plan(changes=...)` 更新；Scheduler 不再定义或注入计划工具。该任务只处理 Run 内计划，不引入 Objective、Assignment todo 或全局目标修改工具。

## 对应决定

- ADR 0038：Assignment 计划复用 RunPlan。
- ADR 0026：planning policy 约束计划粒度与责任范围。
- ADR 0039：Run 内不提供 Objective update tool。

## 依赖

- 无。可以在当前主分支直接开始。

## 当前源码现状

- `agiwo/agent/introspect/models.py` 定义 `GoalState`、`GoalUpdate` 和持久化的 `active_milestone_id`。
- `agiwo/agent/models/run.py` 使用 `RunLedger.goal`。
- `agiwo/scheduler/runtime_tools.py` 定义 `DeclareMilestonesTool`。
- `agiwo/scheduler/engine.py` 构造并注入该工具。
- `agiwo/agent/run_bootstrap.py` 按 `agent_id` 重放 introspect 状态，会把旧 Run 的计划带入新 Run。
- `agiwo/agent/prompt.py` 无条件描述旧计划工具。

## 范围

包含：模型与 fact 改名、Run-local replay、增量更新规则、Agent 内建工具装配、prompt 条件渲染、Console 现有 milestone 投影适配。

不包含：Assignment 完成门禁、Outcome carry_forward、Objective Timeline；它们分别由 P2-05 和 P5-04 实现。

## 实施步骤

1. 在 `agiwo/agent/models/` 收口 `RunPlan`、`Milestone`、状态和 `RunPlanUpdate` 纯数据模型；`active_milestone_id` 改为从唯一 `status=active` 项推导的属性。
2. 新建 `agiwo/agent/plan/`，由它拥有计划规范化、原子变更和 `UpdatePlanTool`。不要把 trajectory review 或 Objective 逻辑放入该包。
3. 把工具改为 `update_plan(changes=[...])`：新 ID 必须带 description；旧 ID 只更新显式字段；遗漏项保持不变；任一 change 非法时整批不提交。
4. 实现确定性 active 规则：最多一个显式 active；切换时旧 active 回到 pending；active 完成/放弃后自动激活第一个 pending；全部解决时没有 active。
5. 保留稳定顺序和 identity：新项按 change 顺序追加，不提供重排或物理删除；同一 Run 中允许 completed/abandoned 重新打开。
6. 将 `GoalMilestonesUpdated` 改为 `RunPlanUpdated`，每次成功更新写入规范化完整快照；模型可见 tool result 只返回 revision 与各状态计数，内部 output 和 RunLog 保存完整计划。
7. 将 replay 查询严格限定为当前 `run_id`。同一 Run pause/resume 可以恢复计划；新 Run 从空计划开始。
8. Agent runtime 始终自行装配 `update_plan` 系统工具，使 direct Run、Scheduler root 和 child Run 的 schema 一致；从 Scheduler 删除计划工具定义、构造和派生逻辑。
9. 根据实际工具装配条件渲染 system prompt。planning policy 明确计划只覆盖当前责任、采用可验证阶段结果，不包含后继 Agent、Verifier 或 User 的工作。
10. 删除旧名称、旧 fact kind、旧 tool name 和兼容别名；同步 SDK 导出、serialization、trace 与 Console milestone board。

## 主要改动位置

- `agiwo/agent/models/`
- `agiwo/agent/plan/`（新建）
- `agiwo/agent/models/run.py`
- `agiwo/agent/models/log.py`
- `agiwo/agent/runtime/state_writer.py`
- `agiwo/agent/run_bootstrap.py`
- `agiwo/agent/prompt.py`
- `agiwo/agent/agent.py`
- `agiwo/scheduler/runtime_tools.py`
- `agiwo/scheduler/engine.py`
- `console/server/services/runtime/runtime_observability.py`
- `console/server/models/session.py`

## 测试计划

- 把现有 `test_introspect_goal.py` 改为 RunPlan/update_plan 测试，并覆盖原子失败、遗漏保持、重新打开、顺序和 active 规范化。
- 增加同 Session 同 agent 的两个 Run，证明第二个 Run 不继承第一个 RunPlan。
- 增加同 run_id replay 测试，证明最后一个 `RunPlanUpdated` 完整恢复。
- 覆盖 direct Agent、Scheduler root、child Run 都能看到相同 `update_plan` schema。
- 覆盖模型可见成功结果不包含计划描述，而 RunLog 快照包含完整内容。
- 更新 Console milestone/review 投影测试，确保只消费 `RunPlanUpdated`。

## 完成标准

- `rg "GoalState|GoalUpdate|GoalMilestonesUpdated|declare_milestones|ledger\.goal" agiwo tests console` 无结果。
- Scheduler runtime tools 中没有计划工具。
- 非空且未解决的规范化 RunPlan 恰有一个 active。
- 新 Run 的门禁读取不到旧 Run 的计划；历史消息仍保持不变。
- 受影响测试和 `uv run python scripts/lint.py ci` 通过。

## 风险与回退

这是破坏性改名，旧 RunLog 开发数据不会兼容。合并前清理本地开发数据库/`.agiwo` 状态（见本阶段 README 入口条件）；代码只保留当前契约，不增加双读、别名或旧 kind 探测逻辑。若测试失败，回退整个提交并重建测试数据库。
