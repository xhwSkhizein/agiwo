# Context Optimization

长时间运行的 agent（特别是 scheduler / Objective 编排场景）会面临上下文膨胀问题。SDK 提供两套语义层机制：**Context Rollback**（空转回退）和 **Trajectory Review**（轨迹回顾与 context repair）。它们与 compaction（token 窗口压缩）独立共存。

计划数据只有一份：**RunPlan**，通过系统工具 `update_plan` 维护。不要再使用已删除的 `declare_milestones` / `GoalState` / `enable_goal_directed_review`。

## Context Rollback

### 问题

在 scheduler 编排场景中，主 agent 被周期性唤醒（PERIODIC）检查子 agent 进展。如果子 agent 尚未完成，主 agent 只是确认“没有新结果”然后再次 sleep。每次空转都产生完整的 wake + assistant steps，占据上下文但不带来信息增量。

### 工作方式

agent 被周期唤醒后，如果判断没有新进展，在调用 `sleep_and_wait` 时声明 `no_progress=True`：

```python
sleep_and_wait(
    wake_type="periodic",
    delay_seconds=60,
    time_unit="minutes",
    no_progress=True,
    explain="子 agent 仍在运行，暂无新结果"
)
```

系统收到 `no_progress` 后，不会物理删除 canonical `RunLog`。scheduler 会追加一条 `RunRolledBack` fact，默认 step replay 会隐藏该范围内的 steps；若运维或调试需要查看原始轨迹，读取 `RunLog` 时可显式打开 `include_rolled_back=True`。

### 配置

```python
AgentOptions(
    enable_context_rollback=True,  # 默认开启
)
```

### 约束

- 仅 scheduler 场景生效
- 仅 `wake_type=periodic` 时 `no_progress` 有效
- agent 不需要感知回退发生过（wake message 不提及 rollback）

---

## Trajectory Review

### 问题

agent 在工作中可能偏离当前责任与计划，产生大量低价值 tool 调用。系统在检查点强制注入一次性 review 通知，要求调用 `review_trajectory`；若偏离，执行 KV-cache-safe 的 step-back（替换 tool result 内容，不删除/重排业务 message）。

### 工作方式

#### 步骤 1：维护 RunPlan

Agent 使用 `update_plan` 声明或更新里程碑式计划项。权威事实是 `RunPlanUpdated`；Console milestone board 与 Objective 时间线都从 RunLog / ObjectiveLog 投影，不解析工具文本。

#### 步骤 2：系统强制 Review

以下条件可触发系统注入 `<system-review>`，要求 agent 调用 `review_trajectory`：

| 触发类型 | 条件 |
|---------|------|
| `STEP_INTERVAL` | 自上次 checkpoint/review 起达到 `review_step_interval` |
| `CONSECUTIVE_ERRORS` | 连续 tool error |
| `MILESTONE_SWITCH` | RunPlan 里程碑状态变化 |

`review_trajectory` 本身不计入间隔计数；成功消费后重置。

#### 步骤 3：Agent 回顾

```python
review_trajectory(
    aligned=False,
    experience="确认 session 生命周期由 SessionManager 管理；后续聚焦 GC 延迟。",
)
```

当 `aligned=false` 时，系统将 checkpoint 之后的低价值 tool result 替换为 `experience`。Agent 的 tool_call 保留；tool result 被精简（KV-cache 安全）。

权威 RunLog facts 包括（名称以源码为准）：`RunPlanUpdated`、introspection trigger/checkpoint/outcome、`ContextRepairApplied` 等。不要把 `<system-review>` 文本当作真相源。

### 配置

```python
AgentOptions(
    enable_trajectory_review=True,  # 默认开启
    review_step_interval=8,
    review_on_error=True,
)
```

评分是可选实验元数据，可供 compaction 参考，**不得**作为固定删除阈值。观测说明见 [docs/objective-observability.md](../objective-observability.md)。

### 约束

- `update_plan` / `review_trajectory` 由 Agent 作为系统工具装配（scheduler 场景同样可用）
- 计划只有 RunPlan 一份；Objective 不维护第二套 todos
- 与 compaction/rollback 独立共存

### 升级兼容性

数据模型变更无 migration。升级前清理本地 `.agiwo` / SQLite；详见 [docs/guides/dev-data-cleanup.md](dev-data-cleanup.md)。

---

## 架构

```text
agiwo/agent/plan/         # RunPlan 规范化与 update_plan 系统工具
agiwo/agent/introspect/   # trajectory review、context repair、replay
```

`run_tool_batch.py` 是 tool batch 的执行 owner，并显式调用 focused introspect / plan apply 函数。RunPlan、introspection trigger、checkpoint、outcome、context repair 必须写 first-class `RunLog` facts，并由 replay/trace/Console 视图消费 facts。
