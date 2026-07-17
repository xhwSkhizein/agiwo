# P0：Agent 运行时前置改造

本阶段先修正现有 Agent 运行时中与 Objective 方案直接冲突的语义。它不创建 Objective，也不改变 Console 普通用户入口；目标是让后续模块能够依赖清楚、稳定的 Run 级契约。

## 入口条件

- 当前主分支包含提交 `6222212`，跨 Run 可以恢复最近一次 `MessagesRebuilt`。
- 现有 Agent、Scheduler 和 Console 测试可作为回归基线。
- `CONTEXT.md` 和 ADR 0038、0042、0043 的术语已经冻结。
- **开始本阶段前清理本地开发持久化数据。** P0 会破坏性重命名 RunLog fact、工具与配置字段；不提供 migration，也不在代码中增加旧数据兼容或探测逻辑。合并/联调前删除本地 `.agiwo` 状态目录、Console 开发用 SQLite/`*.db`，以及任务文档点名的测试库路径；CI 与单元测试继续使用每次重建的临时库。

## 任务顺序

| ID | 任务 | 依赖 | 可否并行 |
| --- | --- | --- | --- |
| P0-01 | RunPlan 与 update_plan | 无 | 可与 P0-03、P0-04 并行 |
| P0-02 | append-only trajectory review | P0-01 | 否，必须消费新 RunPlan 名称和接口 |
| P0-03 | UserMessage 来源与前缀 | 无 | 可并行 |
| P0-04 | 模型调用 attempt 与 Run 上限 | 无 | 可并行 |

## 阶段出口

- 代码中不再出现 `GoalState`、`GoalUpdate`、`GoalMilestonesUpdated`、`declare_milestones` 和 `enable_goal_directed_review`，且不保留兼容别名。
- `update_plan` 与 `review_trajectory` 由 Agent 自行装配；Scheduler runtime tools 只包含调度控制能力。
- RunPlan 只按 `run_id` 重放；新 Run 不能继承上一个 Run 的计划状态。
- trajectory review 不隐藏、删除或改写任何已提交 assistant/tool 消息。
- 每次实际模型请求都有调用序号与 phase，`max_steps_per_run` 能区分普通工作和有限系统收口。
- `UserMessage.is_user_provided` 在序列化、RunLog、StepView 和 Console DTO 中不丢失。
- direct Agent、nested Agent 和 Scheduler persistent root 的现有行为回归通过。

