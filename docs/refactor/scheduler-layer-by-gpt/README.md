# Scheduler Layer Refactor Notes

> 状态：2026-03-23
> 范围：`agiwo/scheduler/**`，并补充 `agiwo/agent` 与 `console/server/channels/agent_executor.py` 的集成边界。
> 目标：在保持外部功能与主要运行语义不变的前提下，给出一版更易维护、可读性更高、扩展成本更低、并且总代码更少的 scheduler 重构方案。

## 文件索引

- `01-current-review.md`
  - 当前实现的 review 结论、关键风险、复杂度热点、兼容基线。
- `02-target-design.md`
  - 我会如何重写整个 scheduler 层：模块布局、状态机、执行链、mermaid 图、代码草图。
- `03-agent-boundary.md`
  - 重构后 `agiwo/scheduler` 与 `agiwo/agent` 的边界，以及我会怎样让 agent 侧更利于 scheduler 集成。
- `04-tradeoffs-and-migration.md`
  - 关键 trade-off、哪些行为应该严格兼容、哪些“当前 accidental behavior”不值得 bug-for-bug 继承，以及迁移步骤。

## 结论先行

1. 当前 `scheduler` 的方向是对的：
   - facade / engine / runner / store / coordinator 已经有明显分层；
   - `IDLE / QUEUED / WAITING` 语义拆分是正确决策；
   - `SchedulerControl` 作为 tool-facing 窄接口也是正确方向。
2. 当前最大的维护成本，不是“功能太多”，而是“一个语义被拆到太多地方”：
   - 调度决策分散在 `engine.py`、`tick_ops.py`、`selectors.py`；
   - 状态变更分散在 `state_ops.py`、`runner.py`、`tree_ops.py`；
   - 同一个恢复路径同时依赖 persisted state、coordinator 内存状态、stream/waiter 通知。
3. 如果要整层重写，我不会继续把文件切得更碎；相反我会把它收敛成几个更强的 owner：
   - `SchedulerService`
   - `SchedulerRuntime`
   - `TickPlanner`
   - `SchedulerExecutor`
   - `SchedulerMessages`
   - `store/`
4. 我会保留公开 API、状态枚举、scheduler tool surface、以及 `AgentStreamItem` 流协议；
   我不会保留一些明显不稳定或带 bug 的 incidental behavior。

## 我明确采用的兼容目标

- 保持公开 API 形状不变：
  - `Scheduler.run/submit/enqueue_input/stream/wait_for/get_state/steer/cancel/shutdown`
- 保持核心状态语义不变：
  - `PENDING / RUNNING / WAITING / IDLE / QUEUED / COMPLETED / FAILED`
- 保持 child agent 调度语义不变：
  - `spawn_agent`
  - `sleep_and_wait`
  - `query_spawned_agent`
  - `cancel_agent`
  - `list_agents`
- 保持 scheduler 仍然是单进程 owner 模型。
- 保持 Console 侧对 scheduler 状态的路由方式基本不变。

## 我不会刻意继承的 accidental behavior

- `QUEUED` 状态下 `steer()` 的提示消息被持久化后又被 tick 清掉。
- `shutdown()` 对 `RUNNING` root 返回成功但不真正发出 shutdown 请求。
- `custom_child_id` 覆盖已有 state。
- `InMemoryAgentStateStorage` 与 SQLite/Mongo 在“对象是否 live mutable”上的行为差异。

这些属于实现瑕疵，不属于值得兼容的产品契约。
