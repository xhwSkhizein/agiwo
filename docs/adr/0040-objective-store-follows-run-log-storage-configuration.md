# ObjectiveStore 跟随 RunLog 存储配置（MVP：memory/SQLite）

ObjectiveLog 与 Objective Dispatch Outbox 使用与 RunLog 相同的全局存储配置和物理数据库，但保留独立的 ObjectiveStore 接口及独立表。Console 不新增第二套 Objective 数据库配置；ObjectiveStore 只在自己的事务边界内保证 ObjectiveLog facts 与 outbox 的原子写入。

第一版（MVP）只实现并支持 `memory` 与 `sqlite`。若统一 storage 配置指向其他 backend（包括尚未为 RunLog/Objective 实现的 Mongo 等），构造 ObjectiveStore 时必须 fail-closed，不得 silently 降级为 memory，也不得提交半成品集合实现。

## Status

accepted

## Considered Options

- ObjectiveStore 使用独立数据库配置：每个领域可以单独扩展，但部署需要维护两套连接、路径和备份策略，ObjectiveLog 与 RunLog 容易落在不同物理位置。
- 直接把 ObjectiveLog 合并到 RunLogStorage：可以复用连接，但会把 Objective 事实强行塞进 RunLog 的 `run_id / agent_id` 约束，破坏此前的领域边界。
- 跟随 RunLog 配置但使用独立 ObjectiveStore（本决定）：保留统一部署入口，同时允许 ObjectiveLog/outbox 使用自己的表、事务和投影逻辑。

## Consequences

- Console 的统一 storage wiring 同时构建 RunLogStorage 与 ObjectiveStore。SQLite 使用同一个数据库文件的独立 `objective_log_entries`、`objective_dispatch_outbox`（及相关 slot/receipt 表）与 RunLog 表。
- memory 后端由同一运行时持有可共享的 RunLog 与 ObjectiveStore 实例，进程退出后按既有 memory 语义丢失。
- MVP 不实现 Mongo/其他集合后端的 ObjectiveStore；配置为非 memory/sqlite 时 factory 明确失败并提示清理或改回支持的 backend。
- 「跟随 RunLog」在 MVP 中的含义是：同一 wiring、同一 sqlite 文件或同一 memory 进程；不是「配置枚举里出现的任何 backend 都已可用」。
- ObjectiveStore 的事务只覆盖 ObjectiveLog facts 与 outbox；RunLog、AgentStateStorage 和 Trace 不参加跨存储分布式事务，恢复仍依赖稳定 `objective_id / assignment_id / run_id`。
- ObjectiveService 面向 Console 使用统一构造的 ObjectiveStore；SDK 直接使用 ObjectiveService 时可以显式提供 ObjectiveStore 实例，但不需要暴露第二套部署级数据库选项。
- ObjectiveStore 的 schema、索引和 replay 逻辑由 `agiwo/objective` 维护，不能因为共享物理数据库就直接依赖 RunLogStorage 的内部表或查询实现。
