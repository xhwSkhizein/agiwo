# P1：Objective 领域内核与存储

本阶段创建 `agiwo.objective` 深模块，但暂不把普通用户请求切换到它。重点是先建立可信的领域事实、状态投影和事务存储，使后续执行控制有唯一依据。

## 入口条件

- P0 阶段出口全部通过。
- `UserMessage` 已能区分真实用户输入和系统输入。
- Run 级计划、复盘和模型调用事实已经稳定。

## 任务顺序

| ID | 任务 | 依赖 |
| --- | --- | --- |
| P1-01 | Objective 领域模型 | P0-03 |
| P1-02 | ObjectiveLog 与投影 | P1-01 |
| P1-03 | ObjectiveStore 与事务 Outbox | P1-02 |
| P1-04 | ObjectiveService 深模块边界 | P1-02、P1-03 |

## 阶段出口

- `Objective / Assignment / Run / Session` 四层概念在类型和状态机中不混用。
- Objective、Assignment、Budget、ObjectiveUserInput、Contribution、Outcome 的当前状态都能只靠 ObjectiveLog 重建。
- 同一 Objective 最多一个非终态 Assignment；同一 Session 最多一个非终态 Objective。
- Session 级唯一 slot 在 memory/SQLite 中原子保证基数约束，不能依赖先查后写。
- 所有 Objective 写命令通过持久化 receipt 绑定 scope、idempotency key、请求 hash 与首次响应。
- ObjectiveLog facts 与 outbox record 可在 memory 和 SQLite 中原子提交。
- ObjectiveStore 与 RunLog 使用同一物理配置，但没有共享表或互相读取内部实现。
- `agiwo.objective.__init__` 只公开 ObjectiveService 和稳定 DTO；import-linter 固化 `objective -> scheduler -> agent`。
