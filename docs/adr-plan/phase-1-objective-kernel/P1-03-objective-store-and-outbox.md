# P1-03：实现 ObjectiveStore 与事务 Outbox

状态：planned

## 目标

提供 ObjectiveLog、Dispatch Outbox、Session 活动 Objective 占用和 command receipt 的持久化边界。一次 Objective 命令必须在同一事务中提交相关 facts、约束索引、回执和待派发 record，避免重复创建、越过 Session 基数约束或留下没有执行者的领域状态。

## 对应决定

- ADR 0018：ObjectiveLog 独立于 RunLog 和 AgentState。
- ADR 0019：Objective 派发使用 Transactional Outbox。
- ADR 0040：ObjectiveStore 跟随 RunLog 存储配置和物理数据库。

## 依赖

- P1-02 已完成。

## 当前源码现状

- RunLogStorage 已有 memory/SQLite append/query 和 session sequence。
- Console storage wiring 为 RunLog、Trace、AgentState 等构建同一路径配置。
- Scheduler PendingEvent 会被消费删除，不适合作为可靠 Assignment 派发证明。
- 当前没有 Objective 表或事务接口。MVP 不规划 Mongo collection。

## 范围

包含：ObjectiveStore contract、memory/SQLite backend、事务 append、Session slot、command receipt、outbox claim lease、查询和存储 wiring。

不包含：真正调用 Scheduler 的 dispatcher；P2-03 实现。重启后的跨 Run 对账由 P3-06 完成。

## 实施步骤

1. 在 Objective 包内定义窄 `ObjectiveStore` contract：原子命令事务、按 sequence 查询 facts、当前 max sequence、Session slot、command receipt、outbox claim/renew/complete/release。
2. 定义 `objective_session_slots` 约束索引：session_id 唯一、objective_id、acquired_at、objective revision。创建非终态 Objective 时取得，COMPLETED/FAILED 事务释放；索引可由 ObjectiveLog 重建，但不能被普通调用绕过。
3. 定义持久化 command receipt：scope、idempotency_key、canonical_request_hash、status、response payload、created/completed time。create scope 使用 `session:{session_id}:objective:create`；已有 Objective 命令使用 `objective:{objective_id}:{command_kind}`。
4. 同一 scope/key + 相同 hash 返回首次持久化响应；同一 scope/key + 不同 hash 返回类型化 `IdempotencyConflict`。receipt 必须与命令 facts/outbox/slot 在同一事务完成。
5. 定义 DispatchRequested record：dispatch_id、objective_id、assignment_id、预分配 run_id、职责、状态、attempt、created_at、lease owner/expiry 和最后错误。
6. 实现 `commit_command(receipt, facts, slot_mutation, outbox_records)` 原子操作。任一写入失败时全部回滚。
7. memory backend 至少按 session_id 和 objective_id 使用有固定加锁顺序的锁；create 不得只锁新 objective_id。维护与 SQLite 相同的 slot/receipt 唯一索引。
8. SQLite 在与 RunLog 相同 db_path 中创建独立 `objective_log_entries`、`objective_session_slots`、`objective_command_receipts` 与 `objective_dispatch_outbox` 表、索引和唯一约束；不读取 RunLog 内部表。
9. 使用共享 SQLite runtime/connection 基础设施，避免同一文件多套连接管理；事务只覆盖 ObjectiveStore 自己的表。
10. ObjectiveStore factory 与 RunLog 共用配置入口，但 MVP **只**实现 memory/SQLite。配置为其他 backend 时立即 fail-closed（明确错误），禁止 silent fallback 到 memory，禁止提交半成品 Mongo/集合实现。不得仅因配置模型曾列出但 factory 不可用的 backend 发明占位实现。
11. Console storage wiring 从 RunLogStorageConfig 派生 ObjectiveStore，不新增 Objective 数据库环境变量。
12. outbox claim 使用有期限 lease 和原子状态更新；重复 claim、lease 过期、owner 不匹配均有确定性结果。
13. slot、receipt、outbox 都是约束/交付记录，不参与 ObjectiveStatus 投影；ObjectiveLog 才是领域状态来源。
14. schema 直接创建当前版本，不提供 migration；测试数据库每次重建。

## 主要改动位置

- `agiwo/objective/store/`（新建）
- `agiwo/objective/outbox.py`
- `agiwo/utils/storage_support/` 或现有共享 SQLite runtime
- `console/server/services/storage_wiring.py`
- `console/server/dependencies.py`
- `tests/objective/test_store.py`
- `tests/objective/test_store_sqlite.py`

## 测试计划

- facts 与 outbox 原子成功、任一失败全回滚。
- 两个不同 objective_id 并发 create 同一 session，memory/SQLite 都只能一个取得 slot。
- terminal 事务释放 slot 后可以创建下一 Objective；非终态 pause/wait 不释放。
- command receipt 同 key/同 hash 返回原 response，同 key/不同 hash 返回 conflict；create 在 objective_id 生成前也能查重。
- sequence 原子递增、稳定 fact id 重复 append 防护和并发提交冲突。
- claim/lease renew/expiry/reclaim/complete 的确定性测试。
- SQLite 关闭重开后日志和 pending outbox 完整恢复。
- RunLog 与 ObjectiveStore 使用同一 db 文件但独立 schema，互不查询内部表。
- memory 与 SQLite contract parity。
- 配置为非 memory/sqlite 的 backend 时 ObjectiveStore factory fail-closed，且不 silent 降级为 memory。

## 完成标准

- 不存在“AssignmentCreated 已提交但没有 DispatchRequested”的可见事务结果。
- 不存在同一 Session 的两个活动 slot，create 不依赖“先查后写”。
- 所有 Objective 写命令都有持久化 receipt，响应重放不依赖进程内 cache。
- ObjectiveStore 不导入 RunLogStorage 实现类或 Scheduler store。
- Console 没有第二套 Objective storage 配置。
- PendingEvent 没有被复用为 outbox。
- objective store 测试、Console storage 测试与 lint 通过。

## 风险与回退

共享 SQLite 文件并不表示共享事务 owner。必须通过已有共享连接基础设施协调连接生命周期。若事务隔离测试不稳定，暂停 P2 派发接入并修复 store，不能以“先写日志、后补 outbox”降级。
