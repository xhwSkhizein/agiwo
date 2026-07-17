# P5-06：实现 Session 归档、恢复与 Fork 语义

状态：planned

## 目标

把普通用户“删除 Session”改为可恢复归档；有活动 Objective 时先安全 pause，再隐藏 Session。恢复只恢复可见性，不自动调用模型。Session fork 创建全新 Session，不复制活动 Objective/checkpoint，并在首个 Objective 中消费一次用户提供的 context summary。

## 对应决定

- ADR 0041：Session 删除实际执行归档。
- ADR 0035：Session fork 和终态 Objective 规则。
- ADR 0029：活动 Objective 归档前使用可恢复 pause。

## 依赖

- P3-05、P5-01、P5-03 已完成。

## 当前源码现状

- `DELETE /sessions/{id}` 只删除 SessionStore 行，可能留下 Scheduler/Run 数据。
- Session model 没有 archived_at。
- fork 保存 source_session_id/fork_context_summary，但当前运行路径没有可靠消费 summary。

## 实施步骤

1. Session 增加 archived_at/archive state，SessionStore memory/SQLite 支持 archive/restore 和 include_archived 查询；不物理删除事实。
2. 替换 DELETE 语义为明确的 archive command，并提供 restore API；前端按钮和确认文案使用“归档”。
3. 终态或无 Objective 的 Session 可直接归档。
4. 有活动 Objective 时调用 ObjectiveService 已支持的 `DRAINING(reason=user_archive)`（枚举与 barrier 由 P3-05 拥有），复用 user pause barrier；只有 Objective 稳定进入 USER_PAUSED 后才写 Session archived_at。本任务不扩展 drain reason 枚举。
5. crash recovery 能识别“archive requested but drain incomplete”和“Objective paused but session not marked archived”，继续同一流程。
6. 默认 Session 列表排除 archived；归档列表可查询和恢复可见性。
7. restore 只清除 archived_at，不自动 resume USER_PAUSED Objective；用户显式继续后才恢复模型执行。
8. fork 创建新 Session/persistent root identity，不复制源 ObjectiveBudget、Assignment、checkpoint、Outcome 或 RunLog；源活动 Objective 不受影响。
9. fork_context_summary 在新 Session 首个 Objective root Run 作为 false `<system-notice>` 使用一次，保存 provenance；后续 Objective 不重复注入。
10. 需要旧 Artifact 时通过 summary/新输入显式引用，不暗中复制整个 Objective。
11. 管理员 physical purge 不在第一版，不保留隐藏 delete endpoint。

## 主要改动位置

- `console/server/models/session.py`
- `console/server/services/session_store/base.py`
- memory/SQLite session stores
- `console/server/services/runtime/session_service.py`
- `console/server/routers/sessions.py`
- Objective archive command adapter
- Session list/detail frontend components

## 测试计划

- 终态 Session 直接 archive/restore。
- 活动 root+child 先 DRAINING/checkpoint，再 archived。
- barrier 未齐时列表/状态不假装完成归档。
- restore 不产生 RunResumed/LLM call。
- crash 在 drain/pause/archive 三个边界后可恢复。
- fork 不复制 Objective/Run，summary 只使用一次且来源 false。
- 前端默认/归档列表和文案。

## 完成标准

- 普通用户路径没有物理 Session delete。
- 归档不丢 ObjectiveLog、RunLog、`sessions/<session_id>/artifacts/`、Budget 或 checkpoint。
- 恢复可见性与恢复执行是两个明确动作。
- fork 不造成两个 Session 恢复同一 Run。
- Console backend tests 与前端 lint/test/build 通过。

## 风险与回退

不能先隐藏 Session、再异步尝试 pause；进程若在中间崩溃，用户会失去仍在运行任务的入口。正确顺序是 drain -> USER_PAUSED -> archive fact。

