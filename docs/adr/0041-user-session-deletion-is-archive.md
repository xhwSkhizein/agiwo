# 普通用户删除 Session 实际执行归档

Console 不再让普通用户物理删除 Session。用户界面的删除行为改为明确的归档：Session 从默认列表隐藏，但 ObjectiveLog、RunLog、`sessions/<session_id>/artifacts/` 目录、预算和 checkpoint 保留。若 Session 仍有活动 Objective，系统先按用户暂停规则进入 DRAINING，等待当前 Assignment/Run 保存安全 checkpoint 并转为 `USER_PAUSED`，随后才完成归档。

## Status

accepted（Session 归档语义仍有效；正文中的活动 Objective / DRAINING / Assignment 前提已由 ADR 0048 作废——有活动 root Run 时先可恢复暂停再归档，以源码为准）

## Considered Options

- 继续只删除 `console_session` 行：实现最少，但会留下仍在运行的 Scheduler state，以及失去 Session 归属的 ObjectiveLog、RunLog 和 outbox。
- 普通用户直接级联物理删除全部数据：语义直观，但需要跨 SessionStore、ObjectiveStore、RunLogStorage、AgentStateStorage、Trace 和 Artifact 存储协调停止与删除，第一版难以保证崩溃恢复安全。
- 普通用户归档，物理清除留给独立管理员操作：保留恢复能力，也把复杂、不可逆的数据清除移出正常交互路径。

## Consequences

- Console Session 增加可查询的归档标记和归档时间；默认列表排除归档 Session，归档列表允许用户恢复可见性。
- 当前 `DELETE /sessions/{session_id}` 的删行语义不能继续保留。实现时改为明确的 archive/restore API 和界面文案，避免把可恢复操作伪装成物理删除。
- 归档含有非终态 Objective 的 Session 时，先提交 `DRAINING(reason=user_archive)`（reason 属 Objective 封闭 drain 枚举，与 `user_pause` 共用 checkpoint barrier），禁止新的模型调用、tool call、handoff、Assignment 和 outbox 派发，待稳定 `USER_PAUSED` 后再完成归档。
- 归档只有在 Objective 已稳定进入 USER_PAUSED 后才生效。进程中途崩溃时，ObjectiveLog 和 Session 归档状态必须能够判断并继续未完成的归档流程。
- 恢复 Session 只清除归档标记，不自动恢复 USER_PAUSED Objective。用户必须显式继续任务，系统才恢复原 Assignment、Run、`RunPlan` 和 checkpoint。
- 归档不删除或重写任何 Objective/Run 事实，也不重新打开 `COMPLETED / FAILED` Objective。
- 管理员物理 purge 是单独的不可逆能力，不属于第一版；在其事务、停止和审计语义被单独设计前，不提供隐藏入口或复用归档 API。
