# P1：Objective 领域内核与存储

状态：**done**（2026-07-18）

本阶段创建 `agiwo.objective` 深模块，但暂不把普通用户请求切换到它。重点是先建立可信的领域事实、状态投影和事务存储，使后续执行控制有唯一依据。

## 入口条件

- P0 阶段出口全部通过。
- `UserMessage` 已能区分真实用户输入和系统输入。
- Run 级计划、复盘和模型调用事实已经稳定。

## 任务顺序

| ID | 任务 | 依赖 | 状态 |
| --- | --- | --- | --- |
| P1-01 | Objective 领域模型 | P0-03 | done |
| P1-02 | ObjectiveLog 与投影 | P1-01 | done |
| P1-03 | ObjectiveStore 与事务 Outbox | P1-02 | done |
| P1-04 | ObjectiveService 深模块边界 | P1-02、P1-03 | done |

## 阶段出口

- [x] `Objective / Assignment / Run / Session` 四层概念在类型和状态机中不混用。
- [x] Objective、Assignment、Budget、ObjectiveUserInput、Contribution、Outcome 的当前状态都能只靠 ObjectiveLog 重建。
- [x] 同一 Objective 最多一个非终态 Assignment；同一 Session 最多一个非终态 Objective。
- [x] Session 级唯一 slot 在 memory/SQLite 中原子保证基数约束，不能依赖先查后写；非终态不可释放 slot。
- [x] 所有 Objective 写命令通过持久化 receipt 绑定 scope、idempotency key、请求 hash 与首次响应。
- [x] ObjectiveLog facts 与 outbox record 可在 memory 和 SQLite 中原子提交；`AssignmentCreated` 必须同事务携带 `DispatchRequested`。
- [x] ObjectiveStore 与 RunLog 使用同一物理配置，但没有共享表或互相读取内部实现。
- [x] `agiwo.objective.__init__` 公开 `ObjectiveService`、稳定 DTO 与 Console 接线所需的 store factory；import-linter 禁止 `agent`/`scheduler` 反向依赖 `objective`。

## 验收证据（2026-07-18）

| 检查 | 结果 |
| --- | --- |
| `uv run pytest tests/objective -q` | 53 passed |
| `console/.venv/bin/python -m pytest console/tests -q` | 233 passed |
| `uv run python scripts/lint.py ci` | All checks passed（含 import-linter + repo_guard） |

下一阶段：`docs/adr-plan/phase-2-assignment-mainline/`（模板、Run identity、outbox dispatcher、主链闭环）。
