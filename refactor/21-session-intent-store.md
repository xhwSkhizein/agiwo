# Task 21：SessionIntent 存储加固 — 并发保护、追加化、截断

Phase 2 · 独立任务 · 修复 review 问题 #7（读-改-写竞态、无限增长）

## 现状与问题

### 竞态：无保护的读-改-写

`agiwo/agent/intent/base.py:24-40` 的 `append_entry`：

```python
current = await self.get(session_id)      # 读
...
current.entries.append(entry)             # 改
await self.upsert(session_id, current)    # 写（整个 JSON payload 覆盖）
```

两个并发写入方真实存在：

- `MainAgent.accept` → `_append_user_input_intent`（用户输入）；
- `MainAgent._maybe_append_run_report`（run 结束报告，在后台
  completion task 中触发）。

`get` 与 `upsert` 之间存在 await 点，asyncio 交错会导致
**后写者覆盖前写者的 entry**。SQLite 实现
（`agiwo/agent/intent/sqlite.py:104-117`）整 payload UPSERT，同样暴露。

### 无限增长

`SessionIntent.entries` 只增不减；长会话下 payload 膨胀，
每次 append 的整读整写成本线性上升。

## 目标设计

### 第一步（低成本，立即做）：基类加 per-session 锁

在 `SessionIntentStore` 基类为 `append_entry` 包一层锁：

```python
class SessionIntentStore(ABC):
    def __init__(self) -> None:
        self._append_locks: dict[str, asyncio.Lock] = {}

    async def append_entry(self, session_id, entry, *, last_run_plan=None):
        lock = self._append_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            # 现有 get → append → upsert 逻辑
```

覆盖单进程内全部竞态（当前 Console 部署形态即单进程单 loop）。
注意子类 `__init__` 需调用 `super().__init__()`
（`InMemorySessionIntentStore`、`SQLiteSessionIntentStore`，以及
`intent/factory.py` 里可能的 Mongo 实现——以源码为准核对）。

### 第二步（结构性，随后做）：SQLite 追加化

把整-payload 覆写改为逐 entry 追加，天然免疫覆盖竞态且支持截断：

```sql
CREATE TABLE IF NOT EXISTS session_intent_entries (
    session_id TEXT NOT NULL,
    seq        INTEGER NOT NULL,     -- per-session 单调递增
    kind       TEXT NOT NULL,        -- user_input | run_report
    text       TEXT NOT NULL,
    at         INTEGER NOT NULL,
    run_id     TEXT,
    PRIMARY KEY (session_id, seq)
);
CREATE TABLE IF NOT EXISTS session_intent_meta (
    session_id    TEXT PRIMARY KEY,
    last_run_plan TEXT,              -- JSON RunPlan 快照，可空
    updated_at    INTEGER NOT NULL
);
```

- `append_entry` = 单事务内 `INSERT entry` + `UPSERT meta`；
- `get` = 两表读出后组装 `SessionIntent`；
- **fail-closed**：沿用现有 schema 校验模式（`sqlite.py:73-80`），
  检测到旧 `session_intent` 单表结构直接报错要求清库，不做 migration
  （与 ADR 0049 "Data is fail-closed" 一致）；旧表定义文件 `mv` 到
  `trash/`，不保留双写。

### 截断策略

SessionIntent 是"指南针不是引擎"（ADR 0049），读取方是完成门/对齐视图，
不需要无限历史：

- 配置常量 `MAX_INTENT_ENTRIES = 200`（先硬编码在 `intent/models.py`，
  不进 `AgentOptions`——避免过早暴露配置面）；
- `get` 组装时只取最新 N 条（SQL `ORDER BY seq DESC LIMIT N` 后反转）；
- 物理清理：append 时若 `seq` 超出 N 的 2 倍，删除最旧的溢出行
  （懒清理，均摊成本低）；
- **user_input 与 run_report 不区别对待**——都按时序截断；若后续
  语义门需要"完整用户诉求"，届时再引入 pinned 条目概念，本任务不做。

## 任务拆分

1. **先写并发回归测试**（`tests/agent/test_session_intent_concurrency.py`）：
   - 对同一 session 并发 `asyncio.gather` 追加 50 条 entry
     （InMemory 与 SQLite 各一），断言 entries 数量 == 50、无丢失；
   - 该测试在当前实现下应能稳定复现丢失（如有必要在 stub 的
     `get` 中插入 `asyncio.sleep(0)` 放大交错窗口）。
2. **实现基类锁**（第一步），跑测试转绿。
3. **SQLite 追加化**（第二步）：新 schema + `append_entry` 事务化 +
   fail-closed 校验；`tests/agent/test_session_intent_c01/c02.py`
   相应调整（断言行为不变，存储形态变化不应影响既有断言）。
4. **截断**：实现懒清理与读取上限；补测试
   "追加 2N+1 条后 get 只返回 N 条且为最新"。
5. **文档同步**：`intent/sqlite.py` 模块 docstring 更新 schema 描述；
   `AGENTS.md` 若有 intent 存储表述则同步。

## 验收标准

- 并发测试稳定通过（连续 `-x --count=20` 或本地重复运行无 flake）；
- `tests/agent/test_session_intent_*.py` 全绿；
- 旧单表 schema 数据库启动时得到明确报错（fail-closed 测试覆盖）。

## 风险

- 第二步 schema 变更是破坏性的（开发库需清理），与本次 M3 改动的
  fail-closed 策略一致，无额外沟通成本；
- Mongo/其他后端如存在（核对 `intent/factory.py`），第一步的基类锁
  已覆盖正确性，追加化可只做 SQLite，其余后端排期跟进。
