# Agiwo SDK Code Review — 确认方案

---

## P0-2/3/4: Scheduler 三层代理链 → 方案 B（保留 Engine 但压平代理）

**方案**:
1. **删除 Engine 中 9 个 dead pass-through private methods** (`_propagate_signals`, `_enforce_timeouts`, `_start_pending`, `_start_queued_roots`, `_process_pending_events`, `_wake_waiting`, `_try_urgent_wake`, `_recursive_cancel`, `_recursive_shutdown`)，调用方直接调用 `self._tick_ops.*` / `self._tree_ops.*`
2. **内联 `control_ops` 逻辑到 Engine** — Engine 的 `SchedulerControl` 方法不再纯转发 `control_ops`，而是直接持有 control 逻辑。删除 `SchedulerControlOps` 类
3. **Scheduler facade 只保留 lifecycle** — 构造、`_tick_loop`、`close`、`_adapt_agent`，其余方法仍转发 Engine 但 Engine 此时有实际逻辑

**结果**: Scheduler(thin lifecycle) → Engine(逻辑 + SchedulerControl) → tick_ops/tree_ops/state_ops

---

## P0-6: `llm/helper.py` 零逻辑纯重导出 → 删除

**方案**: 删除 `llm/helper.py`，将所有 import 改为直接从 `llm/event_normalizer` 和 `llm/message_converter` 导入。

---

## P1-1: sentinel kwargs 3 层重复 → 删除 patch_state，全量 save_state

**方案**:
1. 删除 `AgentStateStorage.patch_state` 抽象方法及 memory/sqlite 实现
2. 删除 `store/semantics.py`（只有 `apply_state_patch`）
3. `SchedulerStateOps.mark_*` 改为 `mark_running(state: AgentState, ...)` — 直接修改 state 对象 → `save_state` → `notify`
4. 调用方（tick_ops 等）已持有 state 对象，直接传入

**收益**: 消除 sentinel kwargs 3 层重复，删除 `store/semantics.py`，存储接口简化

---

## P1-2: 两套手写序列化 → 核心 dataclass 添加 `to_dict()`

**方案**:
1. `StepRecord.to_dict()`, `Run.to_dict()`, `StepMetrics.to_dict()`, `RunMetrics.to_dict()` — 基于 `dataclasses.asdict()` + 统一 enum/datetime 转换
2. `agent/serialization.py`（传输层）调用 `to_dict()` + 传输特有裁剪
3. `agent/storage/serialization.py`（存储层）调用 `to_dict()` + 存储特有处理

**收益**: 基础字段→dict 转换只维护一处，两个序列化模块只负责各自额外变换

---

## P1-3: `ExecutionToolCoordinator` 薄封装 → 内联到 AgentExecutor

**方案**: 将 3 个方法内联到 `AgentExecutor`，删除 `execution_tools.py`，`AgentExecutor` 直接持有 `ToolRuntime`

---

## P1-4: `BaseTraceStorage` 混合 ABC + 运行时 → 分离

**方案**:
1. 删除 ring buffer 缓存（`_buffer`, `_append_to_buffer`, `_get_buffered_trace`, `_query_buffer`, `get_recent`）
2. 提取 pub/sub 到独立 `TraceEventBus` 类（`subscribe/unsubscribe/notify`）
3. `BaseTraceStorage` 变为纯持久化接口：`save_trace`, `get_trace`, `query_traces`, `close`
4. Console trace SSE 依赖 `TraceEventBus`，不再依赖存储接口

---

## P2-3: SchedulerRunner loader 闭包 → 直接参数传递

**方案**: `_execute_agent_run(agent, user_input, ...)` 替代 `_execute_agent_run(agent_loader, input_loader, ...)`。动态加载在调用前完成，删除 `constant_agent_loader`/`constant_input_loader` 闭包
