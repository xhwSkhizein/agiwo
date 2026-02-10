# Agent 核心执行逻辑重构方案

## 一、问题分析

### P1: `SideEffectIO` 混合职责（事件发射 + 存储写入 + 序列号分配）

`SideEffectIO` 本应只负责"副作用 IO"，但实际上混合了三种不同性质的操作：

| 方法 | 事件发射 | 存储写入 | 序列号 |
|------|---------|---------|--------|
| `emit_run_started` | ✅ | ✅ `save_run` | |
| `emit_step_delta` | ✅ | | |
| `commit_step` | ✅ | ✅ `save_step` | |
| `emit_run_completed` | ✅ | ✅ `save_run` | |
| `emit_run_failed` | ✅ | ✅ `save_run` | |
| `allocate_sequence` | | | ✅ 依赖 storage |

**影响**：所有依赖 `SideEffectIO` 的模块（executor, llm_handler）都被迫间接依赖 Storage。

### P2: `AgentExecutor` 核心循环被副作用侵入

`executor.py` 中的执行循环直接通过 `run_io` 调用 storage+event 混合操作：

- **L151**: `await self.run_io.commit_step(step, llm=llm_context)` — LLM 步骤提交
- **L204**: `seq = await self.run_io.allocate_sequence()` — 工具步骤序列号分配
- **L213**: `await self.run_io.commit_step(step)` — 工具步骤提交
- **L262**: `await self.run_io.commit_step(step, llm=llm_context)` — Summary 步骤提交

核心执行循环应该只关心：LLM 调用 → 检查 tool_calls → 执行 Tools → 追加消息 → 循环。
但目前每一步都掺杂着 "提交到 storage" 和 "发射事件" 的副作用。

### P3: `LLMStreamHandler` 依赖 SideEffectIO

`llm_handler.py` 的签名直接接收 `run_io: SideEffectIO`：

- **L27**: 参数 `run_io: SideEffectIO`
- **L58**: `seq = await run_io.allocate_sequence()` — 序列号分配依赖 storage
- **L70**: `emit_delta=run_io.emit_step_delta` — 通过回调间接依赖

LLM 流式处理器不应该知道 Storage 的存在。

### P4: `Agent._execute_workflow` 职责过载

`agent.py` L246-322 的 `_execute_workflow` 方法 (~76 行) 混合了：

1. 创建 SideEffectIO
2. 记录用户步骤（含 storage 写入）
3. 创建 Run 对象 + 发射 RUN_STARTED（含 storage 写入）
4. 创建 AgentExecutor
5. 调度 Hooks (before_run, memory_retrieve)
6. 从 storage 读取历史步骤
7. 消息组装
8. 核心循环执行
9. 调度 Hooks (after_run, memory_write)
10. 最终化 Run + 发射 RUN_COMPLETED/FAILED（含 storage 写入）

### P5: 数据模型被 Observability 字段污染

以下 Trace 专用字段出现在核心数据模型中：

- `StepRecord`: `trace_id`, `span_id`, `parent_span_id` (L229-231)
- `StreamEvent`: `trace_id`, `span_id`, `parent_span_id` (L339-343)
- `ExecutionContext`: `trace_id`, `span_id`, `parent_span_id` (L22-24)

这些字段仅被 `TraceCollector` 使用，但污染了核心数据模型。

---

## 二、依赖关系现状

```
Agent._execute_workflow
  ├── SideEffectIO(context, storage)  ← 混合事件+存储
  │     ├── StreamChannel.write()     ← 事件发射
  │     └── RunStepStorage.*()        ← 存储读写
  ├── AgentExecutor(model, tools, run_io, ...)
  │     ├── LLMStreamHandler(model)
  │     │     └── run_io.allocate_sequence()     ← 依赖 storage
  │     │     └── run_io.emit_step_delta()       ← 依赖事件
  │     ├── run_io.commit_step()                 ← 依赖 storage + 事件
  │     └── run_io.allocate_sequence()           ← 依赖 storage
  └── RunStepStorage.get_steps()                 ← 直接读 storage
```

**核心问题**：Storage 的"写"操作通过 SideEffectIO 渗透到了 executor 和 llm_handler 中。

---

## 三、重构方案：Event Pipeline（事件管道）

### 设计原则

> **核心执行只产生事件，所有副作用在事件流下游处理。**

这与已有的 `TraceCollector` 中间件模式完全一致（AGENTS.md 设计决策 #6）。
将 Storage 也变成事件流的中间件，形成统一的 Pipeline。

### 重构后的依赖关系

```
Agent._execute_workflow
  ├── EventEmitter(context)           ← 纯事件发射，不涉及 Storage
  ├── AgentExecutor(model, tools, emitter, ...)
  │     ├── LLMStreamHandler(model)
  │     │     └── state.next_sequence()       ← 内存计数器
  │     │     └── emitter.emit_step_delta()   ← 纯事件
  │     ├── emitter.emit_step_completed()     ← 纯事件
  │     └── state.next_sequence()             ← 内存计数器
  └── RunStepStorage.get_steps()              ← 只在启动时读一次

Stream Pipeline (消费端):
  Channel.read() → StorageSink(persist) → TraceCollector(trace) → User
```

### 具体改动

#### 改动 1: `SideEffectIO` → `EventEmitter`

移除所有 storage 调用，变成纯事件发射器：

```python
class EventEmitter:
    """纯事件发射器，不涉及任何 Storage 操作。"""

    def __init__(self, context: ExecutionContext) -> None:
        self.context = context

    async def emit_run_started(self, data: dict) -> None:
        event = self._make_event(event_type=EventType.RUN_STARTED, data=data)
        await self.context.channel.write(event)

    async def emit_step_delta(self, step_id: str, delta: StepDelta) -> None:
        event = self._make_event(event_type=EventType.STEP_DELTA, step_id=step_id, delta=delta)
        await self.context.channel.write(event)

    async def emit_step_completed(self, step: StepRecord, llm: LLMCallContext | None = None) -> None:
        event = self._make_event(event_type=EventType.STEP_COMPLETED, step_id=step.id, step=step, llm=llm)
        await self.context.channel.write(event)

    async def emit_run_completed(self, data: dict) -> None:
        event = self._make_event(event_type=EventType.RUN_COMPLETED, data=data)
        await self.context.channel.write(event)

    async def emit_run_failed(self, error: Exception) -> None:
        event = self._make_event(event_type=EventType.RUN_FAILED, data={"error": str(error)})
        await self.context.channel.write(event)
```

#### 改动 2: 序列号内化到 `RunState`

不再从 storage 查最大序列号，改为内存计数器：

```python
@dataclass
class RunState:
    ...
    _next_sequence: int = 0

    def next_sequence(self) -> int:
        seq = self._next_sequence
        self._next_sequence += 1
        return seq
```

初始值在 `_execute_workflow` 中根据加载的历史步骤计算：
```python
existing_steps = await self.run_step_storage.get_steps(session_id=..., agent_id=...)
initial_seq = max((s.sequence for s in existing_steps), default=-1) + 1
```

#### 改动 3: 创建 `StorageSink` 中间件

新建 `agent/inner/storage_sink.py`，消费事件流完成持久化：

```python
class StorageSink:
    """Event stream middleware: persists Run/Step from events."""

    def __init__(self, storage: RunStepStorage, run: Run) -> None:
        self.storage = storage
        self.run = run

    async def wrap_stream(self, stream: AsyncIterator[StreamEvent]) -> AsyncIterator[StreamEvent]:
        async for event in stream:
            await self._persist(event)
            yield event

    async def _persist(self, event: StreamEvent) -> None:
        if event.type == EventType.RUN_STARTED:
            await self.storage.save_run(self.run)
        elif event.type == EventType.STEP_COMPLETED and event.step:
            await self.storage.save_step(event.step)
        elif event.type == EventType.RUN_COMPLETED:
            self._finalize_run_from_event(event)
            await self.storage.save_run(self.run)
        elif event.type == EventType.RUN_FAILED:
            self.run.status = RunStatus.FAILED
            await self.storage.save_run(self.run)
```

#### 改动 4: 统一 Stream Pipeline

在 `Agent` 中组装 pipeline：

```python
def _build_stream_pipeline(self, context, run) -> AsyncIterator[StreamEvent]:
    raw_stream = context.channel.read()

    # Layer 1: Storage persistence
    stream = StorageSink(self.run_step_storage, run).wrap_stream(raw_stream)

    # Layer 2: Trace collection (optional)
    if self._should_trace(context):
        collector = TraceCollector(store=self.trace_storage)
        stream = collector.wrap_stream(stream, ...)

    return stream
```

#### 改动 5: AgentExecutor 和 LLMStreamHandler 签名简化

```python
# executor.py
class AgentExecutor:
    def __init__(self, model, tools, emitter: EventEmitter, options, hooks):
        ...

# llm_handler.py
class LLMStreamHandler:
    async def stream_assistant_step(self, state: RunState, emit_delta, abort_signal):
        seq = state.next_sequence()  # 不再依赖 storage
        ...
```

#### 改动 6: 清理数据模型 Trace 字段（可选，可后续单独做）

- `StepRecord`: 移除 `trace_id`, `span_id`, `parent_span_id`
- `ExecutionContext`: 保留 `trace_id`（跨 Agent 传递），移除 `span_id`, `parent_span_id`
- `TraceCollector` 已有 fallback 逻辑可以从 `run_id` + `parent_run_id` 推导父子关系

---

## 四、影响范围

| 文件 | 改动类型 |
|------|---------|
| `agent/inner/side_effect_io.py` | **删除** → 替换为 `event_emitter.py` |
| `agent/inner/event_emitter.py` | **新建** — 纯事件发射 |
| `agent/inner/storage_sink.py` | **新建** — Storage 中间件 |
| `agent/inner/executor.py` | **修改** — 依赖 EventEmitter 替代 SideEffectIO |
| `agent/inner/llm_handler.py` | **修改** — 移除 SideEffectIO 依赖 |
| `agent/inner/run_state.py` | **修改** — 添加序列号计数器 |
| `agent/agent.py` | **修改** — 使用新的 Pipeline 组装逻辑 |
| `agent/schema.py` | **可选修改** — 清理 trace 字段 |
| `agent/execution_context.py` | **可选修改** — 清理 span 字段 |

### 不受影响的部分

- `observability/collector.py` — TraceCollector 保持不变，只是在 pipeline 中位置更明确
- `tool/executor.py` — 不涉及事件/存储，无需改动
- `agent/hooks.py` — Hook 定义不变
- LLM Provider 层 — 不涉及

---

## 五、待讨论项

1. **改动 6（清理 Trace 字段）是否在本次重构中一起做？** 还是作为后续独立任务？
2. **`_execute_workflow` 是否需要进一步拆分？** 比如提取 Run Lifecycle 管理为独立类？
3. **其他你觉得需要讨论的点？**

---

## 六、讨论记录

### Round 1 (2026-02-10)

**用户反馈**：
1. `_next_sequence` 需要线程安全 — 因为一个 Agent 可以派生多个子 Agent 同时运行，子 Agent 共享同一 session 的序列号
2. 改动 6（清理 Trace 字段）一起做，前提是不影响现有 Trace 逻辑
3. 没有其他问题，确认开始重构

**最终决策**：
- 序列号计数器使用 `asyncio.Lock` 保证并发安全
- 改动 1-6 全部执行，包括 Trace 字段清理
- TraceCollector 的 fallback 逻辑已足够，清理字段不影响功能

### 实施完成 (2026-02-10)

**已完成改动**：
1. `SideEffectIO` → `EventEmitter` (纯事件发射)
2. `RunState` 添加线程安全序列号计数器 (`asyncio.Lock`)
3. `LLMStreamHandler` 移除 SideEffectIO 依赖，使用 `emit_delta` 回调 + `state.next_sequence()`
4. `AgentExecutor` 使用 EventEmitter 替代 SideEffectIO
5. 新建 `StorageSink` 中间件 (含嵌套 Agent 多 Run 追踪)
6. `Agent` 组装统一 Pipeline: `Channel → StorageSink → TraceCollector → User`
7. 清理 `StepRecord` 中 `trace_id`/`span_id`/`parent_span_id` 字段
8. 清理 `ExecutionContext` 中 `span_id`/`parent_span_id` 字段 (保留 `trace_id`)
9. 更新 `TraceCollector._resolve_parent_span` 移除对 `step.parent_span_id` 的引用
10. 更新 SQLite `_deserialize_step` 忽略已移除字段
11. `SideEffectIO` 移至 `trash/`

**测试结果**: 38/38 通过
