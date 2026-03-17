# Agiwo SDK Code Review

> 核心目标：提升代码可维护性。按优先级排列，P0 影响最大。

---

## P0-1: `agent/runtime.py` God File — 393 行 ~15+ 无关类型挤在一个文件

**文件**: `agiwo/agent/runtime.py`

**问题**: 该文件是整个 SDK 认知负荷最高的单点。包含：
- 枚举: `MessageRole`, `RunStatus`, `TerminationReason`
- 度量 dataclass: `StepMetrics`, `RunMetrics`
- LLM 调用上下文: `LLMCallContext`
- Step 领域: `StepDelta`, `StepRecord` (含工厂方法)
- 流式协议: `AgentStreamItemBase` + 7 种 `AgentStreamItem` 事件类型
- Run 生命周期: `Run`, `RunOutput`, `AgentContext`
- 辅助函数: `step_to_message`, `steps_to_messages`

这些类型属于不同关注点（step 构建、run 生命周期、流式协议、度量），但全部堆砌在一个文件中。阅读时需要反复跳转，新增类型时也不知道应该放在哪里。

**影响**: 高认知负荷，新开发者难以导航，类型之间的边界模糊容易引发 drift。

---

## P0-2: SchedulerEngine 9 个死亡代理方法

**文件**: `agiwo/scheduler/engine.py:382-407`

**问题**: 9 个 private 方法是对 `tick_ops` / `tree_ops` 的纯 1:1 转发，无任何附加逻辑：

```python
async def _propagate_signals(self) -> None:
    await self._tick_ops.propagate_signals()
# ... 以及 _enforce_timeouts, _start_pending, _start_queued_roots,
# _process_pending_events, _wake_waiting, _try_urgent_wake,
# _recursive_cancel, _recursive_shutdown
```

更矛盾的是，同文件 `tick()` 方法 (line 370-380) 已经**直接**调用 `self._tick_ops.wake_waiting()`，说明这些 private wrapper 甚至不一致。

**影响**: 增加代码行数和理解负担，无任何封装收益。

---

## P0-3: SchedulerEngine 作为 SchedulerControl 接口的纯透传层

**文件**: `agiwo/scheduler/engine.py:409-465`

**问题**: Engine 实现 `SchedulerControl` 协议，但 `spawn_child`, `sleep_current_agent`, `get_child_state`, `list_child_states`, `inspect_child_processes`, `age_seconds` 全部是对 `self._control_ops` 的纯转发。唯一包含逻辑的是 `cancel_child`（验证权限 + 状态检查）。

Engine 变成了 control_ops 的代理对象，tool 代码调用 Engine，Engine 原封不动转发给 control_ops，增加了一层无意义的间接。

**影响**: DRY 违反，tools → engine → control_ops 三层调用栈增加追踪难度。

---

## P0-4: Scheduler facade 几乎零附加值

**文件**: `agiwo/scheduler/scheduler.py`

**问题**: `Scheduler` 类几乎所有公开方法（`submit`, `run`, `enqueue_input`, `stream`, `wait_for`, `steer`, `cancel`, `shutdown`, `get_registered_agent`）都是 1:1 转发给 `SchedulerEngine`。唯一附加逻辑是：
- `_adapt_agent()` 包装 Agent 为 `AgentSchedulerPort`
- `_tick_loop()` 后台调度循环

加上 P0-2 和 P0-3，形成了 **Scheduler → Engine → ops** 三层代理链，其中中间两层基本没有自有逻辑。

**影响**: 3 层间接增加导航成本，方法签名在 3 个文件中重复。

---

## P0-5: `SchedulerRuntimeTool` 平行工具类型体系

**文件**: `agiwo/scheduler/runtime_tools.py:26-131`

**问题**: `SchedulerRuntimeTool` ABC 重新定义了 `get_name`, `get_description`, `get_parameters`, `is_concurrency_safe`, `get_definition`, `get_short_description` — 与 `BaseTool` 契约完全相同但**不继承**。同时内部定义了 `_success`, `_failed`, `_denied` 模板方法构造 `RuntimeToolOutcome`。

这是一个与核心 `BaseTool` / `AgentRuntimeTool` 完全平行的类型体系，需要独立维护、独立理解。5 个具体 tool 类（SpawnAgentTool, SleepAndWaitTool, QuerySpawnedAgentTool, CancelAgentTool, ListAgentsTool）全部继承此 ABC 而非核心契约。

**影响**: 双体系 drift 风险极高，新开发者需要理解两套 tool 契约。

---

## P0-6: `llm/helper.py` 零逻辑纯重导出

**文件**: `agiwo/llm/helper.py`

**问题**: 28 行代码，100% 是从 `event_normalizer` 和 `message_converter` 的 import + re-export，无任何逻辑、聚合或简化。文件头注释 "Backward-compatible helper facade" 暗示这是遗留兼容层。

按 AGENTS.md 规则："不为'可能以后有用'保留遗留兼容层；除非明确要求，否则直接删除旧路径。"

**影响**: 多一个文件 = 多一层间接 + 多一份 `__all__` 需要同步。

---

## P1-1: sentinel `object = ...` kwargs 在 3 层间完全复制

**文件**:
- `agiwo/scheduler/store/base.py:27-39` (`AgentStateStorage.patch_state`)
- `agiwo/scheduler/state_ops.py` (`SchedulerStateOps.patch_state` + `mark_running`)
- `agiwo/scheduler/store/semantics.py:9-39` (`apply_state_patch`)

**问题**: 相同的 8 个 sentinel kwargs (`status`, `task`, `pending_input`, `wake_condition`, `result_summary`, `explain`, `signal_propagated`, `wake_count`) 在 3 个函数签名中完全重复。每次新增或修改 AgentState 字段，必须同步改 3 处签名 + 3 处逻辑。

**影响**: DRY 严重违反，字段变更时极易遗漏某一层，是 drift 高发区。

---

## P1-2: 两套手写序列化模块

**文件**:
- `agiwo/agent/serialization.py` (212 行 — API/SSE 序列化)
- `agiwo/agent/storage/serialization.py` (133 行 — 存储序列化)
- `agiwo/observability/serialization.py` (可能还有第三套)

**问题**: 对同一组领域类型 (`StepRecord`, `Run`, `StepMetrics`, `RunMetrics` 等) 存在 2+ 套手写 dict 构造函数。每次模型字段变更，必须在多处手动同步。

这些类型大部分是 dataclass，可以用 `dataclasses.asdict()` + 统一转换层替代大量手写代码。

**影响**: DRY 违反，字段新增/删除时多处同步成本高，是 drift 高发区。

---

## P1-3: `ExecutionToolCoordinator` 薄封装

**文件**: `agiwo/agent/inner/execution_tools.py` (105 行)

**问题**: 这个 "coordinator" 只有 3 个公开方法 (`resume_pending_tool_calls`, `execute_tool_calls`, `_commit_batch`)，本质上是将 `ToolRuntime` 和 `RunRecorder` 的调用粘合在一起。每个方法 ~20 行，逻辑可以直接内联到 `AgentExecutor` 中。

独立为类增加了一个需要注入的依赖和理解的抽象层，但封装收益极低。

**影响**: 多一个类 = 多一次间接 + 构造函数注入成本，收益不足以抵消。

---

## P1-4: `BaseTraceStorage` ABC 与具体运行时行为混合

**文件**: `agiwo/observability/base.py`

**问题**: `BaseTraceStorage` 既定义抽象接口 (`save_trace`, `get_trace`, `query_traces`, `close`)，又提供具体的 ring buffer + pub/sub 实现 (`_buffer`, `_subscribers`, `subscribe`, `unsubscribe`, `_notify_subscribers`, `get_recent`, `_query_buffer`)。

所有子类都被迫继承 buffer/subscriber 机制，即使某些实现（如纯 MongoDB）可能不需要。接口和运行时策略耦合。

**影响**: 违反 ISP(接口隔离)，子类被迫接受不需要的运行时状态。

---

## P2-1: `agent/scheduler_port.py` 适配器开销

**文件**: `agiwo/agent/scheduler_port.py` (132 行)

**问题**: `AgentSchedulerPort` 包装 Agent 的 ~8 个方法，其中大部分是纯 1:1 转发 (`start`, `close`, `install_runtime_tools`, `set_termination_summary_enabled`)。还需要 `unwrap_agent()` 来反向解包。适配器存在是为了 scheduler→agent 依赖反转，但大部分方法完全没有适配逻辑。

**影响**: 增加代码量和追踪成本，适配层维护负担。

---

## P2-2: Span/Trace duration 计算重复

**文件**: `agiwo/observability/trace.py`

**问题**: `Span.complete()` (line 112-126) 和 `Trace.complete()` (line 225-237) 包含完全相同的 duration 计算逻辑：
```python
self.end_time = datetime.now(timezone.utc)
if self.start_time and self.end_time:
    delta = self.end_time - self.start_time
    self.duration_ms = delta.total_seconds() * 1000
```

此外 `Trace.add_span()` 使用字符串 key (`tokens.total`, `tokens.input`) 访问 metrics dict，无类型保障。

**影响**: 小规模 DRY 违反 + stringly-typed metrics 有 drift 风险。

---

## P2-3: SchedulerRunner loader 闭包模式

**文件**: `agiwo/scheduler/runner.py`

**问题**: 使用 `Callable[[], Awaitable[...]]` loader 来传递 agent 和 input 给执行方法。`constant_agent_loader` / `constant_input_loader` 创建闭包仅为返回静态值。这使得调用链比简单的参数传递更难追踪。

**影响**: 增加认知负荷，闭包模式对于静态值来说过度设计。

---

## P2-4: `agent/input_codec.py` 混合关注点

**文件**: `agiwo/agent/input_codec.py` (148 行)

**问题**: 包含 4 种不同职责：
1. 输入规范化 (`normalize_to_message`, `extract_text`)
2. 序列化/反序列化 (`serialize_user_input`, `deserialize_user_input`)
3. 内容类型转换 (`to_message_content`)
4. 本地资源渲染 (`render_local_resources`)

**影响**: 文件职责不单一，但影响有限（148 行仍可接受）。
