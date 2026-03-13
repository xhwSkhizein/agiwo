# Code Review — 7 个问题逐一确认与优化建议

> 基于 2026-03-12 源码逐一核实。每个问题给出：确认状态、代码证据、影响范围、建议方案。

---

## 问题 1：child agent 继承时共享父 agent 的 hooks 对象（反向污染）

### 确认状态：**确认存在，高危**

### 代码证据

1. `Agent.__init__` 将 hooks 存为实例属性（line 118）：
   ```python
   self.hooks = hooks or AgentHooks()
   ```

2. `Agent.derive_child()` 直接透传同一个 hooks 引用（line 193–201）：
   ```python
   return Agent(
       ...
       hooks=self.hooks,  # 同一个对象引用
   )
   ```

3. `SchedulerRunner._wrap_on_step_hook()` 原地改写 `agent.hooks.on_step`（line 121–133）：
   ```python
   agent.hooks.on_step = scheduler_on_step  # 改写了共享对象
   ```

### 影响

- `_wrap_on_step_hook` 对 child 的改写会直接反向污染 parent 和所有 sibling 的 `on_step`。
- 更微妙的是，如果 parent 在 child 之后再次 wrap，闭包中 `original_hook` 已经是被 child wrap 过的版本，形成隐式递归链。
- 如果多个 child 并行运行，`on_step` 指向的闭包不断被覆盖，最终只有最后一个 child 的 `_sync_step_to_state` 生效。

### 建议方案

**在 `derive_child()` 中 shallow-copy hooks**：

```python
import copy

child_hooks = copy.copy(self.hooks)  # shallow copy dataclass
return Agent(
    ...
    hooks=child_hooks,
)
```

`AgentHooks` 是纯 dataclass，字段都是 `Callable | None`，shallow copy 即可隔离赋值而不影响回调本身的行为。`_wrap_on_step_hook` 改写的是 child 自己的 copy，不会反向污染 parent。

如果需要更强的隔离（例如不同 child 各自独立的 memory hook），可以进一步升级为 `copy.deepcopy`，但当前场景 shallow copy 已经足够。

---

## 问题 2：child agent 的 session_id 语义前后不一致

### 确认状态：**确认存在，高危**

### 代码证据

1. **spawn 时**：`SpawnAgentTool.execute()` 将父 agent 的 `context.session_id` 传给 `spawn_child()`（tools.py line 84–87），`SchedulerEngine.spawn_child()` 将其存入 `AgentState.session_id`（engine.py line 633–634）：
   ```python
   state = AgentState(
       id=child_id,
       session_id=session_id,  # = 父 agent 的 session_id
       ...
   )
   ```

2. **首次执行**：`SchedulerRunner.run_pending_agent()` 使用 `session_id=state.id`（runner.py line 187）：
   ```python
   session_id=state.id,  # 不是 state.session_id！
   ```

3. **wake 执行**：`SchedulerRunner.wake_agent()` 使用 `session_id=state.session_id`（runner.py line 207）：
   ```python
   session_id=state.session_id,  # 用回 state.session_id
   ```

### 影响

- child agent 首次运行时 session_id = `state.id`（一个 UUID），所有 step 都存在这个 session 下。
- wake 后 session_id 切换到 `state.session_id`（= 父 session_id），Agent 加载历史 step 时拿不到首次运行的 step（它们在 `state.id` 下），导致 **wake 后上下文丢失**。
- 同时 `state.session_id` 保存的是父 session_id，wake 后的 step 会写入父 session，可能和父 agent 的 step 产生混淆。

### 建议方案

**明确 child agent 使用独立 session**：child agent 的 session_id 统一用 `state.id`，在所有路径上保持一致。

```python
# runner.py — wake_agent 和 wake_agent_for_events 都改为：
session_id=state.id,

# engine.py — spawn_child 中 session_id 字段改名为 parent_session_id
# 或在 AgentState 上新增 own_session_id = state.id 的语义
```

`state.session_id` 当前承载了两个概念（"我属于哪个父会话" + "我自己的会话"），应拆分为：
- `parent_session_id`：用于 event 过滤、guard session 隔离
- 实际执行的 session_id 统一用 `state.id`

---

## 问题 3：Tool 并发安全契约声明了但未执行

### 确认状态：**确认存在，中危**

### 代码证据

1. `BaseTool.is_concurrency_safe()` 是 abstractmethod，定义在 base.py line 192。
2. `SleepAndWaitTool.is_concurrency_safe()` 显式返回 `False`（tools.py line 169）。
3. `ToolExecutor.execute_batch()` 对所有 tool_calls 直接 `asyncio.gather()`（executor.py line 97）：
   ```python
   return await asyncio.gather(*(_run_single(tc) for tc in tool_calls))
   ```
   **没有任何对 `is_concurrency_safe()` 的检查**。

### 影响

- LLM 可能在单次回复中同时调用 `sleep_and_wait` 和其他工具，`sleep_and_wait` 会并发执行，可能造成状态竞争。
- API 使用者看到 `is_concurrency_safe() -> False` 会以为 runtime 会自动保护，实际上不会。这是一个误导性契约。

### 建议方案

在 `execute_batch` 中将 tool calls 分为两组：

```python
async def execute_batch(self, tool_calls, context, abort_signal=None):
    safe, unsafe = [], []
    for tc in tool_calls:
        fn = _get_function_payload(tc)
        name = _get_string(fn, "name")
        tool = self.tools_map.get(name)
        if tool and not tool.is_concurrency_safe():
            unsafe.append(tc)
        else:
            safe.append(tc)

    # 并发安全的工具并行执行
    results = list(await asyncio.gather(*(_run_single(tc) for tc in safe)))
    # 非并发安全的工具串行执行
    for tc in unsafe:
        results.append(await _run_single(tc))
    return results
```

注意保持结果顺序与原始 tool_calls 一致（需要用 index 排序）。

---

## 问题 4：`_sync_step_to_state()` 引用不存在的 `step.tool_name`，异常被吞

### 确认状态：**确认存在，高危**

### 代码证据

1. `_sync_step_to_state()` 在 runner.py line 150 访问 `step.tool_name`：
   ```python
   if step.tool_name:
       step_summary["tool_name"] = step.tool_name
   ```

2. `StepRecord` 的定义（runtime.py line 124–143）只有 `name` 字段（line 139），**没有 `tool_name` 属性**：
   ```python
   @dataclass
   class StepRecord:
       ...
       name: str | None = None  # 这是 tool step 的工具名
       # 没有 tool_name!
   ```

3. 异常被 `except Exception: pass` 吞掉（runner.py line 154–155）：
   ```python
   except Exception:  # noqa: BLE001
       pass
   ```

### 影响

- 每次 `_sync_step_to_state` 遇到 `step.tool_name` 都会抛 `AttributeError`，然后被吞掉。
- `append_recent_step()` 不会被调用，意味着 `AgentState.recent_steps` 和 `last_activity_at` **始终不会更新**。
- 下游影响：
  - `query_spawned_agent` 返回空的 recent_steps
  - health check 基于 `last_activity_at` 判断，永远看不到活动
  - `list_agents` 的 last_activity 信息缺失

### 建议方案

**将 `step.tool_name` 改为 `step.name`**：

```python
if step.name:
    step_summary["tool_name"] = step.name
```

同时将 `except Exception: pass` 改为至少 log warning，避免未来类似问题再被静默吞掉：

```python
except Exception:
    logger.warning("sync_step_to_state_failed", state_id=state_id, exc_info=True)
```

---

## 问题 5：状态变更语义在 engine 和 runner 重复实现

### 确认状态：**确认存在，中危**

### 代码证据

`_update_status_and_notify` 在两处有**完全相同**的实现：

**engine.py line 373–391**：
```python
async def _update_status_and_notify(self, state_id, status, *, result_summary=..., wake_condition=...):
    await self._store.update_status(state_id, status, ...)
    if status in (COMPLETED, FAILED):
        self._coordinator.notify_state_change(state_id)
    elif status == SLEEPING and wake_condition is not ...:
        if wake_condition is not None and wake_condition.type == WakeType.TASK_SUBMITTED:
            self._coordinator.notify_state_change(state_id)
```

**runner.py line 487–505**：完全相同的逻辑。

### 影响

- AGENTS.md 明确声明 "SchedulerEngine 是唯一的编排 owner"，但 runner 自己也持有 notify 逻辑。
- 两处漂移风险高：如果 notify 策略变更（例如增加 PERIODIC 的 notify），需要同时改两处。
- 读代码的人必须判断"这两个 `_update_status_and_notify` 语义是否相同"。

### 建议方案

**Runner 委托 Engine（通过 callback 或接口）执行状态变更**：

方案 A：Runner 构造时注入一个 `StatusUpdater` callable：
```python
StatusUpdater = Callable[[str, AgentStateStatus, ...], Awaitable[None]]

class SchedulerRunner:
    def __init__(self, ..., update_status: StatusUpdater):
        self._update_status = update_status
```

方案 B：Runner 只做 `store.update_status()`，由 Engine 在 tick 中统一 notify。这更符合 "Engine 是唯一 owner" 的边界。

方案 A 更简单直接，推荐先用 A。

---

## 问题 6：Agent 公开入口承担过多隐式装配和副作用

### 确认状态：**确认存在，但影响偏设计层面，中低危**

### 代码证据

**构造时（`__init__`）**：
- 合并默认工具 + ensure bash pair（line 113–116）
- 注入默认 memory hook（line 121–127）
- 可选创建 SkillManager + 添加 skill_tool（line 131–137）
- 创建 RunStepStorage（line 140–142）
- 创建 TraceStorage（line 143–145）
- 创建 DefaultSystemPromptBuilder（line 148–155）
- 创建 SessionStorage（line 159）

**运行时（`DefaultSystemPromptBuilder.initialize()`）**：
- 创建目录 MEMORY/、WORK/（line 186–187）
- 拷贝模板文件 IDENTITY.md / SOUL.md / USER.md（line 189）
- 初始化 SkillManager（line 194）
- 读取多个 markdown 拼 prompt（line 203）

### 影响

这不是"功能多"的问题，而是：
- **理解成本**：要完整理解 `Agent()` 做了什么，需要跨 7 个类/模块追副作用。
- **测试困难**：想测 Agent 的执行逻辑，必须 mock 掉 storage/prompt/skill 等全部副作用。
- **派生困难**：`derive_child()` 必须小心处理哪些该继承、哪些该重建。

### 建议方案

这是一个渐进式重构，不建议一次性推倒。核心方向是**提取 AgentFactory**：

```python
class AgentFactory:
    """负责 Agent 的装配（tools 合并、storage 创建、prompt builder 创建）。"""

    @staticmethod
    def create(
        name, description, model, *,
        tools=None, system_prompt="", options=None, hooks=None,
    ) -> Agent:
        resolved_tools = _merge_tools(tools)
        resolved_hooks = _inject_default_hooks(hooks)
        storage = _create_storage(options)
        prompt_builder = _create_prompt_builder(...)
        return Agent(
            name=name, model=model, tools=resolved_tools,
            hooks=resolved_hooks, storage=storage,
            prompt_builder=prompt_builder, ...
        )
```

Agent 构造函数退化为纯赋值，所有装配逻辑收口到 Factory。当前可以先不动，优先解决问题 1–5。

---

## 问题 7：配置模型存在同一概念多份定义

### 确认状态：**确认存在，低危（认知税）**

### 代码证据

**AgentOptions 三套**（options.py）：

| 类 | 用途 | 行号 |
|---|---|---|
| `AgentOptions` (dataclass) | 运行时配置 | line 49 |
| `AgentOptionsInput` (Pydantic) | API 创建请求 | line 141 |
| `AgentOptionsPatch` (Pydantic) | API 补丁请求 | line 173 |

三者的字段几乎完全相同（名称、类型），只是 Patch 版本所有字段都是 `Optional`。

**ModelConfig 三套**（factory.py）：

| 类 | 用途 | 行号 |
|---|---|---|
| `ModelConfig` (dataclass) | 运行时配置 | line 27 |
| `ModelParamsInput` (Pydantic) | API 创建请求 | line 64 |
| `ModelParamsPatch` (Pydantic) | API 补丁请求 | line 91 |

### 影响

- 新增一个配置字段需要改 3 处（dataclass + Input + Patch），容易漏改。
- 字段名、默认值、验证规则必须手动保持一致。
- 但目前代码中已经通过 `_DEFAULT_MODEL_CONFIG` / `_DEFAULT_AGENT_OPTIONS` 做了默认值同步，减轻了部分风险。

### 建议方案

这是 dataclass vs Pydantic 的经典张力。两个可选方向：

**方案 A：用 Pydantic 统一**
将 `AgentOptions` / `ModelConfig` 本身也改为 Pydantic BaseModel，然后 Input/Patch 从它继承。

**方案 B：代码生成 Patch 类**
保持 dataclass 作为 source of truth，用 metaclass 或工具函数自动生成 `*Input` 和 `*Patch` 类。

**当前建议**：暂不动。这属于纯认知税问题，当前字段稳定后风险可控。如果字段开始频繁变更，再考虑统一。

---

## 优先级排序

| 优先级 | 问题 | 风险等级 | 修复复杂度 |
|--------|------|----------|-----------|
| **P0** | #1 hooks 共享可变对象污染 | 高 — 运行时语义错误 | 低 — 一行 copy |
| **P0** | #4 step.tool_name → step.name | 高 — 状态同步整体失效 | 低 — 一行改名 |
| **P1** | #2 child session_id 不一致 | 高 — wake 后上下文丢失 | 中 — 需明确语义 |
| **P1** | #3 并发安全契约未执行 | 中 — 潜在竞争条件 | 中 — 分组执行 |
| **P2** | #5 状态变更语义重复 | 中 — 漂移风险 | 中 — 提取 callback |
| **P3** | #6 Agent 隐式装配过多 | 低 — 设计层面 | 高 — 渐进重构 |
| **P3** | #7 配置模型多份定义 | 低 — 认知税 | 中 — 可选统一 |
