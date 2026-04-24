# Retrospect Batch Refactor

> 将散落在 `run_loop.py` 中的 retrospect 逻辑收口为 `RetrospectBatch` 封装类，
> 拆为独立子包，按触发类型区分 notice 内容，并通过 repo_guard 规则保护约定。

## Problem

当前 retrospect 的实现以"补丁"形式混入 `_execute_tool_calls`：

1. **散落变量**：`retrospect_feedback`、`step_lookup`、`retrospect_enabled` 作为局部变量穿线整个 for 循环。
2. **magic flag**：`result.output.get("_retrospect")` 是 `RetrospectToolResultTool` 和 `run_loop` 之间的暗号协议。
3. **content 隐式篡改**：system notice 注入在 step commit 之前修改 content，读代码时不易预期。
4. **notice 一刀切**：三种触发场景（单次大结果 / 轮次累计 / token 累计）使用相同 notice 文案，无法针对性引导 agent。
5. **上下文变更过于隐性**：`execute_retrospect` 在一个函数调用中同时做了 messages 原地替换、磁盘写入、storage 更新、retrospect 自身痕迹移除、ledger 计数器重置共 5 种 side effect，调用方无法从 call site 预期发生了什么。

## Design

### 1. Package Structure

将 `agiwo/agent/retrospect.py` 拆为子包：

```
agiwo/agent/retrospect/
├── __init__.py          # 公开 API：RetrospectBatch, RetrospectOutcome
├── triggers.py          # 触发检查、RetrospectTrigger 枚举、notice 模板与注入
├── executor.py          # offload / 替换 / 移除自身痕迹 / storage 更新
```

### 2. Trigger Types & Notice Templates

`triggers.py` 引入触发类型枚举，根据类型返回不同的 system notice：

```python
class RetrospectTrigger(Enum):
    NONE = "none"
    LARGE_RESULT = "large_result"
    ROUND_INTERVAL = "round_interval"
    TOKEN_ACCUMULATED = "token_accumulated"
```

| 触发类型 | Notice 侧重 | 引导意图 |
|---------|------------|---------|
| `LARGE_RESULT` | 单条结果过大，强调 concise / replace | "这个结果值不值得留在上下文？如果不值，压缩它" |
| `ROUND_INTERVAL` | 连续多轮 tool call，强调目标聚焦 | "做了 N 轮了，回顾一下方向是否正确，整理发现" |
| `TOKEN_ACCUMULATED` | 累计 token 过多，强调空间紧迫 | "上下文空间紧张，清理低价值内容，保留关键发现" |

`check_retrospect_trigger` 签名变为返回 `RetrospectTrigger`（而非 `bool`）。
`inject_system_notice(content, trigger)` 根据 trigger 选择对应模板。

### 3. RetrospectBatch — 封装类

`__init__.py` 暴露 `RetrospectBatch`，它封装一批 tool call 的 retrospect 生命周期：

```python
class RetrospectBatch:
    """Per-batch retrospect state and operations.

    run_loop 只需要调三个方法：
    - process_result()  → 返回最终 content（可能注入了 notice）
    - register_step()   → 注册已 commit 的 step（供 finalize 时查找）
    - finalize()        → 批次结束后执行 retrospect（如果 agent 调了 retrospect tool）
    """

    def __init__(self, state: RunContext, tools_map: dict[str, BaseTool]):
        self._state = state
        self._enabled = (
            state.config.enable_tool_retrospect
            and "retrospect_tool_result" in tools_map
        )
        self._feedback: str | None = None
        self._step_lookup: dict[str, dict[str, Any]] = {}

    def process_result(self, result: ToolResult) -> str:
        """处理单个 tool result。返回最终 content（可能注入 notice）。"""
        content = result.content or ""
        if not self._enabled:
            return content

        if result.tool_name == "retrospect_tool_result" and result.is_success:
            self._feedback = content
            return content

        update_retrospect_tracking(self._state.ledger, content)
        trigger = check_retrospect_trigger(
            self._state.config, self._state.ledger,
            content, result.tool_name,
        )
        if trigger is not RetrospectTrigger.NONE:
            content = inject_system_notice(content, trigger)
        return content

    def register_step(
        self, tool_call_id: str, step_id: str, sequence: int
    ) -> None:
        """注册已 commit 的 step，供 finalize 时查找。"""
        self._step_lookup[tool_call_id] = {
            "id": step_id, "sequence": sequence,
        }

    async def finalize(self) -> RetrospectOutcome:
        """批次结束后执行 retrospect。

        如果 agent 调了 retrospect_tool_result：
        - 磁盘 offload 和 storage condensed_content 更新在内部完成
        - 返回 RetrospectOutcome(applied=True, messages=新的消息列表)
        - 调用方需显式调用 replace_messages(state, outcome.messages) 完成上下文替换

        如果 agent 未调 retrospect：
        - 返回 RetrospectOutcome(applied=False)
        """
        if not self._enabled or self._feedback is None:
            return RetrospectOutcome()
        return await execute_retrospect(
            feedback=self._feedback,
            state=self._state,
            step_lookup=self._step_lookup,
        )
```

### 4. RetrospectOutcome — 显式化上下文变更

```python
@dataclass
class RetrospectOutcome:
    applied: bool = False
    offloaded_count: int = 0
    messages: list[dict[str, Any]] | None = None
```

`execute_retrospect` 不再原地修改 `RunLedger.messages`。它：
1. 复制当前 messages
2. 在副本上执行 offload 替换、feedback 追加、retrospect 自身痕迹移除
3. 将磁盘写入和 storage condensed_content 更新作为内部 side effect 完成
4. 重置 ledger 的 retrospect 计数器
5. 返回 `RetrospectOutcome(applied=True, messages=新列表)`

调用方（run_loop）显式应用：

```python
outcome = await batch.finalize()
if outcome.applied:
    replace_messages(state, outcome.messages)
```

这样 run_loop 的代码读者能看到 `replace_messages` 调用，知道"上下文在这里被改了"，
和 compaction 的模式统一。

### 5. 消除 Magic Flag

`RetrospectToolResultTool` 不再需要返回 `output={"_retrospect": True}`。
`RetrospectBatch.process_result` 通过 `tool_name == "retrospect_tool_result" and result.is_success`
判断身份，不依赖隐式标记。

### 6. run_loop 变化

`_execute_tool_calls` 简化为：

```python
async def _execute_tool_calls(
    *,
    state: RunContext,
    tool_calls: list[dict[str, Any]],
    tools_map: dict[str, BaseTool],
    abort_signal: AbortSignal | None,
) -> bool:
    tool_results = await execute_tool_batch(
        tool_calls, tools_map=tools_map,
        context=state, abort_signal=abort_signal,
    )
    terminated = False
    batch = RetrospectBatch(state, tools_map)

    for result in tool_results:
        call_id = result.tool_call_id or ""

        if state.hooks.on_after_tool_call:
            await state.hooks.on_after_tool_call(
                call_id, result.tool_name, result.input_args or {}, result,
            )

        content = batch.process_result(result)

        tool_step = StepRecord.tool(
            state,
            sequence=await state.session_runtime.allocate_sequence(),
            tool_call_id=call_id,
            name=result.tool_name,
            content=content,
            content_for_user=result.content_for_user,
            is_error=not result.is_success,
        )
        batch.register_step(call_id, tool_step.id, tool_step.sequence)
        await commit_step(state, tool_step)

        if not terminated and result.termination_reason is not None:
            set_termination_reason(state, result.termination_reason)
            terminated = True

    outcome = await batch.finalize()
    if outcome.applied:
        replace_messages(state, outcome.messages)

    return terminated or state.is_terminal
```

Import 从 4 个函数简化为 1 个类：`from agiwo.agent.retrospect import RetrospectBatch`。

## Repo Guard Rules

### AGW045 — run_loop 只能通过 RetrospectBatch 使用 retrospect

```
文件: agiwo/agent/run_loop.py
检测: import 或 from 引用 agiwo.agent.retrospect.triggers 或 agiwo.agent.retrospect.executor
消息: "run_loop must interact with retrospect only through RetrospectBatch; "
      "do not import retrospect internals (triggers/executor) directly."
```

防止有人绕过 RetrospectBatch 在 run_loop 中重新散落 retrospect 逻辑。

### AGW046 — 禁止 _retrospect magic flag

```
文件: agiwo/ 和 console/ 下所有 Python 文件（排除 tests/ 和 repo_guard.py）
检测: 字符串字面量中包含 "_retrospect"
消息: "Do not use _retrospect magic flags for inter-module signaling; "
      "RetrospectBatch detects retrospect tool results by tool_name."
```

防止 magic flag 被重新引入。

### AGW047 — retrospect/ 不得依赖 scheduler/

```
文件: agiwo/agent/retrospect/ 下所有 Python 文件
检测: import 或 from 引用 agiwo.scheduler
消息: "retrospect/ must not depend on scheduler; dependency direction is "
      "scheduler -> agent, not the reverse."
```

保护依赖方向。

### AGW048 — 禁止在 run_loop 中直接操作 retrospect 追踪状态

```
文件: agiwo/agent/run_loop.py
检测: 属性访问 ledger.last_retrospect_seq / ledger.retrospect_pending_tokens / ledger.retrospect_pending_rounds
消息: "Do not access retrospect tracking state directly in run_loop; "
      "all retrospect state management is encapsulated in RetrospectBatch."
```

防止有人在 run_loop 中直接读写 retrospect 追踪计数器，绕过封装。

### AGW049 — retrospect 内部模块的 import 边界

```
文件: agiwo/agent/ 下除 retrospect/ 之外的所有 Python 文件
允许: 仅可 import agiwo.agent.retrospect（即 __init__.py 暴露的公开 API）
检测: import 或 from 引用 agiwo.agent.retrospect.triggers 或 agiwo.agent.retrospect.executor
消息: "Import retrospect internals only through agiwo.agent.retrospect public API; "
      "do not reach into triggers/executor sub-modules directly."
```

保护 retrospect 包的封装边界。

### File Growth Budget

```python
(re.compile(r"^agiwo/agent/retrospect/triggers\.py$"), 120),
(re.compile(r"^agiwo/agent/retrospect/executor\.py$"), 200),
```

## Module Changes

| 文件 | 变更内容 |
|------|---------|
| `agiwo/agent/retrospect.py` | 删除，拆为 `retrospect/` 包 |
| `agiwo/agent/retrospect/__init__.py` | `RetrospectBatch`、`RetrospectOutcome`，公开 API |
| `agiwo/agent/retrospect/triggers.py` | `RetrospectTrigger` 枚举、触发检查、notice 模板、tracking 累计 |
| `agiwo/agent/retrospect/executor.py` | offload / 替换 / 移除自身痕迹 / storage 更新；返回 `RetrospectOutcome` |
| `agiwo/agent/run_loop.py` | `_execute_tool_calls` 简化，import 改为 `RetrospectBatch` |
| `agiwo/scheduler/runtime_tools.py` | `RetrospectToolResultTool` 删除 `_retrospect` flag |
| `scripts/repo_guard.py` | 新增 AGW045–AGW049 规则 + growth budget |

## What Doesn't Change

- `RunLedger` 字段（`last_retrospect_seq` / `retrospect_pending_tokens` / `retrospect_pending_rounds`）不变
- `StepRecord.condensed_content` 不变
- `AgentOptions` retrospect 配置字段不变
- Console 前后端不变
- Storage 接口不变
