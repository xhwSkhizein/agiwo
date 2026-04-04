# Context Rollback & Tool Result Retrospect

> 两个独立的实验性 feature，用于 scheduler 编排场景下的上下文管理优化。
> 解决长时间运行的 agent 因"空转唤醒"和"大量无效 tool result"导致上下文膨胀的问题。

## Problem

在 scheduler 编排场景中，主 agent 存在两类上下文浪费：

1. **唤醒空转**：agent 被周期性唤醒（PERIODIC）检查子 agent 进展，发现没有新结果后又 sleep 回去。每次空转都产生完整的 user(wake message) + assistant(response) steps，占据上下文窗口但不带来信息增量。多次空转后上下文被无效内容填满。

2. **Tool call 无效结果**：agent 尝试性地调用 tool（如查询数据库、搜索文件），得到了大量结果但对目标帮助有限。完整的 tool result（可能数千 tokens）留在上下文中，真正有价值的只是"这条路不行"这个结论。

## Design

### Feature 1: Scheduler Context Rollback

**目标**：唤醒空转时整轮删除 steps，就像这轮唤醒从未发生过。

**配置**：`AgentOptions.enable_context_rollback: bool = True`

#### 触发方式

仅靠 agent 主动声明。agent 被周期唤醒后，检查发现没有进展，调用 `sleep_and_wait` 时声明 `no_progress=True`：

```python
sleep_and_wait(
    wake_type="periodic",
    delay_seconds=60,
    time_unit="minutes",
    no_progress=True,
    explain="子 agent 仍在运行，暂无新结果"
)
```

没有兜底规则。agent 不声明 `no_progress` 时，行为与当前完全一致。

#### 回退操作

在 `SchedulerRunner._handle_periodic_output` 中，当检测到 `no_progress` 标记时：

1. 记录本轮 run 产生的 step 序列范围（`run_start_seq` ~ `run_end_seq`）
2. 调用 `run_step_storage.delete_steps(session_id, start_seq=run_start_seq)` 删除这些 steps
3. 在 `AgentState` 上递增 `rollback_count`（仅 observability 用途，agent 不可见）
4. agent 回到 WAITING 状态，下次唤醒时上下文恢复到本轮之前

#### 信号传递

`no_progress` 标记需要从 tool 执行传递到 scheduler runner：

1. `SleepAndWaitTool.execute()` 接收 `no_progress` 参数
2. 将标记传递到 `SleepRequest` → `SchedulerToolControl.sleep_current_agent()`
3. 写入 `AgentState` 的某个瞬态字段（如 `no_progress: bool`），供 runner 读取
4. `SchedulerRunner._handle_periodic_output` 读取后执行回退

#### Prompt 行为

不在 wake message 中提及 rollback。agent 不需要感知回退发生过。

#### 约束

- 仅 `enable_context_rollback=True` 时 `sleep_and_wait` 接受 `no_progress` 参数
- 仅 scheduler 场景生效
- `no_progress=True` 只在 `wake_type=periodic` 时有意义（timer/waitset 场景下忽略）

---

### Feature 2: Tool Result Retrospect

**目标**：agent 对无效的 tool result 做回顾反思，用精简的 feedback 替换上下文中的冗长结果，同时保留 storage 中的原始数据供审计。

**配置**：

```python
class AgentOptions(BaseModel):
    enable_tool_retrospect: bool = True
    retrospect_token_threshold: int = 1024       # 单次 tool result token 阈值
    retrospect_round_interval: int = 5           # 每 N 轮 tool call 触发
    retrospect_accumulated_token_threshold: int = 8192  # 累计 token 阈值
```

#### 触发条件（三种，任一满足即注入 system-notice）

| 触发类型 | 条件 | 检查时机 |
|---------|------|---------|
| 单次大结果 | 单个 tool result 的 content 超过 `retrospect_token_threshold` | tool 执行完成后 |
| 轮次累计 | 自上次 retrospect 后连续 `retrospect_round_interval` 轮 tool call | tool 执行完成后 |
| Token 累计 | 自上次 retrospect 后 tool result 累计 tokens 超过 `retrospect_accumulated_token_threshold` | tool 执行完成后 |

#### System Notice 注入

满足触发条件时，在 tool result content 末尾追加：

```
<system-notice>Tool result is large. Please retrospect: if this result has limited value for your current goal, call retrospect_tool_result to replace it with a concise feedback summary.</system-notice>
```

注入发生在 `run_loop.py` 的 `_execute_tool_calls` 中，commit tool step 之前。

#### `retrospect_tool_result` Tool

**位置**：`agiwo/scheduler/runtime_tools.py`，作为 scheduler runtime tool 注入。

**参数**：

```python
{
    "feedback": {
        "type": "string",
        "description": (
            "A thorough retrospective summary: what was tried, "
            "what was learned, what constraints were discovered, "
            "and how this should inform the next approach. "
            "This replaces the original verbose results, so include "
            "all decision-relevant information."
        )
    }
}
```

Agent 只需要提供 feedback，范围由系统运行时自动确定。

**Tool description 引导 agent 做高质量复盘**：

```
Retrospect recent tool call results that provided limited value.
When you see a <system-notice> suggesting retrospect, review the
recent tool results and decide whether they meaningfully advance
your goal.

If not, call this tool with a thorough feedback that includes:
1. What you attempted and why
2. What the results revealed (or failed to reveal)
3. Key lessons or constraints discovered
4. How this informs your next steps

Your feedback will replace the verbose tool results in context,
so make it complete enough to guide future decisions without
needing to refer back to the original output.
```

#### 执行流程

agent 调用 `retrospect_tool_result(feedback="...")` 后，系统运行时：

1. **确定范围**：从 `last_retrospect_seq`（上次 retrospect 覆盖到的 step sequence，或 run 起点）到当前位置，所有未 retrospect 的 tool steps
2. **Offload 原始内容**：每个 tool step 的原始 content 写入临时磁盘文件
3. **更新内存上下文**：`RunLedger.messages` 中，每个 tool message 的 content 替换为 `"[详细结果已归档: {file_path}]"`
4. **追加 feedback**：范围内最后一个 tool message 的 content 追加 agent 的 feedback
5. **移除自身痕迹**：从 `RunLedger.messages` 中移除 `retrospect_tool_result` 自身的 tool_call + tool_result（透明化）
6. **更新跟踪状态**：重置 `retrospect_pending_tokens` 和 `retrospect_pending_rounds`，更新 `last_retrospect_seq`

#### Storage 不可变

`StepRecord` 新增 `condensed_content: str | None = None` 字段：

- **Storage 持久化**：`content` 字段永远保留原始内容不修改。retrospect 时只写入 `condensed_content`
- **RunLedger.messages（内存）**：直接替换为精简内容
- **`StepRecord.to_message()`**：加载 steps 构建 messages 时，`condensed_content` 优先 — `condensed_content or content`。跨 run 重新加载历史时自动使用精简版

Console/trace 可以同时展示原始和精简内容，完整还原当时发生了什么。

#### 上下文效果示例

Retrospect 之前：

```
assistant: [tool_call: search_db(query="plan A")]
tool:      [3000 tokens 的查询结果]
assistant: [tool_call: search_db(query="plan B")]
tool:      [2000 tokens 的查询结果 + <system-notice>...]
assistant: [tool_call: retrospect_tool_result(feedback="...")]
tool:      [确认信息]
```

Retrospect 之后：

```
assistant: [tool_call: search_db(query="plan A")]
tool:      [详细结果已归档: /tmp/session_xx/tool_call_1.txt]
assistant: [tool_call: search_db(query="plan B")]
tool:      [详细结果已归档: /tmp/session_xx/tool_call_2.txt
            ---
            Retrospect: 尝试了 plan A 和 plan B 两种查询方式，
            均未找到目标数据。该表不包含所需字段，需要换用 Y 表。]
```

assistant 的 tool_call 完整保留（探索轨迹），tool result 被 offload + 精简，retrospect 自身的 tool_call + tool_result 透明移除。

#### 运行时状态跟踪

在 `RunLedger` 上增加：

```python
@dataclass
class RunLedger:
    # ... 现有字段 ...
    last_retrospect_seq: int = 0
    retrospect_pending_tokens: int = 0
    retrospect_pending_rounds: int = 0
```

这些字段仅在 run 内有效（run 结束后自然重置），不需要持久化。

#### 约束

- 仅 `enable_tool_retrospect=True` 时注入 notice 和提供 retrospect tool
- 仅 scheduler 场景生效（通过 scheduler runtime tools 注入控制）
- Retrospect 只能操作本轮 run 中的 steps，不能跨 run
- 与 compaction 独立共存：retrospect 在语义层面精简内容，compaction 在 token 窗口层面工作

---

## Module Changes

### 需要变更的模块

| 模块 | 变更内容 |
|------|---------|
| `agiwo/agent/models/config.py` | `AgentOptions` 增加 `enable_context_rollback`、`enable_tool_retrospect` 及 retrospect 阈值字段 |
| `agiwo/agent/models/step.py` | `StepRecord` 增加 `condensed_content` 字段；`to_message()` 优先使用 `condensed_content` |
| `agiwo/agent/models/run.py` | `RunLedger` 增加 `last_retrospect_seq`、`retrospect_pending_tokens`、`retrospect_pending_rounds` |
| `agiwo/agent/storage/base.py` | `InMemoryRunStepStorage.save_step` 透传 `condensed_content` |
| `agiwo/agent/storage/sqlite.py` | SQLite schema 增加 `condensed_content` 列 |
| `agiwo/agent/run_loop.py` | `_execute_tool_calls` 中增加 retrospect 触发条件检查和 system-notice 注入 |
| `agiwo/scheduler/runtime_tools.py` | 新增 `RetrospectToolResultTool`；`SleepAndWaitTool` 增加 `no_progress` 参数 |
| `agiwo/scheduler/commands.py` | `SleepRequest` 增加 `no_progress` 字段 |
| `agiwo/scheduler/tool_control.py` | 传递 `no_progress` 到 `AgentState` |
| `agiwo/scheduler/models.py` | `AgentState` 增加 `rollback_count` 和 `no_progress` 瞬态字段 |
| `agiwo/scheduler/runner.py` | `_handle_periodic_output` 处理 `no_progress` 引发的 full rollback |

### Console 前端展示

| 模块 | 变更内容 |
|------|---------|
| `console/server/models/view.py` | `StepResponse` 增加 `condensed_content` 字段 |
| `console/server/response_serialization.py` | `step_response_from_sdk` 映射 `condensed_content` |
| `console/web/src/lib/api.ts` | `StepResponse` 类型增加 `condensed_content` |
| `console/web/src/lib/chat-types.ts` | `ChatMessage` 增加 `originalContent`；`messageFromStep` 处理双内容 |
| `console/web/src/hooks/use-chat-stream.ts` | SSE tool step 处理透传 `condensed_content` |
| `console/web/src/components/chat-message.tsx` | Tool result 默认展示 retrospect 内容，可点击查看原始内容 |
| `console/web/src/app/sessions/[id]/page.tsx` | `StepCard` 同样支持原始/精简切换 |

### 不涉及的模块

- `agiwo/agent/compaction.py` — 不改
- `agiwo/agent/prompt.py` — 不改（通过 `step.to_message()` 自然获得精简内容）
- `agiwo/observability/` — 不改
