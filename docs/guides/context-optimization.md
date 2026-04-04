# Context Optimization

长时间运行的 agent（特别是 scheduler 编排场景）会面临上下文膨胀问题。SDK 提供两个独立的上下文优化机制：**Context Rollback**（空转回退）和 **Tool Result Retrospect**（工具结果回顾）。

两者与 compaction（上下文压缩）独立共存：compaction 在 token 窗口层面工作，rollback 和 retrospect 在语义层面优化内容质量。

## Context Rollback

### 问题

在 scheduler 编排场景中，主 agent 被周期性唤醒（PERIODIC）检查子 agent 进展。如果子 agent 尚未完成，主 agent 只是确认"没有新结果"然后再次 sleep。每次空转都产生完整的 user(wake message) + assistant(response) steps，占据上下文窗口但不带来信息增量。

### 工作方式

agent 被周期唤醒后，如果判断没有新进展，在调用 `sleep_and_wait` 时声明 `no_progress=True`：

```python
sleep_and_wait(
    wake_type="periodic",
    delay_seconds=60,
    time_unit="minutes",
    no_progress=True,
    explain="子 agent 仍在运行，暂无新结果"
)
```

系统收到 `no_progress` 后，删除本轮产生的所有 steps，agent 回到 WAITING 状态。下次唤醒时上下文恢复到本轮之前，就像这轮空转从未发生过。

### 配置

```python
AgentOptions(
    enable_context_rollback=True,  # 默认开启
)
```

### 约束

- 仅 scheduler 场景生效
- 仅 `wake_type=periodic` 时 `no_progress` 有效
- agent 不需要感知回退发生过（wake message 不提及 rollback）

---

## Tool Result Retrospect

### 问题

agent 探索性地调用工具（查询数据库、搜索文件等），可能得到大量结果但对当前目标帮助有限。完整的 tool result（可能数千 tokens）留在上下文中，真正有价值的只是"这条路不行"这个结论。

### 工作方式

Retrospect 分两步：**系统触发提示** → **agent 主动回顾**。

#### 步骤 1：系统检测并注入提示

每次 tool 执行完成后，系统检查是否满足以下任一触发条件：

| 触发类型 | 条件 | 系统提示侧重 |
|---------|------|------------|
| 单次大结果 | 单个 tool result 超过 `retrospect_token_threshold` tokens | 强调 concise / replace |
| 轮次累计 | 连续 N 轮 tool call（`retrospect_round_interval`） | 强调目标聚焦、回顾方向 |
| Token 累计 | tool result 累计超过 `retrospect_accumulated_token_threshold` | 强调空间紧迫、清理低价值内容 |

满足条件时，系统在 tool result 末尾追加针对触发类型的 `<system-notice>`，引导 agent 调用 `retrospect_tool_result`。

#### 步骤 2：agent 调用 retrospect

agent 看到 system-notice 后，可以调用 `retrospect_tool_result(feedback="...")`，提供一段结构化的回顾总结。系统自动：

1. **确定范围**：从上次 retrospect 位置到当前，所有未处理的 tool steps
2. **归档原始内容**：每个 tool result 的原始 content 写入磁盘文件
3. **替换上下文**：tool message 替换为 `[ToolResult offloaded to ...]`，最后一个 tool message 追加 agent 的 feedback
4. **透明化**：retrospect 自身的 tool_call + tool_result 从上下文中移除
5. **持久化**：storage 中的 `condensed_content` 字段记录精简内容，`content` 字段保留原始数据

#### 上下文效果

Retrospect 前：
```
assistant: [tool_call: search_db(query="plan A")]
tool:      [3000 tokens 的查询结果]
assistant: [tool_call: search_db(query="plan B")]
tool:      [2000 tokens 的查询结果 + <system-notice>...]
assistant: [tool_call: retrospect_tool_result(feedback="...")]
tool:      [确认信息]
```

Retrospect 后：
```
assistant: [tool_call: search_db(query="plan A")]
tool:      [ToolResult offloaded to /path/tool_call_1.txt]
assistant: [tool_call: search_db(query="plan B")]
tool:      [ToolResult offloaded to /path/tool_call_2.txt
            ---
            Retrospect: 尝试了 plan A 和 plan B，均未找到目标数据。
            该表不包含所需字段，需要换用 Y 表。]
```

assistant 的 tool_call 完整保留（探索轨迹），tool result 被归档 + 精简，retrospect 自身透明移除。

### 配置

```python
AgentOptions(
    enable_tool_retrospect=True,                   # 默认开启
    retrospect_token_threshold=1024,               # 单次 tool result token 阈值
    retrospect_round_interval=5,                   # 每 N 轮 tool call 触发
    retrospect_accumulated_token_threshold=8192,   # 累计 token 阈值
)
```

### Storage 双写

`StepRecord` 的 `content` 字段永远保留原始内容。retrospect 只写入 `condensed_content` 字段。

- 加载历史 steps 构建 messages 时，`condensed_content` 优先（`condensed_content or content`）
- Console/trace 可同时展示原始和精简内容
- 跨 run 重新加载历史时自动使用精简版

### 约束

- 仅 scheduler 场景生效（通过 scheduler runtime tools 注入 `retrospect_tool_result` 工具）
- 只能操作本轮 run 中的 steps，不能跨 run
- 与 compaction 独立共存

---

## 架构

### retrospect 包结构

```
agiwo/agent/retrospect/
├── __init__.py      # 公开 API：RetrospectBatch, RetrospectOutcome
├── triggers.py      # RetrospectTrigger 枚举、触发检查、notice 模板
├── executor.py      # offload / 替换 / 持久化 / 清理
```

`run_loop.py` 只通过 `RetrospectBatch` 与 retrospect 交互：

```python
batch = RetrospectBatch(state, tools_map)

for result in tool_results:
    content = batch.process_result(result)     # 可能注入 notice
    # ... commit step ...
    batch.register_step(call_id, step.id, step.sequence)

outcome = await batch.finalize()               # 构建 retrospect 结果
if outcome.applied:
    replace_messages(state, outcome.messages)   # 显式应用上下文变更
```

### 设计原则

- **上下文变更显式化**：`finalize()` 返回 `RetrospectOutcome`，调用方通过 `replace_messages` 显式应用，和 compaction 的模式统一
- **Storage 不可变**：原始 `content` 永远保留，`condensed_content` 只追加不覆盖
- **无 magic flag**：retrospect tool 通过 `tool_name` 识别，不依赖隐式输出标记
- **依赖方向**：`retrospect/` 不依赖 `scheduler/`，`scheduler/runtime_tools.py` 提供 `RetrospectToolResultTool`
