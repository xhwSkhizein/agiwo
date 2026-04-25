# Context Optimization

长时间运行的 agent（特别是 scheduler 编排场景）会面临上下文膨胀问题。SDK 提供两个独立的上下文优化机制：**Context Rollback**（空转回退）和 **Goal-Directed Review**（目标导向回顾与步步后退）。

两者与 compaction（上下文压缩）独立共存：compaction 在 token 窗口层面工作，rollback 和 review/step-back 在语义层面优化内容质量。

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

系统收到 `no_progress` 后，不会物理删除 canonical `RunLog`。scheduler 会追加一条 `RunRolledBack` fact，默认 step replay 会隐藏该范围内的 steps，agent 回到 WAITING 状态。下次唤醒时上下文恢复到本轮之前，就像这轮空转从未发生过；若运维或调试需要查看原始轨迹，可以在读取 `RunLog` 时显式打开 `include_rolled_back=True`。

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

## Goal-Directed Review

### 问题

agent 在工作中可能偏离目标，产生大量与目标无关的 tool 调用和结果。传统的基于 token 计数的触发机制（"结果太大了，回顾一下"）不够精准——应该在语义层面检测目标偏离并执行回退。

### 核心思路

采用 **"系统在关键节点强制 agent 进行目标对齐检查"**。Agent 声明里程碑和阶段性目标，系统在检查点（非 review 工具次数、连续错误、里程碑切换）强制注入一次性 review 通知，agent 必须评估当前进展与目标的匹配度；若偏离，系统执行 KV-cache-safe 的 step-back（仅替换 tool result 内容，不删除/重排业务 message）。无删除消息规则的显式例外：在执行 KV-cache-safe 的 step-back 时，允许仅移除或替换与 `review_trajectory` 配对的临时 review 通知/元数据，但不得删除或重排其他对话 message 或历史记录（仅可替换 tool result 内容）。

### 工作方式

#### 步骤 1：声明里程碑

Agent 在开始时调用 `declare_milestones` 声明目标拆解：

```python
declare_milestones(milestones=[
    {"id": "understand", "description": "理解 session 管理代码结构"},
    {"id": "diagnose", "description": "定位 session 泄漏根因"},
    {"id": "fix", "description": "修复泄漏并验证"},
])
```

系统记录这些里程碑到 `ReviewState`，并自动将第一个里程碑设置为 active。Agent 可以在执行过程中调用 `complete_active_milestone()` 切换里程碑。

#### 步骤 2：系统强制 Review

以下条件触发系统注入 `<system-review>` 要求 agent 调用 `review_trajectory`：

| 触发类型 | 条件 | 触发逻辑 |
|---------|------|---------|
| `STEP_INTERVAL` | 自上次 checkpoint/review 起 >= `review_step_interval` 个非 `review_trajectory` tool result | 常规检查点：评估是否聚焦目标 |
| `CONSECUTIVE_ERRORS` | 连续 >= 3 个 tool call 返回 error | 检测无效尝试模式 |
| `MILESTONE_SWITCH` | 里程碑状态变化（pending → active, active → completed） | 目标切换时的强制对齐 |

计数规则是显式的 tool-result 计数，不再使用 run-log sequence 差值。`declare_milestones` 和 scheduler 控制类工具都会计数；`review_trajectory` 本身不计数，并在成功消费后把计数重置为 0。同一个 tool batch 中即使有多个并发 tool result，也只会把 `<system-review>` 注入到第一个触发结果；后续结果不会重复拼接通知。

#### 步骤 3：Agent 回顾

Agent 调用 `review_trajectory` 提供结构化回顾：

```python
review_trajectory(
    aligned=False,
    experience=(
        "搜索了 session.py 和 manager.py，确认 session 生命周期由 SessionManager "
        "统一管理；后续应聚焦 GC 延迟导致的 session 未释放问题。"
    ),
)
```

当 `aligned=false` 时，系统将 checkpoint 之后的低价值 tool result 替换为 `experience` 内容。Agent 的 tool_call 完整保留（意图轨迹），tool result 被精简为经验总结（KV-cache 安全）。

Review 成功处理后，触发 review 的 tool result 中的 `<system-review>` 会从后续 prompt-visible content 中移除，但仍通过 append-only RunLog facts 保留可观测性。运行时写入以下事实作为权威状态源：

- `ReviewMilestonesUpdated`: 当前 milestone board
- `ReviewTriggerDecided`: 本次 review 触发原因、触发 tool step、计数
- `ReviewCheckpointRecorded`: `aligned=true` 后确认的 checkpoint
- `ReviewOutcomeRecorded`: review 结果、隐藏的 review metadata steps、清理/精简的 step ids

Persistent session 在下一轮启动时会从这些 facts 重建 `ReviewState`，Console 的 milestone board / review cycles 也从这些 facts 投影出的 runtime spans 构建，不再解析 `declare_milestones`、`review_trajectory` 或 `<system-review>` 文本作为权威状态。

#### 上下文效果

Review 前：
```text
assistant: [tool_call: search_file(pattern="*.py")]
tool:      [5000 tokens 的文件列表]
assistant: [tool_call: search_db(query="random_table")]  ← 偏离目标
tool:      [3000 tokens 的查询结果 + <system-review>请调用 review_trajectory]
assistant: [tool_call: review_trajectory(aligned=false, experience="...")]
tool:      [确认信息]
```

Step-back 后：
```text
assistant: [tool_call: search_file(pattern="*.py")]
tool:      [5000 tokens 的文件列表]
assistant: [tool_call: search_db(query="random_table")]  ← tool_call 保留
tool:      [搜索了 random_table，与当前诊断目标无关，该表不包含 session 信息]  ← 替换为经验
assistant: [tool_call: search_db(query="session_registry")]  ← 基于经验修正方向
```

### 配置

```python
AgentOptions(
    enable_goal_directed_review=True,   # 默认开启
    review_step_interval=8,             # 每 N 个非 review tool result 触发常规 review
    review_on_error=True,               # 连续错误时是否触发 review
)
```

### KV Cache 安全性

Step-back 仅修改 tool result 的 `content` 字段，不删除或重排任何业务 message。Agent tool_call 的意图链条完整保留。唯一例外是 `review_trajectory` 自身的 tool_call + tool_result 这对临时 review 元数据：它们在 step-back 或 metadata-only cleanup 后可以从后续 prompt-visible context 隐藏（仅此一对），避免额外占据上下文。这个例外与"不删除或重排其他对话 message"规则并列存在。这些约束确保 LLM 的 KV cache 不被破坏。

### 约束

- 仅 scheduler 场景生效（`declare_milestones` 和 `review_trajectory` 工具通过 scheduler 注入）
- 里程碑由 agent 维护和拆解，用户仅提供顶层大目标
- 系统决定 review 时机和 step-back 范围，agent 仅提供内容
- 与 compaction/rollback 独立共存

### 升级兼容性

Goal-directed review 移除了旧的 tool-result rewrite 运行记录反序列化路径。升级到该版本前，如果本地 `.agiwo` 或 Console SQLite 数据库中仍有旧 session/run-log 数据，先清理旧的 `*.db` 文件或 `.agiwo` 状态目录；否则运行时会返回明确错误，提示清理旧数据库后重启。

---

## 架构

### review 包结构

```text
agiwo/agent/review/
├── __init__.py          # 公开 API：ReviewBatch
├── goal_manager.py      # 里程碑管理（declare/complete/activate）
├── replay.py            # 从 RunLog facts 重建 ReviewState
├── review_enforcer.py   # 触发条件检测、system-review 注入
├── step_back_executor.py # KV-cache-safe 内容精简
```

`run_tool_batch.py` 通过 `ReviewBatch` 与 review 交互：

```python
batch = ReviewBatch(context.config, context.ledger, runtime.tools_map)

for result in tool_results:
    content = batch.process_result(result, current_seq=seq)  # 可能注入 notice
    # ... commit step ...
    batch.register_step(call_id, step.id, step.sequence)

outcome = await batch.finalize(
    storage=context.session_runtime.run_log_storage,
    session_id=context.session_id,
    run_id=context.run_id,
    agent_id=context.agent_id,
)
if outcome.mode != "none":
    if outcome.hidden_step_ids:
        await writer.record_context_steps_hidden(step_ids=outcome.hidden_step_ids)
    for update in outcome.content_updates:
        _replace_tool_message_content(
            context.ledger.messages,
            tool_call_id=update.tool_call_id,
            content=update.content,
        )
    _remove_review_tool_call(
        context.ledger.messages,
        review_tool_call_id=outcome.review_tool_call_id,
    )
    await writer.record_review_outcome_recorded(...)
```

### 设计原则

- **系统强制 review**：不依赖 agent 自我意识，系统在检查点注入 review 通知
- **事实优先**：milestone、trigger、checkpoint、outcome 都写入 first-class RunLog facts；replay、trace、Console 视图从 facts 构建
- **KV-cache 安全**：仅替换 tool result content，不删除/重排 message，不调用 rebuild_messages
- **意图链条完整**：tool_call 永远保留，经验信息以 tool result 形式反馈给 LLM
- **依赖方向**：`review/` 不依赖 `scheduler/`，`scheduler/runtime_tools.py` 提供 `DeclareMilestonesTool` 和 `ReviewTrajectoryTool`
