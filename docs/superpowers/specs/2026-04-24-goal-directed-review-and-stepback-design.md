# Goal-Directed Review & StepBack

> 替换现有的 token/round-count-based retrospect，引入基于目标对齐度的 review 机制。
> 解决三大问题：System prompt/tool schema 未对齐、频繁提醒成为噪音、message offload 破坏 KV cache。

## Problem

当前 retrospect 有 3 个根本性问题：

1. **Review 没有语义锚点**：基于 token 计数和 step 轮次触发，不关心 agent 是否真的偏离目标。agent 做有用工作时也可能反复触发，成为噪音。

2. **System prompt 与 tool schema 割裂**：system prompt 完全不提 `<system-notice>` 或 retrospect，tool schema 中的 `retrospect_tool_result` 是孤立的。agent 不理解 system-notice 的权威性和处理优先级。

3. **KV cache 破坏**：`execute_retrospect` deep-copy 整个 message list，替换所有 tool content 为 placeholder，删除 retrospect tool_call + result，然后 `rebuild_messages` 重建。代价是 KV cache 全量失效 + 重新编码的 token 成本。

## Design

### 1. 核心思路

从 **"按 token 计数触发 → agent 调用 retrospect → 系统 offload"** 转变为 **"基于目标对齐的系统 review → agent 确认/否认 → 系统执行 targeted content 替换"**。

3 个核心组件：

| 组件 | 职责 | 决策权 |
|------|------|--------|
| **Goal Manager** | 存储 agent 声明的 milestones/子目标，追踪当前进度 | Agent 声明，系统存储 |
| **Review Enforcer** | 在检查点主动暂停，注入 `<system-review>`，要求 agent 审视轨迹 | **系统**决定何时 review |
| **StepBack Executor** | 将偏离区间的 tool result content 替换为 experience，保留 tool_call 链 | **系统**决定回退范围，**Agent** 提供经验内容 |

### 2. 交互流程

```text
1. User: "fix the session timeout bug"
2. Agent: declare_milestones([
     {id:"understand", desc:"理解 session 管理逻辑"},
     {id:"locate", desc:"定位超时根因"},
     {id:"fix", desc:"修复并验证"}
   ])
3. Agent 执行 tool calls...
4. 每 N 步或 milestone 切换时，系统注入 <system-review>
5. Agent: review_trajectory(aligned=true)
   → 系统记录 checkpoint
   或 Agent: review_trajectory(aligned=false, experience="...")
   → 系统执行 step-back
6. StepBack: 偏离区间的 tool result content 替换为 experience
7. Agent 从 checkpoint 继续执行
8. 重复 4-7
```

Agent 的角色简化为两件事：
- 声明/更新 milestones
- 在 review 时回答"是否对齐"并提供经验

---

## 3. 数据模型

### Milestone

```python
@dataclass
class Milestone:
    id: str                    # agent 自定义标识
    description: str           # 具体可验证的子目标描述
    status: Literal["pending", "active", "completed", "abandoned"]
    declared_at_seq: int       # 声明时的 step sequence
    completed_at_seq: int | None
```

存储在 `RunLedger` 上，属于 run 生命周期。

### ReviewCheckpoint

```python
@dataclass
class ReviewCheckpoint:
    seq: int                   # 确认"轨迹正确"的 step sequence
    milestone_id: str          # 当时的 active milestone
    confirmed_at: datetime
```

每次 review 通过时记录。回退时，系统默认回退到最近 checkpoint 之后的所有 tool result。

### ReviewState

```python
@dataclass
class ReviewState:
    milestones: list[Milestone] = field(default_factory=list)
    last_review_seq: int = 0
    latest_checkpoint: ReviewCheckpoint | None = None
    consecutive_errors: int = 0
    pending_review_reason: Literal["milestone_switch"] | None = None
```

约束：

- `ReviewState` 仍然是 **run-scoped 内存状态**，不作为单独的 run-log family 持久化。
- `last_review_seq` 只表示“最近一次 review 发生在何时”，用于步数间隔触发。
- `latest_checkpoint` 是 step-back 回退起点的 canonical owner，替代仅靠 `last_checkpoint_seq` 的弱语义。
- `pending_review_reason` 只表达“下一次合适时机必须 review”，当前只需要 `milestone_switch` 一种原因。

### Temporary Review Metadata Visibility

`review_trajectory` 的 assistant tool-call step 和 tool-result step 都仍然作为
**committed step** 写入 run-log，供 trace / observability / 调试使用；但它们属于
**temporary review metadata**，不能在后续 run 的 prompt rebuild 中再次出现。

为此需要新增一类 append-only runtime fact（例如 `ContextStepsHidden`）：

```python
@dataclass(frozen=True, kw_only=True)
class ContextStepsHidden(RunLogEntry):
    step_ids: list[str]
    reason: Literal["review_metadata"]
```

语义：

- canonical truth 仍然是 committed steps + hidden facts。
- observability 默认仍可看到完整 review 历史。
- prompt assembly / `list_step_views(..., include_hidden_from_context=False)` 必须过滤被隐藏的 steps。
- 这类 fact **不需要**对外暴露成新的 stream event；它是 prompt replay 的内部控制事实。

---

## 4. Review Enforcer

### 触发条件（替换旧的 token/round 计数）

| 触发条件 | 逻辑 | 意图 |
|---------|------|------|
| **Milestone 切换** | Agent 通过 `declare_milestones` 更新 milestone `status`，active milestone 发生切换 | 自然检查点 |
| **步数间隔** | 自上次 review 后超过 N 步（默认 8） | 防止长时间无反思 |
| **连续失败** | tool result 连续 error ≥ 2 次 | "这条路可能走不通" |

去掉 `retrospect_token_threshold`、`retrospect_round_interval`、`retrospect_accumulated_token_threshold`。

补充语义：

- **初次声明 milestones** 不触发 review。
- **milestone switch** 不在 `declare_milestones` 的 tool result 上立即注入 review，而是设置
  `pending_review_reason="milestone_switch"`，并在 **下一次非 review tool result** 上注入
  `<system-review>`，避免声明后立刻打断。

### Review Prompt 格式

```text
<system-review>
Active milestone: "{milestone.description}"

Trigger: {trigger_reason}
Steps since last review: {step_count}
{optional review_advice from hooks}

Question: Do the last N steps meaningfully advance the active milestone?
If not, use review_trajectory to:
  1. Indicate misalignment (aligned=false)
  2. Provide a concise experience summary of what was learned
</system-review>
```

说明：

- `Recent trajectory` 渲染是后续可选增强，不是本次 cleanup / closure 的必需项。
- 本次只要求 system prompt、tool schema、trigger reason 与 hook advice 对齐。

### 配置

```python
class AgentOptions(BaseModel):
    enable_goal_directed_review: bool = True
    review_step_interval: int = 8
    review_on_error: bool = True
```

---

## 5. StepBack Executor

### 核心原则

**业务 tool result 只做 targeted content replacement；review metadata 不再依赖“只改内存消息列表”来隐藏。**

| | 当前 Retrospect | 新 StepBack |
|---|---|---|
| Message 删除 | 删除 retrospect 自身的 tool_call + tool_result | prompt-visible 上隐藏 review_trajectory 自身的 tool_call + tool_result；run-log 仍保留 |
| Content 替换 | 替换为 `[offloaded to disk]` placeholder | 替换为 `[EXPERIENCE] ...` |
| 业务 message 变更 | 可能改变 message 顺序（删除 + 重排） | 不删除不新增，只改 content 字段 |
| KV cache 影响 | 删除 + 重排导致全量失效 | 业务 msg 的 content 替换只影响替换点之后；review metadata 通过 hidden fact 从未来 prompt rebuild 过滤 |

### 执行流程

```text
Before StepBack:
  msg[0]: system prompt        ← KV cache ✓
  msg[1]: user: "fix the bug"  ← KV cache ✓
  msg[2]: assistant: tool_call(search_code("session"))  ← KV cache ✓
  msg[3]: tool: [5000 tokens]  ← 需要精简
  msg[4]: assistant: tool_call(search_code("jwt"))
  msg[5]: tool: [3000 tokens]  ← 需要精简

After StepBack (aligned=false):
  msg[0]: system prompt        ← unchanged
  msg[1]: user: "fix the bug"  ← unchanged
  msg[2]: assistant: tool_call(search_code("session"))  ← unchanged
  msg[3]: tool: "[EXPERIENCE] search_code('session'): found SessionManager..."
  msg[4]: assistant: tool_call(search_code("jwt"))       ← preserved
  msg[5]: tool: "[EXPERIENCE] search_code('jwt'): confirmed TokenValidator..."
```

**KV cache 影响：** msg[0..2] 前缀保留；msg[3..5] 从 ~8000 tokens 压缩到 ~200 tokens，重新编码代价极小。下次 LLM 调用的总 token 数大幅下降。

### State Writer 集成

不调用 `rebuild_messages`（这是 KV cache 的最大杀手），改为 targeted update：

```python
async def _apply_step_back_outcome(context, outcome, writer):
    if not outcome.applied:
        return
    # 只更新仍然 prompt-visible 的 tool result content
    for idx, new_content in outcome.content_updates:
        context.ledger.messages[idx]["content"] = new_content
    # 将 review metadata 标记为 hidden，而不是仅依赖当前内存中的删除
    await writer.record_context_steps_hidden(
        step_ids=outcome.hidden_step_ids,
        reason="review_metadata",
    )
    entries = await writer.record_step_back_applied(...)
```

### 原始内容保留

和当前 `condensed_content` 逻辑一致：原始 content 异步持久化到 storage，内存中的 content 被替换为 experience。Storage 不可变——`content` 字段保留原始内容，`condensed_content` 存储精简版。

### Review Metadata Cleanup Contract

`aligned=true` 和 `aligned=false` 两条路径都必须复用同一套 review metadata cleanup 逻辑。

统一 contract：

- `ReviewBatch` 维护 per-batch lookup：
  - `tool_call_id -> assistant_step_id`
  - `tool_call_id -> tool_step_id`
- `review_trajectory` 在 aligned / step-back 两条路径下都返回同一种结构化 outcome：
  - `content_updates`
  - `hidden_step_ids`
  - `step_back_applied`
- 不再在多个模块中各自实现一份 `remove_review_tool_call(...)` 的近似逻辑。

---

## 6. System Prompt 对齐

在 system prompt 中新增专门 section：

```text
## Goal-Directed Review

You are expected to work in a goal-directed manner. The system helps you
stay on track through a review mechanism.

### Milestones
When you receive a task, break it into concrete milestones using the
declare_milestones tool. Each milestone should be a verifiable
sub-goal.

### System Reviews
The system will periodically ask you to review your trajectory against
the active milestone. When you see a <system-review> tag in a tool
result, you MUST respond with the review_trajectory tool:
- aligned=true: briefly note what was accomplished.
- aligned=false: provide a concise experience summary.
The system review is not optional — treat it as a mandatory checkpoint.

### Step-Back
When you indicate misalignment, the system automatically condenses the
off-target tool results into your experience summary. The tool call
history is preserved so future decisions can reference what was tried,
but the verbose outputs are replaced with the lesson learned.
```

### Tool Schema 对齐

**`declare_milestones`:**
```text
Declare or update the milestones for the current task. Break the user's
request into concrete, verifiable sub-goals. Each milestone should have
a clear id and a specific description of what 'done' looks like. Milestones
may include an optional status field (`pending`, `active`, `completed`,
`abandoned`) so the agent can mark progress and switch the active milestone.
```

**`review_trajectory`:**
```text
Respond to a <system-review> prompt. Parameters:
- aligned: true if trajectory aligns with active milestone
- experience: (required when aligned=false) What was attempted, learned,
  and how this should inform the next approach.
```

---

## 7. 与 Compaction 的协作

| | StepBack | Compaction |
|---|---|---|
| 触发层 | 语义层（目标偏离） | 物理层（token 窗口满） |
| 操作 | 替换偏离步骤的 content | 将上下文压缩为 summary |
| KV cache | 后半段受影响，但 token 大幅减少 | 完全重建（本质代价） |

StepBack 先于 Compaction，语义层控制膨胀 → 减少 Compaction 触发频率。

---

## 8. 废弃旧代码的清理范围

| 删除/替换 | 说明 |
|-----------|------|
| `agiwo/agent/retrospect/` 整个包 | 替换为 `GoalManager` + `ReviewEnforcer` + `StepBackExecutor` |
| `RetrospectToolResultTool` | 替换为 `declare_milestones` + `review_trajectory` |
| `RetrospectState` | 替换为 `Milestone[]` + `ReviewCheckpoint` |
| 4 个 retrospect 配置字段 | 替换为 `review_step_interval` + `review_on_error` |
| Repo guard AGW045-049 | 更新为新模块的对应规则 |
| `BEFORE_RETROSPECT` / `AFTER_RETROSPECT` hook phases | 替换为 `BEFORE_REVIEW` / `AFTER_STEP_BACK` |
| `RetrospectApplied` log/stream 类型 | 替换为 `StepBackApplied` |
| Console retrospect 展示逻辑 | 替换为 step-back 展示 |
| Console retrospect 配置表单 / TS payload types | 删除并替换为 review 配置字段 |
| 现行文档与 archived 文档中的 retrospect 术语 | 全量替换为 review / step-back 语义，并在需要处标注“已被替代” |
| 旧测试 `test_retrospect.py` 等 | 替换为新测试 |
| `GoalManager` 薄 OO wrapper | 删除，只保留模块级状态机 helper |
| review tool 中未使用的 scheduler port 依赖 | 删除，保持工具纯粹 |

### 保留/复用

- `StepView.condensed_content` — 保留，语义不变
- `append_step_condensed_content` storage API — 保留
- backend service 的公开 HTTP schema 不需要新增 review 专属 API；但 Console 前端表单与 TS payload 类型需要收敛到真实字段

---

## Module Changes

### 新增

| 模块 | 内容 |
|------|------|
| `agiwo/agent/review/__init__.py` | 公开 API：`ReviewBatch`, `StepBackOutcome` |
| `agiwo/agent/review/goal_manager.py` | `Milestone` 数据模型 + 声明/更新/查询逻辑 |
| `agiwo/agent/review/review_enforcer.py` | 触发条件检查、`<system-review>` notice 注入 |
| `agiwo/agent/review/step_back_executor.py` | Content 替换、experience 注入、storage 持久化 |
| `agiwo/agent/models/review.py` | `Milestone`, `ReviewCheckpoint` 数据类 |
| `agiwo/scheduler/runtime_tools.py` | `DeclareMilestonesTool` + `ReviewTrajectoryTool` |

### 修改

| 模块 | 变更 |
|------|------|
| `agiwo/agent/prompt.py` | 新增 Goal-Directed Review section（system prompt 模板） |
| `agiwo/agent/run_tool_batch.py` | `RetrospectBatch` → `ReviewBatch`，去掉 `rebuild_messages`，统一 review metadata cleanup outcome |
| `agiwo/agent/models/run.py` | `RunLedger` 上新字段收敛到 `ReviewState`（含 `latest_checkpoint` / `pending_review_reason`） |
| `agiwo/agent/models/config.py` | 替换 4 个 retrospect 字段为 review 配置 |
| `agiwo/agent/models/log.py` | `RetrospectApplied` → `StepBackApplied`，新增 hidden-from-context fact（如 `ContextStepsHidden`） |
| `agiwo/agent/models/stream.py` | `RetrospectAppliedEvent` → `StepBackAppliedEvent` |
| `agiwo/agent/hooks.py` | `BEFORE/AFTER_RETROSPECT` → `BEFORE_REVIEW` / `AFTER_STEP_BACK`，并补齐真实调用点 |
| `agiwo/agent/storage/serialization.py` | 类型名称更新 |
| `agiwo/agent/trace_writer.py` | Span kind 名称更新 |
| `agiwo/agent/run_bootstrap.py` | prompt rebuild 改为读取 context-visible steps，而非所有 committed steps |
| `agiwo/agent/storage/base.py` / `sqlite.py` | `list_step_views(...)` 增加 hidden-from-context 过滤语义 |
| `agiwo/scheduler/runtime_tools.py` | `declare_milestones` 支持可选 `status`，review tools 去掉无用 port 依赖 |
| `console/web/src/components/agent-form.tsx` | 删除旧 retrospect 配置，补齐真实 review 配置 |
| `console/web/src/components/session-detail/session-observability-panel.tsx` | 统一展示 `step_back`，移除 `retrospect` 分支 |
| `console/web/src/lib/api.ts` | TS payload / event kind 与后端真实字段对齐 |
| `docs/**` | 清理现行与 archived 文档中的旧 retrospect 术语 |
| `scripts/repo_guard.py` | 规则更新 |

### 删除

| 模块 | 说明 |
|------|------|
| `agiwo/agent/retrospect/` | 整个包 |
| `agiwo/scheduler/runtime_tools.py::RetrospectToolResultTool` | 类删除 |

---

## 约束

- 仅 `enable_goal_directed_review=True` 时生效
- Milestones 不跨 run 持久化（每次 run 重新声明）
- 与 Compaction 独立共存，StepBack 先于 Compaction
- `<system-review>` 只追加在 tool result content 末尾
- 初次声明 milestones 不能立即触发 review
- milestone switch review 采用“下一次非 review tool result 注入”语义
- temporary review metadata 必须对未来 prompt rebuild 不可见，但对 trace / observability 仍可见
