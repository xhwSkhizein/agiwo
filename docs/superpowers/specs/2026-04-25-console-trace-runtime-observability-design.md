# Console Trace Runtime Observability

> 为 Console Web 增加面向回顾的 runtime observability。
> 目标是在 **不引入新的 runtime 真相源** 的前提下，让用户能够在 `Session Detail` 和 `Trace Detail` 中回顾当前 trace 内的完整 agent loop，尤其是 compaction / step-back / rollback / termination / review 这些关键 runtime decisions 与 review 细节。

## Problem

当前 runtime observability 有 4 个明显缺口：

1. **Trace Detail 还是通用 span dump，不是 loop replay。**
   `Trace Detail` 现在主要提供 token summary + span waterfall。虽然 `step_back`、`compaction`、`termination` 已经会被写成 runtime span，但它们只是混在 span 列表里，用户需要自己逐条展开判断发生了什么，无法快速回顾整个 agent loop。

2. **Session Observability 只展示每种 latest 一条。**
   Session 页右侧 `Runtime Decisions` 现在只展示 `latest_termination`、`latest_compaction`、`latest_step_back`、`latest_rollback`。如果同一 session 中连续发生多次 compaction / step-back / rollback，页面只保留每种“最后一次”，这不适合事后排查。

3. **Goal-directed review 细节仍然是可调试但不可回顾。**
   review checkpoint、`review_trajectory` 结果、`declare_milestones` 的输入/输出，目前大多只存在于 committed tool steps 或 trace tool spans 里，没有被整理成更适合阅读的 review lifecycle 视图。用户即使打开 trace，也只能在原始 tool span 或 step 文本中自己搜索 `<system-review>`、`aligned=false`、`milestones` 等片段。

4. **`step_back` 的展示语义当前不稳定，容易误导。**
   `step_back` 的设计语义是：
   - committed step 原始 `content` 保留原始 tool result
   - `condensed_content` 表示 prompt-visible / UI-default 的精简内容（例如 `[EXPERIENCE] ...`）

   但当前 `execute_step_back()` 写入 storage 的 `StepCondensedContentUpdated.condensed_content` 实际保存的是原始内容，而 UI 又优先展示 `condensed_content`。结果是 Session / Chat / Trace 相关页面可能把“原始结果”当成“step-back 后的 condensed 结果”展示，导致 observability 失真。

## Goals

这次改动要满足以下目标：

1. **Trace Detail 必须能回顾当前 trace 内的完整执行过程。**
   重点不是“展示所有 span JSON”，而是让用户理解：
   - loop 是怎么推进的
   - 哪些 review checkpoint 被触发
   - review 结果是什么
   - 是否发生了 step-back / compaction / rollback / termination
   - 这些 runtime decisions 的上下文和具体 payload 是什么

2. **Session Detail 必须展示“近期 runtime decisions”，而不是每种 latest 一条。**

3. **review 细节要可见，但放在次级展开层。**
   最外层只展示高信号摘要；review notice 文本、milestone payload、review tool raw output 等细节放到展开区，不强行塞到最外层。

4. **Trace / Session / Step UI 的语义必须一致。**
   同一条 `step_back` 在 trace、session 和 step card 中不能出现不同说法或相互冲突的展示。

5. **不引入新的 storage 真相源，不做 schema migration。**
   canonical truth 继续保持：
   - `RunLog` 是 runtime facts 的真相源
   - `Trace.spans` 是 committed run-log facts 投影出来的 trace view

## Non-Goals

这次不做下面这些事：

- 不把 Trace Detail 扩展成“跨 session / 邻近 trace / 整轮会话回放”
- 不把所有 review / milestone 信息都提升成新的 first-class run-log family
- 不在 Trace Storage 中再维护一套独立 runtime decision 表
- 不为旧数据做 migration；如果新展示依赖的新字段不存在，就按当前版本行为 fail-fast 或回退到已有信息
- 不把 Session 步骤列表变成“内部调试总线”；review/internal metadata 不应继续污染默认步骤视图

## Design

### 1. Truth Sources And Layering

这次改动继续维持两层真相源：

| 层 | 职责 | 备注 |
| --- | --- | --- |
| `RunLog` | canonical runtime facts | Session 页的 runtime decision 列表直接从这里重建 |
| `Trace.spans` | Trace 级 replayable observability view | Trace 页从这里构建 loop timeline、review insights 和 runtime decision cards |

约束：

- **不在前端做 review / runtime parsing。** 所有 `<system-review>` 文本解析、runtime span 分类、timeline event 组装都收口在 `console/server` 的 read-model builder。
- **不新增 runtime-level side store。** 所有新 UI 视图都只能由已有 `RunLog` entries 或 `Trace.spans` 派生。

### 2. Runtime Span Coverage Must Be Complete Enough For Trace Replay

当前 `trace_writer` 只把一部分 runtime facts 映射为 runtime spans：

- `CompactionApplied` -> `compaction`
- `StepBackApplied` -> `step_back`
- `TerminationDecided` -> `termination`

这不足以支撑“完整 loop 回顾”。Trace writer 需要补齐：

| RunLog fact | Trace runtime span name | Status | Key attributes |
| --- | --- | --- | --- |
| `CompactionApplied` | `compaction` | `ok` | `sequence`, `start_sequence`, `end_sequence`, `before_token_estimate`, `after_token_estimate`, `message_count`, `summary`, `transcript_path` |
| `CompactionFailed` | `compaction_failed` | `error` | `sequence`, `error`, `attempt`, `max_attempts`, `terminal` |
| `StepBackApplied` | `step_back` | `ok` | `sequence`, `affected_count`, `checkpoint_seq`, `experience` |
| `RunRolledBack` | `rollback` | `ok` | `sequence`, `start_sequence`, `end_sequence`, `reason` |
| `TerminationDecided` | `termination` | `ok` | `sequence`, `termination_reason`, `phase`, `source` |
| `HookFailed` | `hook_failed` | `error` | `sequence`, `phase`, `handler_name`, `critical`, `error` |

说明：

- `compaction_failed` 和 `rollback` 必须进入 trace；它们已经是 runtime 级一等事实，不应该在 trace 中缺席。
- `hook_failed` 不是 runtime decision，但它是 runtime loop 中的重要异常事实，应作为 timeline 中的“runtime failure”事件可见。它不需要进入 Session 页最上层 decision summary，但 Trace Detail 需要可回放。
- runtime span 必须补齐 `sequence`，否则无法稳定和 step / run / review timeline 对齐。

### 3. Console Server Adds Read Models, Not New Runtime Models

Console Server 新增两类 read model：

#### 3.1 Session Runtime Decisions Read Model

Session 页不再从 `RuntimeDecisionState.latest_*` 只拼出每种 latest 一条，而是提供 **最近 N 条 runtime decision events**。

新的来源：

- 直接从 `RunLogStorage.list_entries(session_id=..., run_id=..., agent_id=...)` 读取对应 session 的 entries
- 过滤出：
  - `TerminationDecided`
  - `CompactionApplied`
  - `CompactionFailed`
  - `StepBackApplied`
  - `RunRolledBack`
- 按 `sequence DESC` 截断最近 N 条

新的 `SessionObservabilityResponse.decision_events` 仍然复用 `RuntimeDecisionResponse` 形状，但语义从“每种 latest 一条”改成“最近 runtime decision 事件列表”。

`details` 字段必须带完整结构化 payload，至少包含：

- `compaction`
  - `start_sequence`
  - `end_sequence`
  - `before_token_estimate`
  - `after_token_estimate`
  - `message_count`
  - `summary`
  - `transcript_path`
- `compaction_failed`
  - `error`
  - `attempt`
  - `max_attempts`
  - `terminal`
- `step_back`
  - `affected_count`
  - `checkpoint_seq`
  - `experience`
- `rollback`
  - `start_sequence`
  - `end_sequence`
  - `reason`
- `termination`
  - `reason`
  - `phase`
  - `source`

#### 3.2 Trace Observability Read Model

`TraceResponse` 需要扩展为更适合 Trace Detail UI 的 read model，而不是只返回原始 spans：

```python
class TraceResponse(TraceBase):
    end_time: str | None = None
    root_span_id: str | None = None
    max_depth: int = 0
    spans: list[SpanResponse] = Field(default_factory=list)
    runtime_decisions: list[RuntimeDecisionResponse] = Field(default_factory=list)
    timeline_events: list[TraceTimelineEventResponse] = Field(default_factory=list)
```

其中：

```python
class TraceTimelineEventResponse(BaseModel):
    kind: str
    timestamp: str | None = None
    sequence: int | None = None
    run_id: str | None = None
    agent_id: str | None = None
    span_id: str | None = None
    step_id: str | None = None
    title: str
    summary: str
    status: str = "ok"
    details: dict[str, Any] = Field(default_factory=dict)
```

`timeline_events` 只服务 Trace Detail UI，不是新的 SDK contract。

约束：

- `runtime_decisions` 只包含真正的 decision-bearing runtime spans：
  - `compaction`
  - `compaction_failed`
  - `step_back`
  - `rollback`
  - `termination`
- `hook_failed` 只进入 `timeline_events`，不进入 `runtime_decisions`

### 4. Trace Timeline Event Taxonomy

Trace 页要回顾的是“当前 trace 内的 loop”，因此 timeline 事件按用户理解的粒度组织，而不是机械复刻 span 类型。

#### 4.1 Event Kinds

| kind | 来源 | 展示目的 |
| --- | --- | --- |
| `run_started` | agent span / root span | 看到根 run 和 nested run 的开始 |
| `run_finished` | agent span completion | 看到 run 正常结束 |
| `run_failed` | agent span completion | 看到 run 失败 |
| `llm_call` | llm span | 看到每轮模型调用 |
| `tool_call` | tool span | 看到关键工具执行 |
| `review_checkpoint` | 普通 tool span output 中的 `<system-review>` | 看到系统何时要求 agent 做目标对齐检查 |
| `review_result` | `review_trajectory` tool span | 看到 agent 判断 aligned / misaligned |
| `milestone_update` | `declare_milestones` tool span | 看到 milestones 如何被声明/更新 |
| `runtime_decision` | runtime span | 看到 compaction / step-back / rollback / termination / compaction_failed |
| `hook_failed` | runtime span `hook_failed` | 看到 hook 路径失败 |

#### 4.2 How Review Timeline Is Derived

不新增 review 专用 run-log family，review lifecycle 从已有 tool spans 派生：

1. **review checkpoint**
   - 识别条件：普通 tool span 的 output 中包含 `<system-review>...</system-review>`
   - 解析内容：
     - `trigger_reason`
     - `steps_since_last_review`
     - `active_milestone`
     - `hook_advice`（如果存在）
     - `raw_notice`

2. **review result**
   - 识别条件：tool span `tool_name == "review_trajectory"`
   - 展示内容：
     - `aligned`: `true` / `false` / `unknown`
     - `raw_output`
     - `tool_call_id`
   - `aligned` 优先从文本中提取 `aligned=true|false`；无法可靠提取时回退成 `unknown`
   - `experience` 不单独从 `review_trajectory` 文本强解析；以 `step_back` runtime decision 中的 `experience` 作为 canonical summary

3. **milestone update**
   - 识别条件：tool span `tool_name == "declare_milestones"`
   - 展示内容优先来自 `tool_details.input_args["milestones"]`
   - `details` 保留完整 milestone list，便于展开查看

#### 4.3 Runtime Decision Timeline Rules

`runtime_decision` timeline 直接复用 `runtime_decisions` 中的 summary 规则，确保：

- Session 页 runtime decision 列表
- Trace 页 runtime decision 卡片
- Trace 页 timeline 中对应事件

三处的 `title / summary / details` 一致，不允许各说各话。

#### 4.4 Timeline Ordering

`timeline_events` 必须使用稳定排序规则：

1. `timestamp ASC`
2. 如果 timestamp 相同，按 `sequence ASC`
3. 如果 sequence 仍然相同，按事件种类固定顺序：
   - `run_started`
   - `llm_call`
   - `tool_call`
   - `review_checkpoint`
   - `review_result`
   - `milestone_update`
   - `runtime_decision`
   - `hook_failed`
   - `run_finished`
   - `run_failed`

这样可以避免同一 trace 在不同页面或不同浏览器渲染时出现事件顺序漂移。

### 5. Trace Detail Page Layout

Trace Detail 改成 4 层结构：

1. **Trace Summary**
   - 保留现有 token / cost / llm / tool 指标卡
   - 新增：
     - runtime decision count
     - review event count
   - 这些计数可以从 `runtime_decisions` / `timeline_events` 派生，不需要额外 API 字段

2. **Runtime Decisions**
   - 这是新的一级区块，放在 waterfall 之前
   - 列出当前 trace 内全部 runtime decisions，按时间倒序
   - 每条默认显示：
     - icon
     - kind
     - summary
     - seq
     - run_id / agent_id
     - time
   - 展开后显示结构化 details

3. **Loop Timeline**
   - 这是新的一级区块，按时间顺序展示 `timeline_events`
   - 最外层只显示：
     - title
     - summary
     - status badge
     - run / agent / seq / timestamp
   - 展开后才显示：
     - parsed review notice
     - milestones payload
     - raw tool output / input args
     - runtime decision details
   - `review_checkpoint`、`review_result`、`milestone_update` 都放在这里，而不是单独再建一个“Review Internals”页面

4. **Span Waterfall**
   - 保留现有 waterfall
   - 继续作为“原始 span 诊断视图”
   - 但不再承担“用户理解整个 loop”的主要职责

### 6. Session Detail Observability Layout

Session 页右侧 `Runtime Decisions` 区块改为：

- 展示最近 N 条 runtime decision events，而不是 latest-per-kind
- 默认按 `created_at DESC, sequence DESC` 排序
- 每条卡片可展开
- `compaction_failed` 必须进入列表

同时，Session 页的 `Steps` 列表语义要与 prompt-visible history 对齐：

- 默认 `list_session_steps()` 应改为 `include_hidden_from_context=False`
- `review_trajectory` 的 temporary review metadata 不再默认出现在普通步骤列表里
- review / milestone / step-back 细节由 `Observability` 和 `Trace Detail` 承载

说明：

- 这是一次有意的语义收敛：`Steps` 展示“用户视角 / prompt-visible”的历史，`Observability` 展示“内部 runtime 回顾”。
- 这比继续把 hidden review metadata 混在 step list 里更清晰。

### 7. Step-Back Display Semantics Must Be Fixed

`step_back` 的 canonical display contract 必须是：

| 字段 | 含义 |
| --- | --- |
| committed step `content` | 原始 tool result |
| `StepCondensedContentUpdated.condensed_content` | step-back 之后 prompt-visible 的 condensed content，例如 `[EXPERIENCE] ...` |
| `StepView.condensed_content` | UI 默认展示内容 |

因此需要修改 `execute_step_back()`：

- storage `append_step_condensed_content(...)` 写入 `[EXPERIENCE] ...`
- 内存 message list 中 tool message 的 `content` 也更新为同样的 `[EXPERIENCE] ...`
- committed step 原始 `content` 保持不变

修正后，Session / Chat / Trace 的展示 contract 统一变成：

- 默认看 condensed result
- 展开看 original result

这也是 Trace Detail 能正确展示“step-back 之后 runtime 实际可见内容”的前提。

### 8. Parsing Strategy Must Live In One Server-Side Module

为了避免 observability 规则散落在前端，所有解析逻辑集中在 Console Server 的单一模块，例如：

- `console/server/services/runtime/trace_observability.py`

它负责：

- 将 trace spans 转成 `runtime_decisions`
- 将 trace spans 转成 `timeline_events`
- 从 tool output 中提取 `<system-review>`
- 从 `review_trajectory` 输出文本中提取 `aligned` 状态
- 从 `declare_milestones` input args 中提取 milestones
- 构建统一的 `summary / details`

前端只渲染 `TraceResponse.runtime_decisions` 和 `TraceResponse.timeline_events`，不再直接正则解析 span output。

### 9. Summary Rules

所有卡片和 timeline 的 summary 必须遵守统一规则：

- `compaction`
  - `seq {start}-{end}, {before} -> {after} tokens`
- `compaction_failed`
  - `attempt {attempt}/{max_attempts}: {error}`
- `step_back`
  - `{affected_count} results condensed after checkpoint seq {checkpoint_seq}`
- `rollback`
  - `seq {start}-{end} hidden`
- `termination`
  - `{termination_reason} via {source}`
- `review_checkpoint`
  - `triggered by {trigger_reason} after {steps_since_last_review} steps`
- `review_result`
  - `trajectory aligned` / `trajectory misaligned`
- `milestone_update`
  - `{count} milestones declared/updated`
- `hook_failed`
  - `{phase}: {handler_name} failed`

这样用户在 Session 页、Trace 页、waterfall 明细之间切换时，不会看到同一事实被改写成不同措辞。

## Testing

这次改动至少要覆盖以下层面：

1. **Trace collector / trace writer**
   - `CompactionFailed` 会生成 `compaction_failed` runtime span
   - `RunRolledBack` 会生成 `rollback` runtime span
   - `HookFailed` 会生成 `hook_failed` runtime span
   - runtime span attributes 带 `sequence`

2. **Console server read model**
   - Session observability 返回“最近 N 条 decision events”，不是 latest-per-kind
   - `compaction_failed` 会出现在 session decision 列表
   - Trace response 会返回 `runtime_decisions` 和 `timeline_events`
   - `<system-review>` 能被解析成 `review_checkpoint`
   - `review_trajectory` span 会生成 `review_result`
   - `declare_milestones` span 会生成 `milestone_update`

3. **Step-back display semantics**
   - `StepCondensedContentUpdated.condensed_content` 存 `[EXPERIENCE] ...`
   - Session step card / chat message 默认显示 condensed 内容
   - “View original result” 展开后显示原始 tool output

4. **Console web**
   - Trace Detail 渲染 Runtime Decisions 区块
   - Trace Detail 渲染 Loop Timeline 区块
   - Session Observability 能展开 decision details
   - Session steps 默认不再显示 hidden review metadata

## Tradeoffs

### 1. Review Checkpoint Details Are Still Parsed From Text

`review_checkpoint` 不是新的 first-class run-log fact，而是从 `<system-review>` 文本中解析得到。

这意味着：

- 解析规则必须集中在单点 helper 中
- 解析失败时也要保留 `raw_notice`
- timeline event 允许 `trigger_reason="unknown"` 之类的保守回退

这是有意 tradeoff：

- 保持当前 runtime persistence 边界不变
- 先把已有 review 信息可靠地暴露出来
- 避免为 observability 单独引入一套新的 runtime log family

### 2. Session Steps And Observability Will Have Different Default Semantics

修正后：

- `Steps` 更偏 prompt-visible / conversational history
- `Observability` 更偏 runtime replay / debugging history

这不是重复，而是刻意分层。当前把 hidden review metadata 混进 Steps，才是语义混乱的来源。

## Risks

1. **`condensed_content` 语义修正会触及多个 UI 面。**
   Session step card、chat message、trace-related step 展示都依赖这个字段，必须一起更新测试，否则很容易一处修正、一处继续反着显示。

2. **Trace response 扩展后，前后端 summary 规则必须共享。**
   如果 Session 页和 Trace 页各自拼 summary，后续很快会再次漂移。

3. **Review 文本解析必须 tolerant。**
   `<system-review>` notice 未来可能小幅调整措辞，因此 parser 需要按行 key/value 提取，不应依赖完整文本模板逐字匹配。

## Result

完成后，Console Web 的 observability 语义收敛成两句话：

- `Session Detail` 告诉你：**这个 session 最近发生了哪些关键 runtime decisions**
- `Trace Detail` 告诉你：**这个 trace 内的 agent loop 是如何一步步推进、何时 review、何时偏离、系统如何修正、最后如何结束**

整个设计仍然坚持同一个边界：

- `RunLog` 是 canonical fact source
- `Trace` 是 replayable trace view
- `Console Server` 负责把事实组装成可读的 observability read model
- `Console Web` 只负责把这些 read models 渲染成可回顾的页面
