# Console Mainline And Debug Observability

> 为 Console Web 重新设计 `Session Detail` 和 `Trace Detail` 的信息架构。
> 目标不是继续堆更多 `JsonDisclosure`，而是在保留完整调试能力的前提下，把 agent loop 的主线、milestone、review cycle、runtime decisions 和 LLM/tool 细节拆成更适合阅读的结构化视图。

## Problem

当前 Console observability 的主要问题不在“数据缺失”，而在“数据可读性”：

1. `Trace Detail` 仍然以 span dump 和大块 JSON 为主。
   - `LLM Details` 直接显示原始 `messages/tools/response_content` JSON
   - `Tool Details` 和 `timeline event details` 也主要是原始 payload
   - 用户可以看到很多事实，但很难快速回答“这一轮到底发生了什么”

2. milestone / review 信息是碎片事件，不是连续流程。
   - `declare_milestones`
   - `<system-review>` checkpoint
   - `review_trajectory`
   - `step_back` / `rollback`

   这些事实现在分散在 trace timeline、tool spans、runtime decisions、session steps 中，没有形成稳定的“review cycle”视图。

3. `Session Detail` 没有全局 milestone board。
   - 当前页面只有 summary、observability、runs、steps
   - 用户无法直接看到“当前任务主路线是什么”“现在推进到哪个 milestone”“最近一次 review 结论是什么”

4. 对话页和历史页对 review/tool 事件的权重过高。
   - 大量 tool result 和 review 记录占据主视图
   - 被 step-back 压缩后的结果虽然有 `condensed_content`，但页面仍然缺少明确的“已压缩历史事件”语义
   - live transcript 和 history transcript 的信息组织方式也不一致

5. 当前前端大量依赖“summary chips + raw JSON fallback”。
   这种模式对临时调试有用，但无法支撑长期稳定的信息架构，也会迫使前端继续解析脆弱的原始 payload。

## Goals

这次改动的目标如下：

1. `Session Detail` 和 `Trace Detail` 都提供 `Mainline / Debug` 双视图。
   - `Mainline` 优先帮助用户理解任务推进主线
   - `Debug` 保留完整 runtime 细节与原始 payload 下钻能力

2. `Session Detail` 顶部提供全局 milestone board。
   - 显示当前 root run 的 milestone 列表、active milestone、最新 checkpoint、最近一次 review 结果、pending review reason
   - 展示形态类似 TODO / 路线板，而不是事件表格

3. `Trace Detail` 提供可读的单次 run narrative。
   - 明确展示 milestone 变化、review cycle、step-back、rollback、termination
   - 不再要求用户从分散的 event JSON 中自行拼装流程

4. `LLM Call`、`Tool Call`、`Runtime Decision` 细节必须结构化。
   - 原始 JSON 仍保留，但退到次级展开层
   - 最外层先展示高信号字段，如 model、tokens、latency、finish reason、args summary、output preview、impact summary

5. 对话页默认保留 review/tool 事件，但明显降权。
   - user / assistant 仍是主消息
   - tool / review / compressed history 变成较弱的事件卡
   - 支持切换只看主对话、主对话加关键事件、全部事件

6. live transcript 和 history transcript 的语义必须一致。
   - 不能出现“运行中很吵，刷新后干净”的割裂体验
   - step-back / hidden steps 的语义应在流式过程中也能被消费

## Non-Goals

这次不做下面这些事：

1. 不把 `review_trajectory` 改成独立的二次 LLM 调用。
   它仍然是主模型在读到 `<system-review>` 后发起的普通 tool call。

2. 不在 SDK runtime 中立即引入新的 canonical review-cycle facts。
   本轮优先在 Console read model 层聚合已有事实，只有当现有事实无法稳定配对时，才考虑未来补新的 SDK first-class entries。

3. 不把 `Session Detail` 的 milestone board 做成整个 session 历史的大杂烩。
   顶部 board 只跟随“最新 root run”的状态；历史 milestone 演进通过 traces / runs 回看。

4. 不删除现有 `spans`、`steps`、`runtime_decisions`、raw payload 数据面。
   本轮是增加结构化可读层，不是砍掉调试能力。

5. 不做 schema migration。
   Console 读模型继续基于现有 `RunLog` 和 `Trace.spans` 派生。

## Design Principles

### 1. Mainline And Debug Are Different Products

`Mainline` 和 `Debug` 不是同一个页面换皮，而是两种不同阅读任务：

- `Mainline`
  - 用户关心当前任务怎么推进
  - milestone 是否清晰
  - 最近 review 怎么判断
  - 是否发生 step-back / rollback
  - 当前是否已完成或偏航

- `Debug`
  - 用户关心底层 runtime 如何执行
  - 发了哪些 LLM call
  - 调了哪些 tool
  - runtime decision 的具体 payload 是什么
  - 原始 trace / step / payload 能否追溯

因此两者必须共享事实来源，但不共享信息层级。

### 2. Raw Payload Is A Secondary Layer

原始 JSON 继续保留，但一律退居次级展开层：

- 一级：状态结论与高信号摘要
- 二级：结构化细节
- 三级：raw payload / full JSON

任何页面默认不能再让 `JsonDisclosure` 成为第一阅读入口。

### 3. Console Owns Readability, SDK Owns Truth

这轮的职责边界：

- SDK / `RunLog` / `Trace.spans`：真相源
- Console server：结构化 read model builder
- Console web：Mainline / Debug 信息架构与可视化

前端不应继续直接解析 `<system-review>`、`aligned=true` 或 `declare_milestones` payload 来猜语义。

## Information Architecture

### Session Detail

`Session Detail` 改成双视图：

- `Mainline`
  - `Milestone Board`
  - `Current Focus`
  - `Latest Review`
  - `Conversation`

- `Debug`
  - `Observability`
  - `Runs`
  - `Steps`
  - 原始 runtime decisions / raw payload

#### Mainline Layout

1. `Milestone Board`
   - 顺序展示当前 root run 的 milestones
   - 每项显示：
     - `id`
     - `description`
     - `status`
     - `declared_at_seq`
     - `completed_at_seq`
   - `active` milestone 明显高亮
   - `completed` milestone 显示完成标记
   - `abandoned` milestone 显示降权状态

2. `Current Focus`
   - 展示当前 active milestone
   - 展示最近 checkpoint 对应 milestone
   - 展示是否有 `pending_review_reason`

3. `Latest Review`
   - 展示最近一次 review cycle 的结论：
     - `aligned`
     - `misaligned`
     - `step_back_applied`
     - `affected_count`
     - `experience`

4. `Conversation`
   - user / assistant 作为一等消息
   - tool / review / milestone / compressed history 作为降权事件卡
   - 提供过滤器：
     - `Dialogue`
     - `Dialogue + Key Events`
     - `All Events`

#### Debug Layout

保留当前页面的核心能力：

- summary cards
- observability panel
- runs table
- full steps feed

但 steps 呈现要保持与 Mainline 的语义一致：

- 若 `condensed_content` 存在，默认显示 condensed 内容
- 原始内容继续放在展开区
- `review_trajectory` 不伪装成普通业务 tool result，而应明确标注为 review metadata

### Trace Detail

`Trace Detail` 也改成双视图：

- `Mainline`
  - `Run Narrative`
  - `Review Cycles`
  - `Milestone Changes`
  - `Runtime Decisions`

- `Debug`
  - `Waterfall`
  - `LLM Calls`
  - `Tool Calls`
  - `Runtime Decisions`
  - raw span / payload

#### Mainline Layout

面向“单次 run 复盘”的 narrative 视图，按事件流程阅读：

1. run started
2. milestone declared / updated
3. review checkpoint triggered
4. review result produced
5. optional step-back / rollback
6. compaction / termination
7. run finished / failed

其中 `Review Cycle` 是主对象，不再把 checkpoint、review result、step-back 生硬拆成三个同权 event row。

#### Debug Layout

保留 waterfall，但 span detail 改成结构化展示：

- `LLM Call`
  - model / provider
  - finish reason
  - duration
  - first token latency
  - input / output / total tokens
  - message count
  - tool schema count
  - response tool call count
  - output preview

- `Tool Call`
  - tool name
  - args summary
  - status
  - output summary
  - 如果包含 `<system-review>`，直接显示其结构化解析结果

- `Runtime Decision`
  - kind
  - impact summary
  - related sequence range
  - reason / experience

- `Raw Payload`
  - 统一作为最后一级 disclosure

## Read Models

Console server 新增结构化 read models，而不是让前端直接消费原始 JSON。

### 1. SessionMilestoneBoardRecord

```python
class SessionMilestoneBoardRecord:
    session_id: str
    run_id: str | None
    milestones: list[MilestoneRecord]
    active_milestone_id: str | None
    latest_checkpoint: ReviewCheckpointRecord | None
    latest_review_outcome: ReviewOutcomeRecord | None
    pending_review_reason: str | None
```

用途：

- `Session Detail > Mainline > Milestone Board`
- `Current Focus`
- `Latest Review`

来源：

- 优先基于当前 session 最新 root run 的相关 facts 重建
- milestone 原始来源优先取 `declare_milestones` 结果
- checkpoint / latest review outcome 从 review read model 派生

### 2. ReviewCycleRecord

```python
class ReviewCycleRecord:
    cycle_id: str
    run_id: str
    agent_id: str
    trigger_reason: str
    steps_since_last_review: int | None
    active_milestone: str | None
    hook_advice: str | None
    aligned: bool | None
    experience: str | None
    step_back_applied: bool
    rollback_range: tuple[int, int] | None
    affected_count: int | None
    started_at: datetime | None
    resolved_at: datetime | None
    raw_notice: str | None
```

用途：

- `Trace Detail > Mainline`
- `Session Detail > Latest Review`
- 对话页 review event 摘要

说明：

- `cycle_id` 本轮不是 SDK 真实字段，而是 Console read model 内部构造标识
- 聚合依据优先使用 `run_id + sequence window`

### 3. TraceLlmCallRecord

```python
class TraceLlmCallRecord:
    span_id: str
    run_id: str
    agent_id: str
    model: str | None
    provider: str | None
    finish_reason: str | None
    duration_ms: float | None
    first_token_latency_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    message_count: int
    tool_schema_count: int
    response_tool_call_count: int
    output_preview: str | None
```

用途：

- `Trace Detail > Debug > LLM Calls`
- 替代当前“LLM Details = 大 JSON”

### 4. ConversationEventRecord

```python
class ConversationEventRecord:
    id: str
    session_id: str
    run_id: str | None
    sequence: int | None
    kind: Literal[
        "assistant_message",
        "tool_event",
        "review_event",
        "milestone_event",
        "compressed_history_event",
    ]
    priority: Literal["primary", "secondary", "muted"]
    title: str
    summary: str
    details: dict[str, Any]
```

用途：

- `Session Detail > Mainline > Conversation`

它不是取代 `steps`，而是主线视图消费的轻量事件层。

## API Response Shape

本轮推荐以“保留旧字段，新增结构化字段”的方式扩展现有响应。

### SessionDetailResponse

新增字段：

```python
class SessionDetailResponse(BaseModel):
    ...
    milestone_board: SessionMilestoneBoardResponse | None = None
    review_cycles: list[ReviewCycleResponse] = Field(default_factory=list)
    conversation_events: list[ConversationEventResponse] = Field(default_factory=list)
```

约束：

- `milestone_board` 只代表当前 session 最新 root run 的主线状态
- `review_cycles` 提供最近若干条 cycle，供 `Latest Review` 和 debug drill-down 复用
- `conversation_events` 只服务 `Mainline` conversation，不替代现有 `steps`

### TraceResponse

新增字段：

```python
class TraceResponse(BaseModel):
    ...
    mainline_events: list[TraceMainlineEventResponse] = Field(default_factory=list)
    review_cycles: list[ReviewCycleResponse] = Field(default_factory=list)
    llm_calls: list[TraceLlmCallResponse] = Field(default_factory=list)
```

约束：

- `mainline_events` 是 narrative 视图使用的主线事件，不替代现有 `timeline_events`
- `review_cycles` 是从当前 trace 内已有 facts 聚合出的阅读对象
- `llm_calls` 是 `LLM Details` 的结构化视图，不替代原始 `spans`

## Data Sources And Derivation Rules

### Trace-Side Structured Views

Trace 页结构化视图继续从 `Trace.spans` 构建，因为单次 run 的关联关系已经在 trace 中。

#### Review Checkpoint

识别条件：

- tool span output 中包含 `<system-review>...</system-review>`

解析字段：

- `trigger_reason`
- `steps_since_last_review`
- `active_milestone`
- `hook_advice`
- `raw_notice`

#### Review Result

识别条件：

- tool span `tool_name == "review_trajectory"`

解析字段：

- `tool_call_id`
- `aligned`
- `raw_output`

`aligned` 优先从 tool output 结构化内容提取；若只能拿到文本，则从文本模式回退提取；仍然失败则置为 `None`。

#### Milestone Update

识别条件：

- tool span `tool_name == "declare_milestones"`

展示字段优先取：

- `tool_details.input_args["milestones"]`

若输入缺失，允许回退到输出中的 `milestones`。

### Session-Side Structured Views

Session 页结构化视图优先从 `RunLog` / `StepView` 查询层构建，因为它需要跨 traces / runs 聚合。

约束：

- 顶部 `Milestone Board` 只跟随 session 最新 root run
- 历史 traces 和历史 runs 继续通过 Debug 侧查询

## Review Semantics

### `review_trajectory` Keeps Current Runtime Semantics

本轮不改变 `review_trajectory` 的运行机制：

1. 系统在某条 tool result 中注入 `<system-review>`
2. 主模型在下一轮正常 LLM 调用中读到这个 notice
3. 主模型决定是否调用 `review_trajectory`
4. `review_trajectory` tool 本身只验证参数并返回结果
5. runtime 根据结果决定：
   - metadata only
   - step-back

因此：

- 它不是独立的第二次模型请求
- 但它会增加主模型本轮的 token 消耗与 latency

这个语义需要在 Console Debug 视图中被说清楚，而不是继续让用户误以为这是“系统单独做了一次判断”。

### Review Trigger Sources

Debug 视图需要明确展示 review 的触发来源：

- `step_interval`
- `consecutive_errors`
- `milestone_switch`

默认配置展示只读：

- `enable_goal_directed_review`
- `review_step_interval`
- `review_on_error`

## Live Transcript Consistency

当前历史查询和 live stream 的语义存在割裂风险：

- history steps 查询会隐藏 `hidden_from_context` 的步骤
- live stream 仍然可能先把这些 tool/review steps 展示出来

因此本轮需要补一个显式的流式压缩/隐藏事件语义。

候选方案：

1. 复用已有 `messages_rebuilt` 事件，并在前端消费它
2. 新增更直接的 stream event，例如：
   - `context_steps_hidden`
   - `transcript_compacted`

推荐做法：

- 对 SDK 现有 stream 语义做最小扩展，显式传递“哪些 step 现在应被降权/折叠”
- 前端在 live transcript 中原地更新对应消息，而不是等页面刷新后才变干净

这一步是为了保证：

- `Mainline` conversation 在运行中也能保持清晰
- `Debug` conversation 仍然可追溯原始过程

## UI Component Strategy

### Session Components

建议新增或重构以下组件：

- `MilestoneBoard`
- `CurrentFocusCard`
- `LatestReviewCard`
- `ConversationFilterBar`
- `ConversationEventList`
- `ConversationEventCard`

### Trace Components

- `TraceViewModeToggle`
- `TraceRunNarrative`
- `TraceReviewCycleCard`
- `TraceMilestoneTimeline`
- `TraceLlmCallCard`
- `TraceToolCallCard`
- `TraceRuntimeDecisionCard`

### Shared Rule

所有组件都必须遵守：

- 一级摘要优先
- 二级结构化明细次之
- 原始 JSON 最后展开

不能再继续用“所有对象统一渲染成 JsonDisclosure”的模式。

## Risks

### 1. Review Cycle Pairing Ambiguity

当前没有显式 `review_cycle_id`。

风险：

- 嵌套 run
- 高频 tool calls
- 未来更复杂的 review 触发节奏

可能导致 checkpoint / result / step-back 的聚合歧义。

本轮应对：

- 先按 `run_id + sequence window` 聚合
- 为未配对情况提供降级展示
- 若测试证明现有事实不足，再在未来将 `cycle_id` 下沉为 SDK fact

### 2. Milestone Lifecycle Is Not Fully System-Driven

当前 milestone 更多仍由 agent 通过 `declare_milestones` 主动更新，而不是完整自动状态机。

因此 `Milestone Board` 这轮应被定义为：

- 当前任务声明出来的路线板
- 加上 review/checkpoint 派生出来的状态补充

而不是“系统完全可信的自动任务计划器”。

### 3. Mainline / Debug Drift

如果前端继续自行 parse raw JSON，或者部分页面直接绕过结构化 read model，会导致：

- `Session Mainline`
- `Session Debug`
- `Trace Mainline`
- `Trace Debug`

四个地方说法不一致。

本轮必须坚持：

- 结构化字段由 server 统一生成
- web 只负责展示，不重新推理语义

## Testing

### 1. Console Server Read Model Tests

覆盖：

- `LLM details -> TraceLlmCallRecord`
- `review checkpoint + review result + step_back/rollback -> ReviewCycleRecord`
- `declare_milestones -> SessionMilestoneBoardRecord`
- 缺字段、异常字段、未配对 review cycle 的兜底行为

### 2. API Tests

覆盖：

- `SessionDetailResponse` 新增 `milestone_board` / `review_cycles` / `conversation_events`
- `TraceResponse` 新增 `mainline_events` / `review_cycles` / `llm_calls`
- 序列化后的时间字段、结构化 details 与排序稳定

### 3. Web Component Tests

覆盖：

- `Mainline / Debug` 视图切换
- milestone board 状态展示
- review cycle 卡片摘要与展开
- LLM call 结构化细节
- Conversation filter 行为
- 压缩后的 history event 降权展示

### 4. Live Stream Tests

覆盖：

- step-back / hide 事件到达后，live transcript 能原地降权或折叠历史 tool/review 项
- 不会出现刷新前后语义不一致

## Rollout Plan

推荐实现顺序：

1. 先补 Console server structured read models
2. 先做 `Trace Detail` 双视图
3. 再做 `Session Detail` milestone board + mainline conversation
4. 最后补 live transcript 一致性修正

原因：

- `Trace` 的单次 run 数据面更完整，适合作为新信息架构的试验田
- `Session` 和 live transcript 后续复用同一套 read model 语义，返工更少

## Acceptance Criteria

完成后应满足：

1. 用户在 `Session Detail` 顶部无需展开任何 JSON，就能看到当前任务 milestone 主线和最近一次 review 结论。
2. 用户在 `Trace Detail` 中无需手工比对多个 event JSON，就能读出完整 review cycle 与 runtime decision 流程。
3. `LLM Call` 不再以大块 JSON 作为默认 detail 表达。
4. `review_trajectory`、step-back 后的 tool history、milestone 更新在主对话中默认保留，但明显降权。
5. `Mainline` 与 `Debug` 两条视图共享同一套事实来源，不出现语义冲突。
