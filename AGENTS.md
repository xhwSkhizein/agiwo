# AGENTS.md

> Agiwo 开发指南。供 AI 编码助手和开发者快速理解项目结构、约定和决策。

## Project Map

```
agiwo/
├── agent/                    # Agent 核心
│   ├── agent.py              # Agent 类：唯一公开入口 (run / run_stream)
│   ├── execution_context.py  # ExecutionContext + SessionSequenceCounter
│   ├── hooks.py              # AgentHooks：生命周期钩子 (dataclass, 全部可选)
│   ├── options.py            # AgentOptions + RunStepStorageConfig + TraceStorageConfig
│   ├── schema.py             # 核心数据模型: Run, StepRecord, StreamEvent, RunOutput 等
│   ├── stream_channel.py     # StreamChannel：asyncio.Queue 事件流通道
│   ├── inner/                # 内部实现 (不对外暴露)
│   │   ├── executor.py       # AgentExecutor：LLM + Tool 执行循环
│   │   ├── event_emitter.py  # EventEmitter：纯事件发射 (不涉及 Storage)
│   │   ├── llm_handler.py    # LLM 流式响应处理
│   │   ├── message_assembler.py  # 消息组装 (system prompt + history + memory)
│   │   ├── run_state.py      # 运行状态追踪 (序列号委托给 SessionSequenceCounter)
│   │   ├── storage_sink.py   # StorageSink：事件流中间件，消费事件完成持久化
│   │   ├── step_builder.py   # StepRecord 构建
│   │   ├── summarizer.py     # 终止摘要生成
│   │   └── system_prompt_builder.py  # System Prompt 动态构建
│   └── storage/              # Run/Step 持久化
│       ├── base.py           # RunStepStorage ABC + InMemoryRunStepStorage
│       ├── factory.py        # StorageFactory：根据配置创建存储实例
│       ├── sqlite.py         # SQLite 实现
│       └── mongo.py          # MongoDB 实现
├── llm/                      # LLM Provider 抽象层
│   ├── base.py               # Model ABC + StreamChunk
│   ├── openai.py             # OpenAIModel (其他 OpenAI 兼容 Provider 的基类)
│   ├── anthropic.py          # AnthropicModel (独立实现，非 OpenAI 兼容)
│   ├── deepseek.py           # DeepseekModel (继承 OpenAIModel)
│   ├── nvidia.py             # NvidiaModel (继承 OpenAIModel)
│   └── helper.py             # 工具函数: JSON 解析, usage 标准化
├── tool/                     # Tool 系统
│   ├── base.py               # BaseTool ABC + ToolResult + ToolDefinition
│   ├── agent_tool.py         # AgentTool + as_tool()：Agent 作为 Tool 嵌套
│   ├── executor.py           # ToolExecutor：批量并行执行 + 缓存
│   ├── cache.py              # ToolResultCache
│   ├── builtin/              # 内置工具 (装饰器自动注册)
│   │   ├── registry.py       # @builtin_tool + @default_enable 装饰器
│   │   ├── calculator.py     # 计算器
│   │   ├── current_time.py   # 当前时间
│   │   └── http_request.py   # HTTP 请求
│   └── permission/           # 工具权限管理
├── observability/            # 可观测性
│   ├── base.py               # BaseTraceStorage ABC
│   ├── trace.py              # Trace + Span 模型 (兼容 OpenTelemetry)
│   ├── collector.py          # TraceCollector：从 StreamEvent 流构建 Trace
│   ├── otlp_exporter.py      # OTLP 导出器
│   ├── store.py              # MongoDB TraceStorage
│   └── sqlite_store.py       # SQLite TraceStorage
├── scheduler/                # Agent 调度系统
│   ├── models.py             # AgentState, WakeCondition, SchedulerOutput, OutputChannelState, SchedulerConfig
│   ├── executor.py           # SchedulerExecutor：Agent 执行 + 输出事件发射
│   ├── store.py              # AgentStateStorage ABC + InMemory + SQLite 实现
│   ├── guard.py              # TaskGuard：集中式护栏 (spawn/wake 限制, 超时检测)
│   ├── tools.py              # SpawnAgentTool, SleepAndWaitTool, QuerySpawnedAgentTool
│   └── scheduler.py          # Scheduler：编排层入口 (run / submit / submit_task / subscribe / shutdown / cancel)
├── skill/                    # Skill 系统 (可选)
│   ├── manager.py            # SkillManager：发现、加载、热重载
│   ├── registry.py           # SkillRegistry + SkillMetadata
│   ├── loader.py             # SkillLoader：SKILL.md 解析
│   └── skill_tool.py         # SkillTool：将 Skill 暴露为 Tool
├── config/
│   └── settings.py           # AgiwoSettings (pydantic-settings, env prefix: AGIWO_)
└── utils/
    ├── abort_signal.py       # AbortSignal：优雅取消
    ├── logging.py            # structlog 日志配置
    ├── retry.py              # 异步重试装饰器
    └── tojson.py             # JSON 序列化工具
```

## Core Components

### Agent (唯一入口)

`Agent` 是具体类 (非 ABC)，负责配置持有和执行生命周期。两个公开方法：

- `run(user_input, ...)` -> `RunOutput` -- 阻塞执行
- `run_stream(user_input, ...)` -> `AsyncIterator[StreamEvent]` -- 流式执行

`Agent` 不做 LLM 调用和 Tool 执行，委托给 `AgentExecutor`。

### AgentExecutor (内部循环)

`agent/inner/executor.py`。核心 `_run_loop`：LLM 调用 -> 检查 tool_calls -> 并行执行 Tools -> 追加消息 -> 循环直到完成或触发限制。

核心执行只依赖 `EventEmitter`（纯事件发射），不涉及 Storage。序列号通过 `RunState.next_sequence()` 分配（内存计数器，`asyncio.Lock` 保证并发安全）。

### Model (LLM 抽象)

`@dataclass` + ABC。子类必须实现 `arun_stream(messages, tools) -> AsyncIterator[StreamChunk]`。

- OpenAI 兼容 API：继承 `OpenAIModel`，仅覆写 `_resolve_api_key` / `_resolve_base_url`
- 非兼容 API (如 Anthropic)：直接继承 `Model`，完整实现 `arun_stream`

### BaseTool (Tool 抽象)

ABC，子类实现 5 个方法：`get_name`, `get_description`, `get_parameters`, `is_concurrency_safe`, `execute`。

`execute` 直接返回 `ToolResult` (包含 `content` 给 LLM + `content_for_user` 给前端)。

### AgentTool (Agent 嵌套)

通过 `AgentTool` 或 `as_tool()` 将 Agent 包装为 Tool，实现 **Main Agent + Agent as Tools** 模式。内置深度限制和循环引用检测。

### AgentHooks (生命周期钩子)

纯 dataclass，所有字段可选。不需要子类化，直接传入回调函数。

钩子点：`on_before_run`, `on_after_run`, `on_before_tool_call`, `on_after_tool_call`, `on_before_llm_call`, `on_after_llm_call`, `on_step`, `on_event`, `on_memory_write`, `on_memory_retrieve`。

### Storage

- **RunStepStorage**：Run/Step 持久化接口。通过 `AgentOptions.run_step_storage` 配置
- **BaseTraceStorage**：Trace 持久化接口。通过 `AgentOptions.trace_storage` 配置
- 实现：InMemory / SQLite / MongoDB
- 存储通过配置注入，`StorageFactory` 创建实例，不依赖全局状态

### Scheduler (编排层)

`scheduler/scheduler.py`。Agent 之上的编排层，管理 Agent 的 spawn/sleep/wake/completion 生命周期。

- Agent **不感知** Scheduler（依赖方向：`scheduler/ → agent/`，单向）
- Scheduler 从外部注入调度工具（`spawn_agent`, `sleep_and_wait`, `query_spawned_agent`）
- **三种 API 模式**：
  - `run()` 阻塞等待
  - `submit()` + `wait_for()` 非阻塞，获取最终结果
  - `submit_and_subscribe()` / `submit_task_and_subscribe()` 输出流，逐条接收 Agent 文本产出
- `agent_id` 作为 `AgentState` 主键，同一 agent_id 不能并发运行
- 工具通过 `context.agent_id`（已有字段）获取当前 Agent 身份，无需额外 metadata 注入
- `ToolResult.termination_reason` 是通用的执行终止机制，Executor 在 `_execute_tools` 后检查
- **Persistent Root Agent**：`submit(persistent=True)` 使 root agent 完成后保持 SLEEPING，通过 `submit_task()` 接收新任务
- **TaskGuard**：集中式护栏，所有 spawn/wake 限制检查收敛于此（`check_spawn`, `check_wake`, `find_timed_out`）
- **WakeCondition 类型**：`WAITSET`（等子 Agent）、`TIMER`（一次性延迟）、`PERIODIC`（周期唤醒）、`TASK_SUBMITTED`（新任务提交）
- **WaitSet 语义**：`wait_for` 指定等待的子 Agent ID 列表，`wait_mode` 控制 ALL/ANY，`completed_ids` 追踪已完成的
- **超时保护**：`timeout_at` 防止永久 sleep，超时后 Scheduler 唤醒 Agent 并注入已有结果让其生成摘要
- **递归操作**：`cancel()` 硬取消整棵树，`shutdown()` 优雅关闭（让 Agent 生成最终报告）
- **自动结果注入**：WAITSET 唤醒时 `_build_wake_message()` 自动收集子 Agent 结果注入 wake message
- **Output Streaming**：`SchedulerExecutor._emit_output()` 将 Agent 文本产出推送到 `OutputChannelState` 队列，消费者通过 `subscribe` API 逐条接收 `SchedulerOutput`。`include_child_outputs` 控制是否推送子 Agent 输出（默认 True）

### Event Pipeline (事件管道)

核心执行通过 `EventEmitter` 发射事件到 `StreamChannel`，所有副作用在事件流下游处理：

```
Executor → EventEmitter → Channel → StorageSink(持久化) → TraceCollector(Trace) → User
```

- **StorageSink**：消费事件完成 Run/Step 持久化，支持嵌套 Agent 多 Run 追踪
- **TraceCollector**：消费事件构建 Trace + Span，兼容 OpenTelemetry，支持 OTLP 导出

## Code Style & Conventions

- **Python 3.11+**，不使用 `from __future__ import annotations`
- **所有 import 放在文件顶部**，禁止局部导入 / 延迟导入
- 类型注解：所有公开方法和核心数据结构必须有类型注解
- 命名：PascalCase 类名，snake_case 函数/变量，按意图命名 (`*_handler`, `*_builder`, `*_store`, `*_hook`)
- `is not None` 做哨兵检查；truthy 检查仅在明确意图时使用
- 避免可变默认参数：用 `None` + `__init__` 内初始化
- async 代码保持显式，不在隐式 helper 中藏 await
- 偏好小函数而非深层嵌套的大方法
- SOLID + KISS，单一职责，不做不必要的抽象
- **禁止向后兼容** (除非明确要求)，及时删除遗留代码
- **循环依赖**：遇到时必须重构组件耦合关系，禁止延迟导入等绕行手段
- 不要使用 `rm` 删除文件，用 `mv` 移动到 `trash/` 目录

## Configuration Governance

- **分层归属明确**：SDK 能力配置放 `agiwo/config/settings.py`；Console 部署/渠道配置放 `console/server/config.py`
- **命名空间统一**：项目自有环境变量仅允许 `AGIWO_*`（SDK）和 `AGIWO_CONSOLE_*`（Console）
- **读取单一入口**：仅配置模块允许读取环境变量；业务模块禁止新增 `os.getenv(...)`
- **外部 Provider 键保留标准名**：如 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `AWS_REGION`
- **每个配置项必须有 owner**：新增配置前先判断归属（SDK 核心、Console 服务、或 Agent 持久化配置）
- **Console 不承载通用 SDK 运行参数**：Agent 行为参数应存入 `agent_configs.options/model_params`，不应长期堆在全局 `.env`
- **文档同步**：新增/删除/重命名配置项必须同步更新 `.env.example`、`README.md`、`docs/CONFIGURATION_REFACTOR_PLAN.md`

## Key Design Decisions

1. **Agent 是具体类，不是 ABC**。扩展通过 Hooks + Tool 组合，不通过继承
2. **Agent as Tools + Scheduler 编排**。同步嵌套通过 `AgentTool`，异步编排通过 `Scheduler`
3. **Streaming-first**。`run()` 内部也走流式路径，只是自动 drain 事件流
4. **StreamChannel 单消费者**。`read()` 只能被 claim 一次，避免竞争
5. **存储通过配置注入**。`AgentOptions` 持有 `RunStepStorageConfig` + `TraceStorageConfig`，`StorageFactory` 负责创建。不使用全局 config 自动创建
6. **Event Pipeline 中间件模式**。Storage 和 Trace 都是事件流的下游消费者，核心执行只产生事件，不涉及任何 Storage 操作
7. **Tool 注册用装饰器**。`@builtin_tool("name")` + `@default_enable` 自动注册，支持 entry_points 扩展
8. **ToolResult 双内容字段**。`content` 给 LLM 消费，`content_for_user` 给前端展示
9. **Skill 系统可选启用**。通过 `AgentOptions.enable_skill` 控制，不影响核心流程
10. **Scheduler 是 Agent 之上的编排层**。Agent 不感知 Scheduler，无循环依赖。调度工具从外部注入
11. **ToolResult.termination_reason 通用终止机制**。工具可通过返回 `termination_reason` 终止执行循环，不限于 Scheduler 场景
12. **TaskGuard 集中式护栏**。所有调度限制（spawn 深度、子 Agent 数量、唤醒次数、超时）收敛到 `TaskGuard`，Tools 和 Scheduler 不自行检查
13. **Persistent Root Agent**。root agent 可通过 `persistent=True` 保持存活，完成后自动进入 SLEEPING 等待新任务
14. **Guaranteed Summarize-on-Timeout**。`enable_termination_summary` 默认 True + 超时唤醒机制，确保子 Agent 即使超时也能产出摘要
15. **Scheduler Output Streaming**。Agent 文本产出（而非 SLEEPING 状态）驱动消息投递。`submit_and_subscribe` / `submit_task_and_subscribe` 返回 `AsyncIterator[SchedulerOutput]`，每当 Agent 产出文本就 yield 一次，消费者（如飞书 Channel）逐条发送给用户
16. **Pending Events + Debounce 机制**。子 Agent 睡眠/完成/失败时向父 Agent 投递 `PendingEvent`，Scheduler tick 检测 debounce 条件（min_count 或 max_wait）后唤醒父 Agent 处理通知。解决父 Agent 无法感知子 Agent 中间状态的问题
17. **Steering 机制**。外部（Scheduler/Channel）可通过 `steering_queue` 向 RUNNING 中的 Agent 注入实时消息，在下次 LLM 调用前注入为 `<system-steering>` tag。`Scheduler.steer()` 同时作为 `USER_HINT` 持久化，保证 Agent 睡眠时也能收到
18. **Health Check**。`TaskGuard.find_unhealthy()` 检测 RUNNING 状态超过阈值无活动的 Agent（`last_activity_at` 追踪），Scheduler 每 tick 检查并向父 Agent 投递 `HEALTH_WARNING` 事件（带去重）
19. **管理工具 cancel_agent + list_agents**。父 Agent 可通过工具主动取消子 Agent（含 force 保护）和查看所有子 Agent 详情，解决子 Agent 卡死无法管理的问题
20. **配置治理采用“单一读取入口 + 分层命名空间”**。环境变量只在配置层读取，SDK 与 Console 通过命名空间分离，避免业务模块散落 `os.getenv` 和跨层配置污染

## Build & Test Commands

```bash
# 安装依赖 (使用 uv)
uv sync

# 运行单元测试 (mock, 不需要 API Key)
uv run pytest tests/ -v

# 运行集成测试 (需要 .env 中的 API Key)
uv run python test_real_agent.py
uv run python test_real_api.py

# 类型检查 (如果配置了 mypy/pyright)
uv run mypy agiwo/

# 单独测试某个模块
uv run pytest tests/llm/ -v
uv run pytest tests/agent/ -v
uv run pytest tests/scheduler/ -v
```

## Development Notes

### RecentChanges

**2026-03-06 (Skills Directories + Console Agent Config UI)**

- **`skills_dirs` Unified**: `AgentOptions` skills path config is now `skills_dirs`, accepting either a single string or a list; single values are normalized to a one-item list.
- **Relative Path Resolution Fixed**: Agent-level `skills_dirs` resolve relative to `config_root` / effective root path; env-level `AGIWO_SKILLS_DIRS` resolve relative to `AgiwoSettings.root_path`.
- **Legacy Console Data Normalized**: Console API schema and builder normalize legacy `skills_dir` payloads/records into `skills_dirs` so existing saved agents can be edited and resaved cleanly.
- **Console Agent Form Expanded**: Web create/edit pages now expose all currently supported persisted agent options and model params, including config root, termination summary settings, memory/stream limits, compact prompt, and sampling penalties.

**2026-03-05 (Configuration Namespace Cleanup — Phase 3)**

- **SDK Prefix Standardized**: `AgiwoSettings` env prefix switched to `AGIWO_` (`case_sensitive=False`)
- **Legacy Env Aliases Removed**: 移除 `AGIO_*` 工具配置别名与混合大小写 skills 环境变量兼容
- **Console Env Cleanup**: 移除 `default_agent_options` / `default_agent_model_params` 环境变量注入入口
- **Env Templates Unified**: 根目录与 `console/.env.example` 已统一为 `AGIWO_* / AGIWO_CONSOLE_*` 规范

**2026-03-05 (Configuration Governance Plan — SDK + Console)**

- **Configuration Inventory Added**: 新增 `docs/CONFIGURATION_REFACTOR_PLAN.md`，完整盘点 SDK/Console/散落 env 读取点与问题根因
- **Layered Ownership Defined**: 明确 Provider 凭证、SDK 运行配置、Console 服务配置、Agent 持久化配置四层边界
- **Single-Read Rule Defined**: 约束环境变量仅在配置模块读取，业务模块禁止新增 `os.getenv` 直读
- **Namespace Cleanup Plan**: 规划 `AGIWO_*` / `AGIWO_CONSOLE_*` 主命名空间，逐步收敛历史 `AGIO_*` 与混合命名

**2026-03-04 (Scheduler Redesign — Health Check, Pending Events, Steering)**

- **Pending Events System**: `PendingEvent` dataclass + `SchedulerEventType` enum added to `models.py`. Events saved to `pending_events` table (SQLite) or in-memory dict. Consumed via debounce: `find_agents_with_debounced_events(min_count, max_wait, now)`.
- **Debounce Config**: `SchedulerConfig.event_debounce_min_count` and `event_debounce_max_wait_seconds` (also in `AgiwoSettings`). Default: wake on first event, max wait 30s.
- **AgentState New Fields**: `explain: str | None` (set by `sleep_and_wait`), `last_activity_at: datetime | None` (updated each step), `recent_steps: list[dict] | None` (rolling 10-entry window of step summaries).
- **Steering Mechanism**: `ExecutionContext.steering_queue: asyncio.Queue | None` — each execution gets its own queue. `AgentExecutor._drain_steering_queue()` injects pending messages into the next LLM call. `Agent.get_steering_queue()` exposes queue for external callers. `Scheduler.steer(state_id, message)` puts message in queue + saves as `USER_HINT` pending event.
- **on_step Hook Wrapping**: `SchedulerExecutor._wrap_on_step_hook()` wraps existing hook to call `_sync_step_to_state()`, updating `last_activity_at` and `recent_steps` after each step.
- **Pending Event Generation**: `SchedulerExecutor._emit_event_to_parent()` creates events on child sleep (`CHILD_SLEEP_RESULT`), completion (`CHILD_COMPLETED`), and failure (`CHILD_FAILED`).
- **Health Check**: `TaskGuard.find_unhealthy(now, threshold_seconds)` finds RUNNING agents with stale activity. `Scheduler._check_health()` runs each tick, emitting `HEALTH_WARNING` events to parent agents with deduplication.
- **Pending Events Processing**: `Scheduler._process_pending_events()` runs each tick — finds agents meeting debounce threshold, wakes them via `SchedulerExecutor.wake_agent_for_events()` with formatted event context.
- **New Tools**: `CancelAgentTool` (cancel child + subtree, requires `force=true` for RUNNING agents), `ListAgentsTool` (list direct children with status/explain/recent_steps/result).
- **Enhanced Tools**: `SleepAndWaitTool` — new `explain` parameter stored in `AgentState.explain`. `QuerySpawnedAgentTool` — returns `explain`, `last_activity_at`, `recent_steps`.
- **Channel Layer Steering**: `AgentRuntimeManager._handle_running_state()` now calls `Scheduler.steer()` instead of blocking on `wait_for()`, enabling non-blocking real-time hints to running agents.
- **BashTool Agent Tracking**: `ProcessRecord.agent_id` field. `ProcessRegistry.start_process()` accepts `agent_id`. `ProcessRegistry.list_processes_by_agent()` for per-agent process lookup. `LocalSandbox` and `Sandbox` protocol updated.
- **`_tick` Extension**: Order is now `propagate_signals → enforce_timeouts → check_health → process_pending_events → start_pending → wake_sleeping`.
- **`AgentStateStorage` New Abstracts**: `find_running()`, `save_event()`, `get_pending_events()`, `delete_events()`, `find_agents_with_debounced_events()`, `has_recent_health_warning()` added to ABC + both implementations.
- **`update_status` Extended**: Now accepts `explain`, `last_activity_at`, `recent_steps` sentinel kwargs.

**2026-03-03 (Structured UserMessage Input)**

- **Structured Input Model**: Introduced `UserMessage` dataclass with `content: list[ContentPart]` + optional `ChannelContext`. `UserInput = str | list[ContentPart] | UserMessage`. Normalize via `normalize_to_message()`.
- **KV Cache Protection**: Dynamic context (channel metadata, memories, hook results) remains in user message content via `MessageAssembler._prepend_to_user_message()`. System prompt stays static.
- **MessageAssembler Refactor**: `assemble()` now accepts `channel_context: ChannelContext | None` kwarg. Renders context/memories/hook as structured text blocks prepended to last user message.
- **Scheduler UserInput**: All Scheduler APIs (`run`, `submit`, `submit_task`, `submit_and_subscribe`, `submit_task_and_subscribe`) accept `UserInput` instead of `str`. `AgentState.task` and `WakeCondition.submitted_task` are `UserInput`, serialized via `serialize_user_input`/`deserialize_user_input` in SQLite store.
- **System Instruction Migration**: `<system-instruction>` no longer prepended to task string. Now appended to child agent's system prompt as `<task-instruction>` block (set once per child, KV cache friendly).
- **Feishu Multimodal**: `FeishuApiClient` gains `download_image()`, `download_message_resource()`, `_authorized_binary_request()`. Service downloads attachments to local tmp dir, builds `ContentPart(url=local_path, mime_type, metadata={name, size})`. `to_message_content()` renders local-path resources as text placeholders with file info + path for Agent tool use.
- **Channel Infra**: `BatchPayload.rendered_user_input: str` → `user_message: UserMessage`. `BaseChannelService._render_batch_prompt()` → `async _build_user_message()`. `AgentRuntimeManager.submit_to_scheduler()` accepts `UserInput`.

**2026-03-02 (Scheduler Output Streaming)**

- **Output-driven message delivery**: Agent text output (not SLEEPING state) triggers message delivery to users. `SchedulerExecutor._handle_agent_output()` emits `SchedulerOutput` via `_emit_output()` on every branch (SLEEPING, persistent idle, child completion, periodic).
- **New subscribe API**: `submit_and_subscribe()` and `submit_task_and_subscribe()` combine task submission with output consumption, returning `AsyncIterator[SchedulerOutput]`. Channel is created before execution starts to prevent output loss.
- **`include_child_outputs` config**: Controls whether child agent outputs are pushed to the output channel (default True).
- **`wait_for()` race fix**: Persistent agent in SLEEPING state with a pending `submitted_task` no longer causes `wait_for()` to return stale results.
- **Feishu Channel streaming**: `_run_batch_with_scheduler()` rewritten to consume `subscribe` output stream; each output is sent as a separate Feishu message (first as reply, subsequent as new messages).

**2026-03-01 (Token Limits Refactor)**

- **Limit Names Clarified**:
  - `max_context_window_tokens`: single LLM call `input + output` limit
  - `max_tokens_per_run`: accumulated `input + output` across a run
  - `max_output_tokens_per_call`: model parameter name in console, mapped to provider `max_tokens`
- **Cost Guardrail Added**: `max_run_token_cost` in `AgentOptions`, with model pricing params (`cache_hit_price`, `input_price`, `output_price`, USD per 1M tokens).
- **Termination Reasons Split**:
  - `MAX_OUTPUT_TOKENS_PER_CALL`
  - `MAX_CONTEXT_WINDOW_TOKENS`
  - `MAX_TOKENS_PER_RUN`
  - `MAX_RUN_TOKEN_COST`
- **Executor Flow Updated**: one loop now performs `pre-check -> LLM -> tools -> post-check`, ensuring tool calls for the current step execute before token-limit stop and fixing the previous `no tool_calls -> COMPLETED` precedence bug.
- **Summary Policy Tightened**: token/cost limit terminations do not auto-generate termination summaries.

**2025-02-27 (System Prompt Auto-Refresh)**

- **Lazy System Prompt Building**: `SystemPromptBuilder.build()` is now called lazily in `initialize()` at first execution, not during Agent construction.
- **Auto-refresh on Change**: `get_system_prompt()` automatically detects changes to SOUL.md (mtime check) and skills directory (fingerprint check), refreshing the prompt when needed.
- **Change Tracking**: Builder tracks `_soul_mtime` and `_skills_fingerprint` to detect modifications without unnecessary rebuilds.
- **Agent Delegation**: Agent no longer caches `system_prompt` locally; it delegates to `SystemPromptBuilder.get_system_prompt()` on each execution, ensuring fresh prompts when files change.
- **Manual Refresh**: `SkillManager.reload()` available for explicit skill refresh; builder's `_refresh_prompt()` handles the full rebuild cycle.

**2025-02-27 (Scheduler Hardening)**

- **Scheduler session isolation**: `SleepAndWaitTool` and `TaskGuard.check_spawn()` now filter children by `session_id`, preventing cross-session pollution.
- **Child agent no spawn**: `create_child_agent()` filters out `spawn_agent` tool from parent — child agents cannot cascade spawn grandchildren.
- **Fix wake_agent_not_found**: `_maybe_cleanup_agent()` now async-checks actual state before cleanup; SLEEPING agents remain in registry for wake.
- **Dedup scheduling**: `_dispatched_state_ids` prevents duplicate scheduling of same state across ticks.
- **Wake message clarity**: `_build_wake_message()` separates "## Successful Results" from "## Failed Agents", preventing LLM from treating errors as results.
- **Spawn tool guidance**: Improved `SpawnAgentTool` description to discourage unnecessary spawning.

**2025-02-27**

- **AgentId Counter**: `Agent._generate_default_id()` now uses `{name}-{seq:03d}` format (e.g., `bodhi-001`) instead of random UUID. Console `build_agent()` no longer passes `id` to use SDK auto-generation.
- **Lazy Workspace Init**: Directory creation and system prompt building moved to `initialize()`, called at execution time. Agents created but never run no longer create side-effect directories or build prompts.
- **System Prompt Auto-refresh**: `DefaultSystemPromptBuilder` now tracks SOUL.md mtime and skills directory fingerprint, automatically refreshing when files change. Agent retrieves fresh prompts via `get_system_prompt()` on each execution.
- **Child Agent Instruction**: `config_overrides["instruction"]` is now injected into task via `<system-instruction>` tag at runtime, instead of overriding `system_prompt` at creation time.
- **Name/Id Separation**: `Agent(name, description, model, *, id=None...)` — `name` is first positional param (config identifier), `id` is keyword-only optional (runtime identity, auto-generated if not provided).

### 添加新 LLM Provider

1. OpenAI 兼容：继承 `OpenAIModel`，覆写 `_resolve_api_key()` 和 `_resolve_base_url()`
2. 非兼容：继承 `Model`，实现 `arun_stream()`，确保输出标准化为 `StreamChunk`
3. 在 `agiwo/llm/__init__.py` 中导出
4. 在 `agiwo/config/settings.py` 中添加对应的环境变量配置

### 添加新 Hook

1. 在 `agiwo/agent/hooks.py` 中添加类型别名和字段
2. 在 `Agent._execute_workflow` 或 `AgentExecutor._run_loop` / `_execute_tools` 中调用

### 添加新 Builtin Tool

1. 在 `agiwo/tool/builtin/` 下创建新 `.py` 文件
2. 用 `@builtin_tool("name")` 装饰类
3. 可选用 `@default_enable` 标记为默认启用 (Agent tools=None 时自动加载)

### 注意事项

- `agent/inner/` 下的模块是内部实现，不要在 `agiwo/__init__.py` 中导出
- `TYPE_CHECKING` 仅用于解决 `agent_tool.py` 中 Agent 的前向引用 (已是唯一例外)
- `StepRecord` 使用工厂方法 (`StepRecord.user()`, `.assistant()`, `.tool()`) 创建，不要直接构造
- `ToolExecutor` 的 cache 功能尚未完全实现 (见 executor.py 中的 FIXME)
- Anthropic provider 有独立的流式实现，不走 OpenAI 兼容路径

## Maintaining AGENTS.md

当以下变更发生时，更新此文件：

- **新增/删除/重命名顶层模块或包** -> 更新 Project Map
- **核心组件 API 变更** (Agent, Model, BaseTool, AgentHooks) -> 更新 Core Components
- **新增设计决策或推翻旧决策** -> 更新 Key Design Decisions
- **新增编码约定** -> 更新 Code Style & Conventions
- **构建/测试流程变更** -> 更新 Build & Test Commands

原则：保持此文件简洁，只记录 AI 助手和新开发者需要的关键信息。不要在此文件中重复代码中已有的 docstring。
