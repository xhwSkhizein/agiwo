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
│   ├── models.py             # AgentState, WakeCondition, WakeType, WaitMode, TaskLimits, SchedulerConfig
│   ├── store.py              # AgentStateStorage ABC + InMemory + SQLite 实现
│   ├── guard.py              # TaskGuard：集中式护栏 (spawn/wake 限制, 超时检测)
│   ├── tools.py              # SpawnAgentTool, SleepAndWaitTool, QuerySpawnedAgentTool
│   └── scheduler.py          # Scheduler：编排层入口 (run / submit / submit_task / shutdown / cancel)
├── skill/                    # Skill 系统 (可选)
│   ├── manager.py            # SkillManager：发现、加载、热重载
│   ├── registry.py           # SkillRegistry + SkillMetadata
│   ├── loader.py             # SkillLoader：SKILL.md 解析
│   └── tool.py               # SkillTool：将 Skill 暴露为 Tool
├── config/
│   └── settings.py           # AgiwoSettings (pydantic-settings, env prefix: agiwo_)
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
- 双模式 API：`run()` 阻塞等待 / `submit()` + `wait_for()` 非阻塞
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
