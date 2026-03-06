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
├── embedding/                # Embedding 抽象层与工厂
├── llm/                      # LLM Provider 抽象层
│   ├── base.py               # Model ABC + StreamChunk
│   ├── factory.py            # ModelFactory：统一构建入口
│   ├── openai.py             # OpenAIModel (其他 OpenAI 兼容 Provider 的基类)
│   ├── anthropic.py          # AnthropicModel (独立实现，非 OpenAI 兼容)
│   ├── bedrock_anthropic.py  # Bedrock Anthropic 实现
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
│   │   ├── config.py         # 内置工具配置聚合
│   │   ├── bash_tool/        # Bash 工具
│   │   ├── retrieval_tool/   # 记忆检索工具
│   │   ├── web_search/       # 网页搜索工具
│   │   └── web_reader/       # 网页阅读/抽取工具
│   └── permission/           # 工具权限管理
├── tool/storage/             # Tool 存储实现（如 citation）
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
│   ├── tools.py              # Spawn/Sleep/Query/Cancel/List 等调度工具
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
    ├── sqlite_pool.py        # SQLite 连接池
    ├── mongo_pool.py         # MongoDB 连接池
    ├── retry.py              # 异步重试装饰器
    └── tojson.py             # JSON 序列化工具

console/
├── server/                   # Console API 与 Agent runtime 管理
├── web/                      # Console 前端
└── tests/                    # Console 后端测试
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
- Agent 和 Tool 的模型实例都通过 `agiwo.llm.factory` 统一构建

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
- Scheduler 从外部注入调度工具（`spawn_agent`, `sleep_and_wait`, `query_spawned_agent`, `cancel_agent`, `list_agents`）
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
- **DRY 强约束**：同一语义逻辑出现第 2 次时就要评估抽象；出现第 3 次必须收敛为共享实现（helper / spec table / strategy map），禁止继续复制分支代码
- **Builtin Tool 依赖自构建**：内置工具需要的 Model、HTTP Client、Storage 等基础设施依赖应由 Tool 在构造函数内根据配置创建；不要从 Agent / Console builder 注入 live client、live model、store 实例。例外仅限必须包装宿主运行时对象的 Tool（如 `AgentTool`、`SkillTool`、Scheduler tools、`BashTool` 的 sandbox/hook）。
- **Model 构建走统一工厂**：无论 Agent 还是 Tool，需要创建 LLM `Model` 时都应走共享的 model factory / config，不要新增绕过 `agiwo.llm` 抽象的专用 HTTP client。
- **禁止向后兼容** (除非明确要求)，及时删除遗留代码
- **循环依赖**：遇到时必须重构组件耦合关系，禁止延迟导入等绕行手段
- 不要使用 `rm` 删除文件，用 `mv` 移动到 `trash/` 目录

## Configuration Governance

- **分层归属明确**：SDK 能力配置放 `agiwo/config/settings.py`；Console 部署/渠道配置放 `console/server/config.py`
- **命名空间统一**：项目自有环境变量仅允许 `AGIWO_*`（SDK）和 `AGIWO_CONSOLE_*`（Console）
- **读取单一入口**：配置模块是主入口；业务模块禁止新增散落 `os.getenv(...)`。唯一例外是 `agiwo.llm.factory` 基于 `api_key_env_name` 做运行时密钥解析
- **外部 Provider 键保留标准名**：如 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `AWS_REGION`
- **兼容 Provider 显式配置**：`openai-compatible` / `anthropic-compatible` 视为协议适配器，不复用 `OPENAI_*` / `ANTHROPIC_*` 的默认凭证；必须显式提供 `base_url` 和 `api_key_env_name`
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
21. **Builtin Tool 自行创建基础设施依赖**。内置工具通过 config 创建 Model/HTTP/Storage 等依赖；只有包装宿主运行时对象的 Tool（如 Bash sandbox、Agent/Skill/Scheduler wrappers）才接受外部 live 依赖注入

## Build & Test Commands

```bash
# 安装依赖 (使用 uv)
uv sync

# 运行单元测试 (mock, 不需要 API Key)
uv run pytest tests/ -v

# 运行 Console 服务测试
(cd console && uv run pytest tests/ -v)

# 类型检查 (如果配置了 mypy/pyright)
uv run mypy agiwo/

# 单独测试某个模块
uv run pytest tests/llm/ -v
uv run pytest tests/agent/ -v
uv run pytest tests/scheduler/ -v
```

## Development Notes

### RecentChanges

只保留最近且影响开发决策的变更，历史细节请查 Git 记录。

**2026-03-06 (Model/Tool 配置收敛)**
- Provider 命名统一为 `openai-compatible` / `anthropic-compatible`，移除 `generic` 兼容逻辑。
- Agent/Tool 模型配置统一为：`provider + model_name + base_url + api_key_env_name + sampling params`。
- 禁止在 Agent 记录或默认配置里存明文 `api_key`，统一使用 `api_key_env_name`。
- `web_reader` / `web_search` 改为内置工具自构建依赖，模型创建统一走 `agiwo.llm.factory`。

**2026-03-06 (连接池与存储复用)**
- SQLite / Mongo 相关存储统一使用共享连接池（`agiwo/utils/sqlite_pool.py`、`agiwo/utils/mongo_pool.py`）。
- Citation store 与 Console/SDK 持久化复用同一类连接资源，避免重复连接实例。

**2026-03-04 ~ 2026-03-02 (Scheduler 能力增强)**
- 增加 Pending Events + Debounce、Steering、Health Check、子 Agent 管理工具（cancel/list）。
- 输出流 API `submit_and_subscribe` / `submit_task_and_subscribe` 成为 channel 推送主路径。

**2026-03-03 (输入结构化)**
- `UserInput` 升级为 `str | list[ContentPart] | UserMessage`，调度与通道链路统一支持结构化输入。

### 添加新 LLM Provider

1. 实现 Provider 类：OpenAI 兼容建议继承 `OpenAIModel`；非兼容继承 `Model` 并实现 `arun_stream()`
2. 在 `agiwo/llm/factory.py` 的 `ModelProvider` 和 `PROVIDER_SPECS` 中注册
3. 在 `agiwo/llm/__init__.py` 中导出
4. 如需 provider 级默认配置，在 `agiwo/config/settings.py` 新增字段
5. 补充 `tests/llm/test_factory.py` 与相关集成测试

### 添加新 Hook

1. 在 `agiwo/agent/hooks.py` 中添加类型别名和字段
2. 在 `Agent._execute_workflow` 或 `AgentExecutor._run_loop` / `_execute_tools` 中调用

### 添加新 Builtin Tool

1. 在 `agiwo/tool/builtin/` 下创建新 `.py` 文件
2. 用 `@builtin_tool("name")` 装饰类
3. 可选用 `@default_enable` 标记为默认启用 (Agent tools=None 时自动加载)

### 注意事项

- `agent/inner/` 下的模块是内部实现，不要在 `agiwo/__init__.py` 中导出
- `TYPE_CHECKING` 仅用于打断类型环依赖或隔离重依赖；新增前应确认不能通过重构依赖关系解决
- `StepRecord` 使用工厂方法 (`StepRecord.user()`, `.assistant()`, `.tool()`) 创建，不要直接构造
- `ToolExecutor` 的缓存是会话级缓存（`ToolResultCache`），仅 `tool.cacheable=True` 时生效
- Anthropic provider 有独立的流式实现，不走 OpenAI 兼容路径

## Maintaining AGENTS.md

当以下变更发生时，更新此文件：

- **新增/删除/重命名顶层模块或包** -> 更新 Project Map
- **核心组件 API 变更** (Agent, Model, BaseTool, AgentHooks) -> 更新 Core Components
- **新增设计决策或推翻旧决策** -> 更新 Key Design Decisions
- **新增编码约定** -> 更新 Code Style & Conventions
- **构建/测试流程变更** -> 更新 Build & Test Commands

原则：保持此文件简洁，只记录 AI 助手和新开发者需要的关键信息。不要在此文件中重复代码中已有的 docstring。
