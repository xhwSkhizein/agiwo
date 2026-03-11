# AGENTS.md

> Agiwo 开发指南。供 AI 编码助手和开发者快速理解当前仓库结构、边界和约定。
>
> 本文件只维护“目录级职责 + 稳定 API/边界”，不再逐个文件镜像源码；实现细节以源码为准。

## How To Use This File

- 如果文档与源码冲突，以源码为准，并顺手更新本文件。
- 优先用本文件定位“应该去哪个目录/模块看”，不要把它当成逐文件索引。
- 如果某条信息很难长期与源码同步，就应该提升抽象层级或直接删掉。

## Repository Layout

### SDK (`agiwo/`)

| Path | Responsibility |
| --- | --- |
| `agiwo/agent/` | Agent 对外入口与运行时领域模型。包含 `inner/` 执行内部实现、`storage/` Run/Session 持久化、`compact/` 上下文压缩能力。`schema.py` 现在是兼容门面，canonical 模型分别落在 `input.py`、`runtime.py`、`compact_types.py`、`memory_types.py`。 |
| `agiwo/llm/` | Model 抽象、Provider 适配器、配置策略、消息/事件归一化，以及统一的 model factory。 |
| `agiwo/tool/` | Tool 抽象、执行器与缓存、builtin tools、权限管理，以及工具侧存储（如 citation）。 |
| `agiwo/scheduler/` | Agent 之上的编排层。`store/` 管持久化状态，`services/` 管 tick orchestration，包根目录保留公开模型、runtime、executor 与 scheduler tools。 |
| `agiwo/observability/` | Trace/Span 模型、事件到 Trace 的收集、序列化，以及 trace storage 实现。 |
| `agiwo/embedding/` | Embedding 抽象与 factory，包含本地/OpenAI 风格实现。 |
| `agiwo/skill/` | Skill 的发现、加载、注册、异常定义，以及 `SkillTool` 桥接。 |
| `agiwo/config/` | SDK 全局配置入口、Provider 枚举与共享设置。 |
| `agiwo/utils/` | 跨模块运行时工具。`storage_support/` 负责共享 SQLite/Mongo runtime、schema/index 初始化等基础设施。 |

### Console (`console/`)

| Path | Responsibility |
| --- | --- |
| `console/server/` | FastAPI 控制面与 runtime 集成。 |
| `console/server/routers/` | API/SSE 边界，只做 HTTP 路由与请求/响应装配。 |
| `console/server/services/` | 应用服务层，负责 agent lifecycle、registry、storage wiring、SSE、session summary 等。 |
| `console/server/domain/` | Console 共享领域模型，避免业务层直接依赖 API DTO。 |
| `console/server/channels/` | 渠道运行时抽象、session binding/context 管理，以及 Feishu 等渠道集成。 |
| `console/server/tools.py` | Console 侧唯一的工具 catalog、tool reference 解析与工具组装入口。 |
| `console/web/` | Console 前端。 |
| `console/tests/` | Console 后端测试。 |

### Supporting Directories

| Path | Responsibility |
| --- | --- |
| `tests/` | SDK 测试，按子系统分目录。 |
| `scripts/` | 低噪音 lint 入口与 repo guard。 |
| `lint/` | import-linter contract 等机器护栏配置。 |
| `docs/` | 设计文档与渠道说明，不是运行时源码真相。 |
| `templates/` | 运行时会消费的模板内容。 |
| `trash/` | 删除文件的落点；优先 `mv` 到这里，不要直接 `rm`。 |

## Core Components

### Agent

- `Agent` 是具体类，不是 ABC。
- 对外执行入口是 `run(...)` 和 `run_stream(...)`；`derive_child(...)` 是当前继承父 Agent 配置生成子 Agent 的标准方式。
- `Agent` 负责配置持有、默认工具/skill 注入、storage wiring、hook dispatch；核心执行循环在 `agiwo/agent/inner/`。
- `AgentHooks` 是可选 async 回调的 dataclass；当前 hook 覆盖 run、tool、LLM、step/event、memory write/retrieve。
- `StepRecord` 使用工厂方法创建，不要直接构造。

### Model

- `agiwo.llm.base.Model` 是通用 dataclass + ABC；Provider 通过 `arun_stream(...) -> AsyncIterator[StreamChunk]` 接入。
- 统一通过 `agiwo.llm.factory` 构建 model；不要在业务代码里散落实例化具体 Provider。
- 新增 Provider 时，当前需要同步更新：
  1. `agiwo/config/settings.py` 中的 `ModelProvider` / `ALL_MODEL_PROVIDERS`
  2. `agiwo/llm/factory.py` 中的 `PROVIDER_SPECS`
  3. `agiwo/llm/__init__.py`（如果需要公开导出）
- `openai-compatible` / `anthropic-compatible` 是显式协议适配器，要求显式 `base_url` 和 `api_key_env_name`。

### Tool

- `BaseTool` 定义稳定契约：名称、描述、参数 schema、并发安全性、`execute(...) -> ToolResult`。
- 生产代码统一通过 `ToolResult.success()/failed()/aborted()/denied()` 构造结果。
- builtin tools 放在 `agiwo/tool/builtin/`，通过 `@builtin_tool(...)` 注册；`@default_enable` 控制默认自动启用。
- `bash` 与 `bash_process` 是分离工具；后台任务的巡检/日志/停止/输入属于 `bash_process`。
- citation 等工具侧持久化在 `agiwo/tool/storage/citation/`。

### Scheduler

- `Scheduler` 是 Agent 之上的编排层；依赖方向保持 `scheduler -> agent`。
- 当前公开编排接口包括：`run`、`submit`、`submit_task`、`wait_for`、`submit_and_subscribe`、`submit_task_and_subscribe`、`steer`、`cancel`、`shutdown`。
- `SchedulerRuntime` 管运行时协作状态，`SchedulerExecutor` 管 agent 执行/唤醒，`scheduler/services/tick_engine.py` 管 tick 阶段，`scheduler/store/` 管 scheduler state 持久化。
- `TaskGuard` 是 spawn/wake/timeout/health check 的唯一护栏入口。
- 当前唤醒路径包括 `WAITSET`、`TIMER`、`PERIODIC`、`TASK_SUBMITTED`、`PENDING_EVENTS`。
- scheduler state storage 当前支持 memory/sqlite，由 `Scheduler` 自己创建和持有。

### Storage & Observability

- Run/Step 持久化通过 `AgentOptions.run_step_storage` 配置；Trace 持久化通过 `AgentOptions.trace_storage` 配置。
- Session/compact metadata 存储与 run-step storage 分离，位于 `agiwo/agent/storage/session.py`。
- 事件管线仍然是：`Executor -> EventEmitter -> StreamChannel -> Storage/Trace consumers -> user`。
- `BaseTraceStorage` 既支持查询，也支持实时订阅，Console trace SSE 会用到这个能力。

## Architecture Boundaries

### SDK

- Python 版本基线是 3.11+。
- 所有公开方法与核心数据结构都应带类型注解。
- async 逻辑保持显式，不要把 `await` 藏进难追踪的 helper。
- 哨兵判断优先用 `is not None`；truthy 检查只在语义明确时使用。
- 不为“可能以后有用”保留遗留兼容层；除非明确要求，否则直接删除旧路径。
- 遇到循环依赖优先重构依赖方向，不要靠局部导入规避。

### Console

- `console/server/schemas.py` 与 `console/server/response_serialization.py` 只属于 API/SSE 边界。
- 共享的 Console 领域模型放 `console/server/domain/`。
- session identity 相关字段（`current_session_id`、`base_agent_id`、`runtime_agent_id`、`scheduler_state_id`）应通过 `console/server/channels/session_binding.py` 协调，并经 `ChannelChatSessionStore.apply_session_mutation(...)` 原子写入。
- Console 工具的展示、解析、组装都必须经过 `ConsoleToolCatalog`。
- Agent 配置写入保持 full replace；不要回到 partial merge / patch DTO 语义。
- `StorageManager` 只管理 run-step/trace/citation；scheduler state storage 由 `Scheduler` 持有。
- Feishu 入站先规范成 `FeishuInboundEnvelope`；parser 保持薄编排，内容提取、sender lookup、群历史状态交给专门模块。

### Configuration

- SDK 配置入口在 `agiwo/config/settings.py`；Console 专属部署/渠道配置入口在 `console/server/config.py`。
- 项目自有环境变量命名空间保持 `AGIWO_*` 与 `AGIWO_CONSOLE_*`；外部 Provider 继续使用其标准变量名。
- 业务模块不要新增散落的 `os.getenv(...)`；环境变量读取应集中在配置层，当前唯一例外是 `agiwo.llm.factory` 根据 `api_key_env_name` 做运行时密钥解析。

### Construction & Layering

- builtin tools 应自行按配置构建 model/HTTP/storage 依赖，除非它本质上是在包装宿主运行时对象。
- LLM 创建统一走共享 factory / config policy。
- 不要在 agent 包外越权依赖 `agiwo.agent.inner`。
- 当同一语义逻辑出现第 2 次时就要评估抽象；不要继续把热点文件堆成 God Object。

## Lint & Guardrails

- Python 代码改动后默认执行：`uv run python scripts/lint.py changed`
- 脏工作区下只检查改动文件：`uv run python scripts/lint.py files <path1> <path2> ...`
- 只跑 import contract：`uv run python scripts/lint.py imports`
- 当前阻塞层是：`ruff + repo_guard.py + import-linter`
- 代码格式/import 排序/全量类型检查目前不是默认阻塞门槛
- 新增 `# noqa` / `# type: ignore` 前先修代码；必须压制时精确到规则
- 不要继续扩展死掉的 `needs_permissions()` API 面
- 生产代码里不要直接构造 `ToolResult(...)`、`StepRecord(...)`，也不要绕过共享工厂去直接实例化具体存储后端或具体 LLM Provider

## Build & Test Commands

```bash
# 安装依赖
uv sync

# AI 修改后的标准检查入口
uv run python scripts/lint.py changed

# 脏工作区下只检查本次改动
uv run python scripts/lint.py files path/to/file.py path/to/other.py

# 只跑 import contract
uv run python scripts/lint.py imports

# SDK 测试
uv run pytest tests/ -v

# Console 后端测试
(cd console && uv run pytest tests/ -v)
```

类型检查不是当前仓库默认门槛；只有分支实际接入某个 checker 时，才运行对应工具。

## Development Notes

### Recent Changes

- **2026-03-11**：`agiwo/agent/schema.py` 已降为兼容门面；输入/运行时/compact/memory 模型按职责拆到独立模块。
- **2026-03-11**：Scheduler 的结构以 `store/ + services/ + runtime/executor` 为准，`PendingEvent`、`steer()`、输出流订阅是当前心智模型的一部分。
- **2026-03-11**：Console 边界继续收敛到 `session_binding`、`ChannelChatSessionStore.apply_session_mutation()`、`FeishuInboundEnvelope`、`ToolReference`、`ConsoleToolCatalog`。
- **2026-03-10**：低噪音 lint（`ruff + repo_guard + import-linter`）已经成为 AI 改码后的默认工作流。
- **2026-03-06**：Model/Tool 配置统一为 `provider + model_name + base_url + api_key_env_name`；builtin web tools 自构建依赖；共享存储 runtime/pool helper 已被多处复用。

### Common Changes

#### 添加新 LLM Provider

1. 在 `agiwo/llm/` 实现 Provider 类。
2. 在 `agiwo/config/settings.py` 注册 provider literal / 默认配置。
3. 在 `agiwo/llm/factory.py` 注册 `ProviderSpec`。
4. 如需公开导出，在 `agiwo/llm/__init__.py` 增补导出。
5. 在 `tests/llm/` 补测试。

#### 添加新 Hook

1. 在 `agiwo/agent/hooks.py` 增加 hook 类型与字段。
2. 在 `agiwo/agent/agent.py` 或 `agiwo/agent/inner/executor.py` 接线。

#### 添加新 Builtin Tool

1. 在 `agiwo/tool/builtin/` 下新增实现。
2. 用 `@builtin_tool(...)` 注册；仅在确实需要默认启用时再加 `@default_enable`。
3. 如果 Console 侧需要自定义展示文案或 build-time 依赖，在 `console/server/tools.py` 增补对应 catalog 逻辑。

### Notes

- `agent/inner/` 是内部实现，不要从 `agiwo/__init__.py` 暴露，也不要在包外直接依赖。
- `TYPE_CHECKING` 只用于确实无法通过重构解决的类型环依赖。
- `ToolExecutor` 缓存是 session 级缓存，只有 `tool.cacheable = True` 才生效。
- Anthropic 有独立实现路径；只有显式 `anthropic-compatible` 场景才按兼容协议处理。

## Maintaining AGENTS.md

- 当“目录职责”、“公开 API”、“机器护栏规则”、“标准开发流程”发生变化时更新本文件。
- 继续保持目录/包级别描述，不要退回到逐文件目录树。
- 如果某条说明开始频繁失真，优先上移抽象层级，而不是继续堆更多细节。
