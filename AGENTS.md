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
| `agiwo/agent/` | Agent 对外入口与运行时领域模型。包含 `runtime_tools/` 宿主工具适配层、`trace/` agent trace adapter、`inner/` 执行内部实现、`prompt/` prompt runtime、`storage/` Run/Session 持久化。包根 `agiwo.agent` 是公开 API，纯配置模型落在 `config.py`，其余 canonical 模型分别落在 `input.py`、`runtime.py`、`compact_types.py`、`memory_types.py`。 |
| `agiwo/llm/` | Model 抽象、Provider 适配器、配置策略、消息/事件归一化，以及统一的 model factory。 |
| `agiwo/tool/` | Tool 抽象、最小执行上下文、builtin tools、后台进程 registry（`process/`），以及工具侧存储（如 citation）。 |
| `agiwo/scheduler/` | Agent 之上的编排层。`scheduler.py` 是 facade，`engine.py` 负责公开编排 API 并组装内部 owner；共享状态迁移收口到 `state_ops.py`，tick phases 收口到 `tick_ops.py`，tree cancel/shutdown 收口到 `tree_ops.py`，tool-facing control helpers 收口到 `control_ops.py`，`runner.py` 负责单次 agent cycle 执行，`wake_messages.py` 负责唤醒消息构造，`coordinator.py` 只管进程内协作状态，`control.py` 定义 tools 依赖的窄接口，`store/` 只负责持久化。 |
| `agiwo/observability/` | Trace/Span 模型、查询接口与 trace storage 实现；agent 事件到 Trace 的适配层已收口到 `agiwo/agent/trace/`。 |
| `agiwo/embedding/` | Embedding 抽象与 factory，包含本地/OpenAI 风格实现。 |
| `agiwo/skill/` | Skill 的发现、路径规则（`config.py`）、加载、注册、异常定义，以及 `SkillTool` 桥接。 |
| `agiwo/workspace/` | Agent workspace 路径语义、模板/bootstrap、工作区文档读取与变更 token。 |
| `agiwo/memory/` | 共享 MEMORY 索引/切块/搜索能力，以及 `WorkspaceMemoryService`。 |
| `agiwo/config/` | SDK 全局配置入口、Provider 枚举与共享设置。 |
| `agiwo/utils/` | 跨模块运行时工具。`storage_support/` 负责共享 SQLite/Mongo runtime、schema/index 初始化等基础设施。 |

### Console (`console/`)

| Path | Responsibility |
| --- | --- |
| `console/server/` | FastAPI 控制面与 runtime 集成。 |
| `console/server/routers/` | API/SSE 边界，只做 HTTP 路由与请求/响应装配。 |
| `console/server/services/` | 应用服务层。`agent_lifecycle.py`（构建/恢复/恢复 agent）、`agent_registry/`（配置 CRUD + store 子包）、`storage_wiring.py`（存储 config builders + NotifyingTraceStorage）、`chat_sse.py`、`metrics.py`。 |
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
- 公开构造入口是 `Agent(AgentConfig(...), *, model=..., tools=..., hooks=..., id=...)`；`AgentConfig` 只承载纯配置，不放 live object。
- 对外执行原语是 `start(...) -> AgentExecutionHandle`；`run(...)` / `run_stream(...)` 只是便利封装。嵌套 child 定义统一通过 `derive_child_spec(...) -> ChildAgentSpec` 生成；`ChildAgentSpec` 只保留覆写数据，不携带 live runtime。
- `run(...)` 只支持 root run；嵌套 agent 执行是内部协议，由 `runtime_tools/AgentTool` 通过 `Agent.run_child(...)` 进入，不再暴露公开 `context` 参数。
- `AgentExecutionHandle` 是一次活执行实例，持有 `run_id/session_id`、`stream()/wait()/steer()/cancel()`；`steer()` 不再属于 `Agent` 模板对象。
- `Agent` 内部分离 definition-scoped owner（`AgentDefinitionRuntime` 负责 hooks / sdk tools / prompt runtime / skill manager / root-child-scheduler child 派生）与 resource-scoped owner（`AgentResourceOwner` 持有 run-step storage / session storage / trace storage / active root executions）；核心执行循环在 `agiwo/agent/inner/`。
- `AgentSessionRuntime` 是 session 级共享 owner，持有 sequence owner、trace_id、abort signal、steering queue 和 stream subscribers。
- `AgentHooks` 是可选 async 回调的 dataclass；当前 hook 覆盖 run、tool、LLM、step、memory write/retrieve。
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

- `BaseTool` 定义稳定契约：名称、描述、参数 schema、并发安全性、可选 `gate(..., context: ToolContext) -> ToolGateDecision` 预检，以及 `execute(..., context: ToolContext) -> ToolResult`。
- agent 运行时内部统一通过 `AgentRuntimeTool` 执行工具；scheduler 控制型 tools 走 runtime tool 契约，不再把终止控制塞进 `ToolResult`。
- `AgentTool` / `as_tool()` 属于 `agiwo.agent.runtime_tools`，是 agent runtime adapter，不属于 `agiwo.tool/` core。
- 生产代码统一通过 `ToolResult.success()/failed()/aborted()/denied()` 构造结果。
- builtin tools 放在 `agiwo/tool/builtin/`，通过 `@builtin_tool(...)` 注册；`@default_enable` 控制默认自动启用。
- `bash` 与 `bash_process` 是分离工具；后台任务的巡检/日志/停止/输入属于 `bash_process`。
- 共享 MEMORY 检索统一通过 `agiwo.memory.WorkspaceMemoryService`，builtin retrieval tool 只是 adapter。
- citation 等工具侧持久化在 `agiwo/tool/storage/citation/`。

### Scheduler

- `Scheduler` 是 Agent 之上的编排层；依赖方向保持 `scheduler -> agent`。
- 当前公开编排接口包括：`run`、`submit`、`enqueue_input`、`stream`、`wait_for`、`steer`、`cancel`、`shutdown`。
- `Scheduler` 只做 facade 和 lifecycle；所有编排语义统一收口到 `SchedulerEngine`。
- `SchedulerEngine` 是公开编排 owner：提交、等待、steer、cancel、shutdown 等入口留在这里；共享状态迁移下沉到 `SchedulerStateOps`，tick 各阶段下沉到 `SchedulerTickOps`，tree cancel/shutdown 下沉到 `SchedulerTreeOps`，tool-facing control helper 下沉到 `SchedulerControlOps`。
- `SchedulerRunner` 只负责单次执行：运行 root/child/wake cycle、构造 wake message、把 `RunOutput` 翻译成 scheduler outcome；scheduler 内部执行原语已经收口到 `SchedulerAgentPort.start(...) -> AgentExecutionHandlePort`，唯一 live observation path 是 `AgentExecutionHandle.stream()`。
- `SchedulerCoordinator` 只保存进程内状态：已注册 agent、active execution handle、abort signal、active task、dispatch dedupe、wait event、stream channel；不得依赖 store。
- `SchedulerControl` 是 scheduler tools 唯一允许依赖的接口；tools 不得直接碰 store / guard / coordinator / runner。
- `TaskGuard` 是 spawn/wake 的唯一护栏入口。
- scheduler 状态现在显式区分 `WAITING`、`IDLE`、`QUEUED`；不要再把待命/排队语义塞回一个泛化 `SLEEPING`。
- 当前唤醒路径包括 `WAITSET`、`TIMER`、`PERIODIC`、`PENDING_EVENTS`。
- scheduler state storage 当前支持 memory/sqlite，由 `Scheduler` 自己创建和持有；store 边界保持为通用 repo（`save/get/patch/list`），wake/timeout/debounce/signal 规则统一放在 scheduler 侧 selectors/tick/state ops。
- `Scheduler` 的外部流式协议直接复用 `AgentStreamItem`；不要在 SDK core 再维护第二套 live-output protocol。

### Storage & Observability

- Run/Step 持久化通过 `AgentOptions.run_step_storage` 配置；Trace 持久化通过 `AgentOptions.trace_storage` 配置。
- Session/compact metadata 存储与 run-step storage 分离，位于 `agiwo/agent/storage/session.py`。
- Agent 运行记录管线已收口为同步单 owner：`RunRecorder` 统一负责 step/run lifecycle、storage write、trace callback、step observer 与 stream fanout；`AgentExecutionHandle.stream()` / `run_stream()` 对外暴露的是 typed `AgentStreamItem`。
- `BaseTraceStorage` 既支持查询，也支持实时订阅，Console trace SSE 会用到这个能力。

## Architecture Boundaries

### SDK

- Python 版本基线是 3.11+。
- 所有公开方法与核心数据结构都应带类型注解。
- async 逻辑保持显式，不要把 `await` 藏进难追踪的 helper。
- 哨兵判断优先用 `is not None`；truthy 检查只在语义明确时使用。
- 不为“可能以后有用”保留遗留兼容层；除非明确要求，否则直接删除旧路径。
- 遇到循环依赖优先重构依赖方向，不要靠局部导入规避。
- 同一外部 use case 不得并列暴露两套 public API；内部可以保留多种 lifecycle mutation 动作，但 facade 必须内化复杂度。

### Console

- `console/server/schemas.py` 与 `console/server/response_serialization.py` 只属于 API/SSE 边界。
- 共享的 Console 领域模型放 `console/server/domain/`。
- Console 自己持有 API/表单 DTO；不要再把 `Input/Patch` 请求模型塞回 SDK。
- session 相关代码收口到 `console/server/channels/session/` 包：`models.py`（数据模型与 store protocol）、`binding.py`（domain 操作与异常）、`context_service.py`（session/chat-context 协调）、`manager.py`（消息批处理与防抖）。session identity 字段通过 `binding.py` 协调，经 `ChannelChatSessionStore.apply_session_mutation(...)` 原子写入。
- 渠道运行时由三个独立子服务组成：`SessionContextService`（session/chat-context 生命周期）、`RuntimeAgentPool`（runtime agent 缓存与 config 指纹刷新）、`AgentExecutor`（scheduler 交互与状态路由）。`BaseChannelService` 直接持有这三者，不再经过 facade。
- `AgentExecutor.execute()` 对 scheduler 状态做显式路由：None/COMPLETED/FAILED → submit；IDLE/FAILED(persistent) → enqueue；RUNNING/WAITING/QUEUED → steer；PENDING → wait then retry。timeout 为可选参数，默认不设硬超时。
- Feishu store 实现收口到 `console/server/channels/feishu/store/` 包：`__init__.py`（protocol + factory）、`memory.py`（内存实现）、`sqlite.py`（SQLite 实现）。
- Feishu 模块合并约定：`message_parser.py` 包含 envelope 类型 + sender 解析 + 解析 facade；`message_builder.py` 包含 attachment 解析 + UserMessage 构建；`connection.py` 包含 SDK 适配层 + WebSocket 连接管理。
- Console 工具的展示、解析、组装都必须经过 `ConsoleToolCatalog`。
- Agent 配置写入保持 full replace；不要回到 partial merge / patch DTO 语义。
- `storage_wiring.py` 同时包含 storage config builders 和 `NotifyingTraceStorage`（Console 侧 trace pub/sub）。
- Agent registry 收口到 `console/server/services/agent_registry/` 包：`registry.py`（CRUD 服务）、`models.py`（AgentConfigRecord）、`store/`（protocol + factory + memory/mongo/sqlite 实现）。

### Configuration

- SDK 配置入口在 `agiwo/config/settings.py`；Console 专属部署/渠道配置入口在 `console/server/config.py`。
- 项目自有环境变量命名空间保持 `AGIWO_*` 与 `AGIWO_CONSOLE_*`；外部 Provider 继续使用其标准变量名。
- 业务模块不要新增散落的 `os.getenv(...)`；环境变量读取应集中在配置层，当前唯一例外是 `agiwo.llm.factory` 根据 `api_key_env_name` 做运行时密钥解析。

### Construction & Layering

- builtin tools 应自行按配置构建 model/HTTP/storage 依赖，除非它本质上是在包装宿主运行时对象。
- LLM 创建统一走共享 factory / config policy。
- 不要在 agent 包外越权依赖 `agiwo.agent.inner`。
- 不要在 scheduler tools 里依赖 `SchedulerEngine` / `SchedulerCoordinator` / `AgentStateStorage` / `TaskGuard`；一律只依赖 `SchedulerControl`。
- 不要把 store mutation、递归 cancel/shutdown、tick dispatch 重新塞回 `Scheduler` facade。
- 不要在 SDK core 里同时维护 text-only 和 typed 两套 scheduler stream API；如需文本适配，只能放在消费侧边缘。
- 当同一语义逻辑出现第 2 次时就要评估抽象；不要继续把热点文件堆成 God Object。

## Lint & Guardrails

- Python 代码改动后默认执行：`uv run python scripts/lint.py changed`
- 脏工作区下只检查改动文件：`uv run python scripts/lint.py files <path1> <path2> ...`
- 只跑 import contract：`uv run python scripts/lint.py imports`
- 提交前必须至少跑一遍与 CI 一致的 lint 四步，不要只依赖 `scripts/lint.py changed`：
  1. `uv run ruff check --ignore C901 --ignore PLR0911 --ignore PLR0912 agiwo/ console/server/ tests/ console/tests/ scripts/`
  2. `uv run ruff format --check agiwo/ console/server/ tests/ console/tests/ scripts/`
  3. `uv run python scripts/lint.py imports`
  4. `uv run python scripts/repo_guard.py`
- 当前阻塞层是：`ruff + repo_guard.py + import-linter`
- 代码格式/import 排序/全量类型检查目前不是默认阻塞门槛
- 新增 `# noqa` / `# type: ignore` 前先修代码；必须压制时精确到规则
- 不要写 schema migration / 自动补列逻辑；数据模型变更时直接让旧数据失败并通知用户清理历史数据
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

# 提交前跑一遍与 CI 对齐的 lint
uv run ruff check --ignore C901 --ignore PLR0911 --ignore PLR0912 agiwo/ console/server/ tests/ console/tests/ scripts/
uv run ruff format --check agiwo/ console/server/ tests/ console/tests/ scripts/
uv run python scripts/lint.py imports
uv run python scripts/repo_guard.py

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

See [CHANGELOG.md](./CHANGELOG.md) for the full change history.

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
- Tool runtime 缓存是 session 级缓存，只有 `tool.cacheable = True` 才生效。
- Anthropic 有独立实现路径；只有显式 `anthropic-compatible` 场景才按兼容协议处理。

## Maintaining AGENTS.md

- 当“目录职责”、“公开 API”、“机器护栏规则”、“标准开发流程”发生变化时更新本文件。
- 继续保持目录/包级别描述，不要退回到逐文件目录树。
- 如果某条说明开始频繁失真，优先上移抽象层级，而不是继续堆更多细节。
