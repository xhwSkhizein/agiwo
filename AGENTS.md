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
| `agiwo/agent/` | Canonical agent runtime。public API 只从 `agiwo.agent` 暴露；顶层只保留稳定入口与核心 orchestrator（如 `agent.py`、`definition.py`、`run_loop.py`、`llm_caller.py`、`tool_executor.py`、`prompt.py`、`trace_writer.py`）。纯数据模型收口在 `models/`，hook contract 收口在 `hooks/`，nested-agent adapter 收口在 `nested/`，run/session runtime context 与 state helper 收口在 `runtime/`，termination logic 收口在 `termination/`，上下文回顾优化收口在 `retrospect/`，`storage/` 负责持久化。 |
| `agiwo/llm/` | Model 抽象、Provider 适配器、配置策略、消息/事件归一化，以及统一的 model factory。 |
| `agiwo/tool/` | Tool 抽象、最小执行上下文、builtin tools、后台进程 registry（`process/`），以及工具侧存储（如 citation）。 |
| `agiwo/scheduler/` | Agent 之上的编排层。`scheduler.py` 是 facade 与 loop lifecycle，`engine.py` 是唯一编排 owner，`runner.py` 负责单次 dispatch action 执行，`commands.py` 承载调度动作与 tool DTO，`runtime_state.py` 承载进程内 live state 与 tick helpers，`tool_control.py` 收口 child/sleep/cancel 的 tool-facing control，`runtime_tools.py` 是注入给 agent 的 scheduler runtime tools，`store/` 只负责持久化。 |
| `agiwo/observability/` | Trace/Span 模型、查询接口与 trace storage 实现；agent 事件到 Trace 的适配层收口在 `agiwo/agent/trace_writer.py`。 |
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
| `console/server/services/` | 应用服务层。`runtime/`（agent factory、runtime cache、session runtime / session service）、`tool_catalog/`（tool reference / catalog / runtime builder）、`agent_registry/`（配置 CRUD + store 子包）、`runtime_config.py`（运行时全局配置查看/覆盖）、`storage_wiring.py`（存储 config builders）、`metrics.py`。 |
| `console/server/models/` | Console 数据模型目录。`view.py` 只放 API/SSE 视图模型；`session.py`、`agent_config.py`、`runtime_config.py`、`metrics.py` 放共享运行时/配置/聚合模型。不要再新增 `schemas.py` 或平级 `domain/`。 |
| `console/server/channels/` | 渠道适配层，负责批处理、消息解析、delivery，以及 Feishu 等渠道集成。 |
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
- **`id` 必须稳定**：在 Console 这类跨请求复用会话的场景里，每次构造 `Agent` 都必须传稳定 `id`（如 registry `config.id`），否则历史 steps 会丢。
- `AgentConfig.allowed_skills` 进入 SDK runtime 前必须已经展开成“显式 skill 名列表”或 `None`；不再接受 wildcard/pattern。
- 对外执行原语是 `start(...)` 返回 live execution handle；`run(...)` / `run_stream(...)` 只是便利封装。
- 嵌套 agent 执行是内部协议，由 `nested/agent_tool.py` 通过 `Agent.run_child(...)` 进入；不要再暴露公开 `context` 参数。
- `SessionRuntime` 是 session 级 owner；`RunContext` 组合 immutable identity 与 mutable ledger。运行时状态变更优先通过 `runtime/state_ops.py` / `runtime/step_committer.py` 收口。
- `StepRecord` 使用工厂方法创建，不要直接构造。
- `UserMessage` 是 `UserInput` 的 canonical structured owner；input normalization 和 storage encoding/decoding 统一收口在它本身。
- 纯数据模型统一放 `models/`，不要再新增顶层 `*_types.py`、one-model 文件或垃圾桶式 `types.py`。

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
- plain tool 只看 `agiwo.tool.context.ToolContext`；nested-agent runtime bridge 由 `agiwo.agent.nested.context.AgentToolContext` 内部承载，不要把 `SessionRuntime` 再泄漏回通用工具边界。
- agent 运行时内部统一通过 `AgentRuntimeTool` 执行工具；scheduler 控制型 tools 走 runtime tool 契约，不再把终止控制塞进 `ToolResult`。
- `AgentTool` / `as_tool()` 属于 `agiwo.agent.nested.agent_tool`，并由 `Agent.as_tool()` 暴露；它是 agent runtime adapter，不属于 `agiwo.tool/` core。
- 生产代码统一通过 `ToolResult.success()/failed()/aborted()/denied()` 构造结果。
- builtin tools 放在 `agiwo/tool/builtin/`，通过 `@builtin_tool(...)` 注册；`@default_enable` 控制默认自动启用。
- `bash` 与 `bash_process` 是分离工具；后台任务的巡检/日志/停止/输入属于 `bash_process`。
- 共享 MEMORY 检索统一通过 `agiwo.memory.WorkspaceMemoryService`，builtin retrieval tool 只是 adapter。
- citation 等工具侧持久化在 `agiwo/tool/storage/citation/`。

### Skills

- 全局 skill 发现目录只由 SDK `settings.skills_dirs` 决定；`skills_dirs` 不再是 per-agent 选项。
- `enable_skill`、Agent 级 `skills_dirs` 等历史 skill 开关/目录字段已删除；发现旧字段时直接 fast-fail，不保留静默兼容层。
- `agiwo/skill/allowlist.py` 是唯一的 allowlist 归一化/展开/校验入口：pattern 展开、显式 skill 名合法性校验、runtime 前的“必须已展开”校验都收口在这里。
- 只有最外层输入面可以接收 skill pattern：Console env/default-agent 配置和 Agents API。进入内部模型后，只允许显式 skill 名列表。
- `Agent`、`AgentConfigRecord`、scheduler child override、`spawn_agent.allowed_skills` 都只接受“展开后的显式 skill 名列表”；未知 skill 名和 pattern 都必须在边界被拒绝。
- 子 Agent 的 `allowed_skills` 必须是父 Agent `allowed_skills` 的子集；如果父 Agent 未限制 skills，child 才能继承 `None` 语义。
- `SkillManager` 的全局实例 key 由 `root_path + skills_dirs` 决定；`settings.skills_dirs` 变化时，后续 `get_global_skill_manager()` 会切到新的 manager。

### Scheduler

- `Scheduler` 是 Agent 之上的编排层；依赖方向保持 `scheduler -> agent`。
- 当前公开编排接口包括：`run`、`submit`、`enqueue_input`、`route_root_input`、`stream`、`wait_for`、`steer`、`cancel`、`shutdown`，以及查询面 `list_states`、`list_events`、`get_stats`、`rebind_agent`。
- `Scheduler` 只做 facade 和 lifecycle；所有编排语义统一收口到 `SchedulerEngine`。
- `SchedulerEngine` 是唯一编排 owner；`SchedulerRunner` 只负责单次执行；`TaskGuard` 是 spawn/wake 的唯一护栏入口。
- scheduler 状态现在显式区分 `WAITING`、`IDLE`、`QUEUED`；不要再把待命/排队语义塞回一个泛化 `SLEEPING`。
- 当前唤醒路径包括 `WAITSET`、`TIMER`、`PERIODIC`、`PENDING_EVENTS`。
- scheduler state storage 当前支持 memory/sqlite/mongodb，由 `Scheduler` 自己创建和持有。
- `Scheduler` 的外部流式协议直接复用 `AgentStreamItem`；不要在 SDK core 再维护第二套 live-output protocol。
- `route_root_input` 对所有非 RUNNING 路径（submitted / enqueued / steered+WAITING / steered+QUEUED）都返回带 stream 的 `RouteResult`；只有 steered+RUNNING 不带 stream（原 stream subscriber 仍在消费）。下游不需要为 steered 场景单独维护 deferred reply 补丁。

### Context Optimization

- Scheduler 编排场景下有两个独立的上下文优化机制：Context Rollback（空转回退）和 Tool Result Retrospect（工具结果回顾）。详见 `docs/guides/context-optimization.md`。
- Context Rollback 通过 `sleep_and_wait(no_progress=True)` 触发，删除空转轮次的所有 steps；仅 PERIODIC 唤醒有效。
- Tool Result Retrospect 收口在 `agiwo/agent/retrospect/` 包中；`run_loop.py` 只通过 `RetrospectBatch` 交互，不直接依赖 `triggers` / `executor` 子模块。
- `RetrospectBatch` 是 per-batch lifecycle 对象，三个方法：`process_result()` → `register_step()` → `finalize()`。`finalize()` 返回 `RetrospectOutcome`，调用方显式 `replace_messages` 应用上下文变更。
- Retrospect 按触发类型（`LARGE_RESULT` / `ROUND_INTERVAL` / `TOKEN_ACCUMULATED`）注入不同内容的 `<system-notice>`，不使用统一模板。
- `RetrospectToolResultTool` 是 scheduler runtime tool；不依赖 magic flag，通过 `tool_name` 识别。
- `retrospect/` 不依赖 `scheduler/`，保持 `scheduler -> agent` 单向依赖。
- `StepRecord.condensed_content` 记录精简内容，`content` 永远保留原始数据。加载历史时 `condensed_content` 优先。

### Storage & Observability

- Run/Step 持久化通过 `AgentOptions.run_step_storage` 配置；Trace 持久化通过 `AgentOptions.trace_storage` 配置。
- Session/compact metadata 存储与 run-step storage 分离，位于 `agiwo/agent/storage/session.py`。
- Agent 运行记录与流式事件通过 session runtime 内部执行链统一提交；`handle.stream()` / `run_stream()` 对外暴露的是 typed `AgentStreamItem`。
- `BaseTraceStorage` 既支持查询，也支持实时订阅，Console trace SSE 会用到这个能力。

## Architecture Boundaries

### SDK

- Python 版本基线是 3.11+。
- 所有公开方法与核心数据结构都应带类型注解。
- async 逻辑保持显式，不要把 `await` 藏进难追踪的 helper。
- 哨兵判断优先用 `is not None`；truthy 检查只在语义明确时使用。
- 顶层模块名只保留稳定入口与核心 orchestrator；领域性 helper 进入具名子包（如 `models/`、`runtime/`、`nested/`）。
- 模块名优先表达职责，避免新增 `helpers.py`、`utils.py`、`misc.py`、`pipeline.py` 这类弱语义文件名。
- 不为“可能以后有用”保留遗留兼容层；除非明确要求，否则直接删除旧路径。
- 遇到循环依赖优先重构依赖方向，不要靠局部导入规避。
- 同一外部 use case 不得并列暴露两套 public API；内部可以保留多种 lifecycle mutation 动作，但 facade 必须内化复杂度。

### Console

- `console/server/models/view.py` 与 `console/server/response_serialization.py` 只属于 API/SSE 边界。
- 共享的 Console 数据模型统一放 `console/server/models/`；不要再新增 `schemas.py`、`schema.py` 或平级 `domain/` 目录。
- `console/server/models/session.py` 中的 `Session` 只表示完整会话线程元数据；它不再承载 runtime agent id、scheduler state id、task id 或消息计数。`Session.id` 直接作为 root persistent scheduler state id 使用。
- `ChannelChatContext.current_session_id` 是 channel 专用的“当前会话指针”；web console 不使用 server-side current session 语义。
- Console 自己持有 API/表单 DTO；不要再把 `Input/Patch` 请求模型塞回 SDK。
- Console web 现在提供 `/settings` 全局配置页；它查看的是“当前运行进程的 effective config”，不是 `.env` 文件内容。
- `console/server/services/runtime_config.py` 只负责运行时 override，不负责持久化 env。当前只允许热更新 `skills_dirs` 和 `default_agent`；其余配置只读展示并标记“需重启生效”。
- 运行时配置更新必须先校验，再整体应用；如果 `skills_dirs` 或默认 Agent skill/model 参数校验失败，必须回滚，避免半更新状态。
- `console/server/channels/batch_manager.py` 只承载渠道批处理/防抖能力；`ChannelBatchManager` 负责按 channel context 聚合消息，不承担 Session CRUD 语义。渠道代码直接依赖 `server.models.session`，不要再引入 compatibility shim。
- 渠道运行时由三个独立子服务组成：`SessionContextService`（channel session/chat-context 生命周期）、`AgentRuntimeCache`（按 `session.id` 缓存 runtime agent，并在 config 指纹变化时刷新）、`SessionRuntimeService`（scheduler 交互与状态路由）。Channel service 直接持有这三者，不再经过 facade。
- Console web 与 Feishu channel 的会话输入统一走 `scheduler.route_root_input(...)`；web console 已不再保留独立 `Chat` 执行语义。
- `SessionRuntimeService.execute()` 只委托 `scheduler.route_root_input(...)` 做 root 输入路由，且固定使用 `state_id=session.id`、`session_id=session.id`、`persistent=True`；Console 不再自己编码 scheduler 状态机。timeout 为可选参数，默认不设硬超时。
- `SessionRuntimeService.execute()` 统一承载 session 输入路由；当 Console/Web 需要同一 root session 在 `sleep -> wake -> rerun` 后继续沿用当前 HTTP stream 时，通过 `scheduler.route_root_input(..., stream_mode="until_settled")` 实现，不再额外扩展 follow-only API。
- web console 只使用显式 `session_id`：不会调用 server-side `switch_session`，也不会隐式恢复“当前 session”。channel 才有 `/switch` / `current_session_id` 语义。
- Channel 消息管线是线性流程：consume stream → deliver output → fallback。SDK `route_root_input` 在 steered+WAITING/QUEUED 时已返回 stream，channel 层不需要 deferred reply 补丁。steered+RUNNING 时无 stream，channel 只发 ack（原 stream subscriber 仍在消费）。
- Feishu store 实现收口到 `console/server/channels/feishu/store/` 包：`__init__.py`（protocol + factory）、`memory.py`（内存实现）、`sqlite.py`（SQLite 实现）。
- Feishu/Session SQLite 元数据存储是破坏性 schema 演进：不写 migration、不自动补列。字段变化时直接 fast-fail，并要求清理旧 session 元数据。
- Feishu 模块约定：`message_parser.py` 只保留 envelope 类型与解析 facade；sender 解析、attachment 解析、delivery、连接管理放在各自具名模块。
- Console 工具的展示、解析、组装都必须经过 `ConsoleToolCatalog`。
- **`build_agent` 必须传入稳定 `id`**：`services/runtime/agent_factory.py` 中的 `build_agent()` 构造 Agent 时必须使用 `id=id or config.id`，确保同一个 agent config 在不同请求中拿到相同的 `agent_id`。若遗漏此 fallback，每次 HTTP 请求都会产生随机 agent id，导致 `run_step_storage.get_steps(session_id, agent_id)` 查不到历史步骤，对话上下文无法延续。`repo_guard.py` 中有对应的 `AGW043` 检查。
- Agent 配置写入保持 full replace；不要回到 partial merge / patch DTO 语义。
- Agent registry 收口到 `console/server/services/agent_registry/` 包：`registry.py`（CRUD 服务）、`models.py`（AgentConfigRecord）、`store/`（protocol + factory + memory/sqlite 实现）。

### Configuration

- SDK 配置入口在 `agiwo/config/settings.py`；Console 专属部署/渠道配置入口在 `console/server/config.py`。
- 项目自有环境变量命名空间保持 `AGIWO_*` 与 `AGIWO_CONSOLE_*`；外部 Provider 继续使用其标准变量名。
- 业务模块不要新增散落的 `os.getenv(...)`；环境变量读取应集中在配置层，当前唯一例外是 `agiwo.llm.factory` 根据 `api_key_env_name` 做运行时密钥解析。
- `skills_dirs` 是 SDK 全局配置，不是 Console Agent 配置字段。Console 侧的 Agent/default-agent 只配置 `allowed_skills`，不直接配置 skill 目录。
- Console runtime config API / 页面修改的是进程内 override；重启后仍以环境配置为准，当前不写回 `.env`。

### Construction & Layering

- builtin tools 应自行按配置构建 model/HTTP/storage 依赖，除非它本质上是在包装宿主运行时对象。
- LLM 创建统一走共享 factory / config policy。
- 不要在 agent runtime 外越权依赖 `agiwo.agent.run_loop`、`agiwo.agent.tool_executor`、`agiwo.agent.prompt`、`agiwo.agent.runtime` 等内部模块；统一依赖 `agiwo.agent` 公共 API 或稳定边界模块。
- `agiwo.agent.types` 是 public facade；agent 内部实现优先直接依赖 `models/`、`hooks/`、`runtime/` 这些 focused 模块，不要再把内部依赖都堆回 facade。
- 不要在 Console 或其他集成侧直读 `scheduler.store`；统一走 `Scheduler` facade 的查询 API。
- scheduler runtime tools 直接依赖 `SchedulerEngine` 的 tool-facing 方法；不要重新引入 `SchedulerControl` protocol。
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
- `run_loop.py` 只能通过 `RetrospectBatch` 使用 retrospect，不得直接 import `triggers` / `executor` 子模块（AGW045）
- 不要使用 `_retrospect` magic flag 做模块间信号传递（AGW046）
- `retrospect/` 模块不得依赖 `agiwo.scheduler`（AGW047）
- `run_loop.py` 不得直接操作 `ledger.last_retrospect_seq` / `retrospect_pending_tokens` / `retrospect_pending_rounds`（AGW048）
- agent 内部其他模块只能 import `agiwo.agent.retrospect` 公开 API，不得直接 import `triggers` / `executor`（AGW049）

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

# Console 前端测试 / lint / build
(cd console/web && npm test)
(cd console/web && npm run lint)
(cd console/web && npm run build)
```

类型检查不是当前仓库默认门槛；只有分支实际接入某个 checker 时，才运行对应工具。

## Common Changes

#### 添加新 LLM Provider

1. 在 `agiwo/llm/` 实现 Provider 类。
2. 在 `agiwo/config/settings.py` 注册 provider literal / 默认配置。
3. 在 `agiwo/llm/factory.py` 注册 `ProviderSpec`。
4. 如需公开导出，在 `agiwo/llm/__init__.py` 增补导出。
5. 在 `tests/llm/` 补测试。

#### 添加新 Hook

1. 在 `agiwo/agent/hooks/__init__.py` 增加 hook 类型与字段。
2. 在 `agiwo/agent/agent.py`、`agiwo/agent/run_loop.py` 或相关运行时模块接线。

#### 添加新 Builtin Tool

1. 在 `agiwo/tool/builtin/` 下新增实现。
2. 用 `@builtin_tool(...)` 注册；仅在确实需要默认启用时再加 `@default_enable`。
3. 如果 Console 侧需要自定义展示文案或 build-time 依赖，在 `console/server/services/tool_catalog/` 增补对应 catalog / builder 逻辑。

### Notes

- `agiwo.agent` 是当前唯一 canonical agent 包；不要新增对已废弃迁移路径的引用。
- `StepRecord` 的结构性变更继续通过共享工厂与 recorder 管线驱动；不要在包外直接拼装内部执行状态。
- `TYPE_CHECKING` 只用于确实无法通过重构解决的类型环依赖。
- Tool runtime 缓存是 session 级缓存，只有 `tool.cacheable = True` 才生效。
- Anthropic 有独立实现路径；只有显式 `anthropic-compatible` 场景才按兼容协议处理。

## Maintaining AGENTS.md

- 当“目录职责”、“公开 API”、“机器护栏规则”、“标准开发流程”发生变化时更新本文件。
- 继续保持目录/包级别描述，不要退回到逐文件目录树。
- 如果某条说明开始频繁失真，优先上移抽象层级，而不是继续堆更多细节。
