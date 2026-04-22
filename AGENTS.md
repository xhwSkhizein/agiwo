# AGENTS.md

> Agiwo 开发指南。供 AI 编码助手和开发者快速理解当前仓库结构、边界和约定。
>
> 本文件只维护“目录级职责 + 稳定 API/边界”，不再逐个文件镜像源码；实现细节以源码为准。

## How To Use This File

- 如果文档与源码冲突，以源码为准，并顺手更新本文件。
- 优先用本文件定位“应该去哪个目录/模块看”，不要把它当成逐文件索引。
- 如果某条信息很难长期与源码同步，就应该提升抽象层级或直接删掉。

## Logging

统一日志格式：`logger.{level}("event_name", key=value, ...)`. 详见 [docs/logging-guidelines.md](docs/logging-guidelines.md).

## Config Hot Reload

配置热更仅在 agent 为 IDLE/COMPLETED/FAILED 状态时生效，运行中 agent 配置延迟生效。详见 [docs/config-hot-reload.md](docs/config-hot-reload.md).

## Repository Layout

### SDK (`agiwo/`)

| Path | Responsibility |
| --- | --- |
| `agiwo/agent/` | Canonical agent runtime。public API 只从 `agiwo.agent` 暴露；顶层只保留稳定入口与核心 orchestrator（如 `agent.py`、`definition.py`、`run_loop.py`、`llm_caller.py`、`tool_executor.py`、`prompt.py`、`trace_writer.py`）。纯数据模型收口在 `models/`，hook contract 收口在 `agiwo.agent.hooks`，nested-agent adapter 收口在 `nested/`，run/session runtime context 与 state helper 收口在 `agiwo.agent.runtime`，termination logic 收口在 `termination/`，上下文回顾优化收口在 `retrospect/`，`storage/` 负责持久化。`run_loop.py` 使用 `RunLoopOrchestrator` 类封装运行循环逻辑，消除多层嵌套。 |
| `agiwo/llm/` | Model 抽象、Provider 适配器、配置策略、消息/事件归一化，以及统一的 model factory。 |
| `agiwo/tool/` | Tool 抽象、最小执行上下文、builtin tools、后台进程 registry（`process/`），以及工具侧存储（如 citation）。 |
| `agiwo/scheduler/` | Agent 之上的编排层。`scheduler.py` 是 facade 与 loop lifecycle，`engine.py` 是唯一编排 owner，`runner.py` 负责单次 dispatch action 执行，`commands.py` 承载调度动作与 tool DTO，`runtime_state.py` 承载进程内 live state 与 tick helpers，`tool_control.py` 收口 child/sleep/cancel 的 tool-facing control，`runtime_tools.py` 是注入给 agent 的 scheduler runtime tools，`store/` 只负责持久化。`runner.py` 使用策略表驱动 output-handling 链，消除 chained responsibility。 |
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
| `console/server/services/` | 应用服务层。`runtime/`（agent factory、runtime cache、session runtime / session service、scheduler tree view）、`tool_catalog/`（tool reference / catalog / runtime builder）、`agent_registry/`（配置 CRUD + store 子包）、`session_store/`（Console 会话存储工厂与实现）、`runtime_config.py`（运行时全局配置查看/覆盖）、`storage_wiring.py`（存储 config builders）、`metrics.py`。 |
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
- 公开构造入口是 `Agent(config: AgentConfig, *, model: Model, tools: list[BaseTool] | None = None, hooks: HookRegistry | list[HookRegistration] | None = None, id: str | None = None)`；`hooks` 统一走 `agiwo.agent.hooks` 中的 phase-based registry；`tools` 是功能/用户自定义工具（受 `allowed_tools` 过滤），`system_tools` 是系统工具（不受 `allowed_tools` 过滤）；`AgentConfig` 只承载纯配置，不放 live object。
- **`id` 必须稳定**：在 Console 这类跨请求复用会话的场景里，每次构造 `Agent` 都必须传稳定 `id`（如 registry `config.id`），否则历史 steps 会丢。不传 id 时自动生成 `{name}-{hex[:6]}` 格式。
- `AgentConfig.allowed_skills` 进入 SDK runtime 前必须已经展开成“显式 skill 名列表”或 `None`；不再接受 wildcard/pattern。
- 对外执行原语是 `start(...)` 返回 live execution handle；`run(...)` / `run_stream(...)` 只是便利封装。
- 嵌套 agent 执行是内部协议，由 `nested/agent_tool.py` 通过 `Agent.run_child(...)` 进入；不要再暴露公开 `context` 参数。
- `SessionRuntime` 是 session 级 owner；`RunContext` 组合 immutable identity 与 mutable ledger。运行时状态变更优先通过 `runtime/state_ops.py` / `runtime/step_committer.py` / `runtime/state_writer.py` 收口。
- `StepView` 是唯一的已提交 step 视图与运行时 step 数据模型；统一通过 `StepView.user()/assistant()/tool()` 创建，不要再引入第二套 step 记录模型。
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

- 工具分为两类：**功能工具**（受 `allowed_tools` 控制）和**系统工具**（不受 `allowed_tools` 控制）。
  - **功能工具**：builtin tools（bash、web_search 等）和用户自定义工具（AgentTool、自定义 BaseTool）。
  - **系统工具**：SkillTool（受 `allowed_skills` 控制）和 Scheduler runtime tools（通过 `system_tools` 注入）。
- `allowed_tools: list[str] | None` 控制功能工具：`None` = 默认 builtin + 所有 extra；`[]` = 无功能工具；显式列表同时过滤 builtin 和 extra_tools。
- `ToolManager.get_tools()` 接受 `system_tools` 参数，无条件注入，不受 `allowed_tools` 约束。
- `ToolManager.parse_allowed_tools()` 接受 `list[str] | None`，返回类型安全的 `list[ToolReference] | None`，支持解析 `agent:` 前缀的 agent tool 引用。
- `BaseTool` 定义稳定契约：名称、描述、参数 schema、并发安全性、可选 `gate(..., context: ToolContext) -> ToolGateDecision` 预检，以及 `execute(..., context: ToolContext) -> ToolResult`。
- plain tool 只看 `agiwo.tool.context.ToolContext`；nested-agent runtime bridge 由 `agiwo.agent.nested.context.AgentToolContext` 内部承载，不要把 `SessionRuntime` 再泄漏回通用工具边界。
- agent 运行时内部统一通过 `AgentRuntimeTool` 执行工具；scheduler 控制型 tools 走 runtime tool 契约，不再把终止控制塞进 `ToolResult`。
- `AgentTool` / `as_tool()` 属于 `agiwo.agent.nested.agent_tool`，并由 `Agent.as_tool()` 暴露；它是 agent runtime adapter，属于功能工具，受 `allowed_tools` 约束。
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
- Scheduler runtime tools（`SpawnAgentTool`、`SleepAndWaitTool` 等）通过 `runtime_agent.inject_system_tools(...)` 注入，不混入 `tools`（extra_tools），不受 `allowed_tools` 约束。
- 子 Agent 的 system_tools 由 `SchedulerRunner` 从父 Agent 的 `system_tools` 派生；非 fork 模式排除 `spawn_agent`，fork 模式继承全部（gate 检查仍阻止实际 spawn）。
- 当前公开编排接口包括：`route_root_input`（统一入口）、`enqueue_input`、`wait_for`、`steer`、`cancel`、`shutdown`，以及查询面 `list_states`、`list_events`、`get_stats`、`rebind_agent`。`submit`、`run`、`stream` 已合并到 `route_root_input` 或改为内部方法。
- `Scheduler` 只做 facade 和 lifecycle；所有编排语义统一收口到 engine 层（facade 直接委托给 `_tick`、`_stream`、`_tree_ops` 等同包 helper）。
- `SchedulerRunner` 只负责单次 dispatch action；`TaskGuard` 是 spawn/wake 的唯一护栏入口。
- scheduler 状态现在显式区分 `WAITING`、`IDLE`、`QUEUED`；不要再把待命/排队语义塞回一个泛化 `SLEEPING`。
- 当前唤醒路径包括 `WAITSET`、`TIMER`、`PERIODIC`、`PENDING_EVENTS`。
- scheduler state storage 当前支持 memory/sqlite/mongodb，由 `Scheduler` 自己创建和持有。
- `Scheduler` 的外部流式协议直接复用 `AgentStreamItem`；不要在 SDK core 再维护第二套 live-output protocol。
- `route_root_input` 对所有非 RUNNING 路径（submitted / enqueued / steered+WAITING / steered+QUEUED）都返回带 stream 的 `RouteResult`；只有 steered+RUNNING 不带 stream（原 stream subscriber 仍在消费）。下游不需要为 steered 场景单独维护 deferred reply 补丁。
- **Root runtime 严格复用**：`_ensure_root_runtime_agent` 以"canonical Agent 身份"为键缓存 scheduler-managed runtime agent；`_submit` / `enqueue_input(agent=same)` 不再每次 clone，persistent 会话下 `run_log_storage` / `trace_storage` / workspace 贯穿所有轮次。只有 `rebind_agent` 或传入新 canonical agent 才会 close 旧 runtime 并重建。
- **非 persistent root 自动收口**：`_cleanup_after_run` 对非 persistent 且进入 terminal 的 root 立即 `pop + await close`；persistent root 与 child 仍按原语义。`Scheduler.stop()` 在关闭前会批量 close 所有残留 runtime agent。
- **Cancel/Shutdown 写终态结果**：`cancel_subtree` / `shutdown_subtree` 写 `with_failed(...)` 时一并写入 `last_run_result=SchedulerRunResult(termination_reason=CANCELLED, ...)`；`wait_for()` 依赖该字段，能够在 cancel 完成那一刻直接返回 `RunOutput(error, termination_reason=CANCELLED)` 而非等到 timeout。runner 只在终态未写 `last_run_result` 时兜底补一条。
- **Urgent steer bypass**：`PendingEvent.urgent` 标志位让 `plan_tick` 在 WAITING 分支跳过 `event_debounce_*` 限制；`Scheduler.steer(..., urgent=True)` 据此令 WAITING root 立即 WAKE_EVENTS。非 urgent 事件继续遵循 debounce。
- **结构化 `UserInput` 贯通**：`Scheduler.steer()` 不再 `extract_text()`；`PendingEvent.USER_HINT payload` 统一写 `{"user_input": UserMessage.to_storage_value(...)}`；`build_mailbox_input` / `build_events_message` 返回 `UserMessage`，保留 `ContentPart`（图片/文件等）与 `ChannelContext`（Feishu 等 channel metadata）。

### Context Optimization

- Context Rollback 通过 `sleep_and_wait(no_progress=True)` 触发，删除空转轮次。
- Tool Result Retrospect 由 `agiwo/agent/retrospect/` 处理，`run_loop.py` 只通过 `RetrospectBatch` 交互。
- `StepView.condensed_content` 记录精简内容，加载历史时优先。

### Storage & Observability

- Agent 运行记录的 canonical persistence 是 `AgentOptions.storage.run_log_storage`；storage 层以 append-only `RunLog` entries 为真相源，并从中重建 `RunView` / `StepView` 查询结果。
- `SessionRuntime` 统一负责 sequence 分配、run-log 追加，以及把同一批 runtime facts 喂给 trace writer；`trace`/`stream` 应感知 `compaction`、`retrospect`、`termination` 并保持可记录、可重放。
- Agent 运行记录通过 session runtime 统一提交；流式输出通过 `Agent.start()` 返回的 handle 暴露 `AgentStreamItem`。
- `BaseTraceStorage` 支持查询和实时订阅。

## Architecture Boundaries

### SDK

- Python 版本基线是 3.10+。
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
- 共享的 Console 数据模型统一放 `console/server/models/`。
- `agiwo-console` 现在有两条稳定启动面：`serve`（宿主机模式）和 `container ...`（Docker 托管模式）。Docker 模式对外只暴露一个公开端口 `8422`，默认持久化根在容器内 `/data/root`，宿主机目录只有通过显式 `--mount <source>:<alias>` 才会映射到 `/mnt/host/<alias>`。
- `Session.id` 直接作为 root persistent scheduler state id 使用。
- Console web 与 Feishu channel 统一走 `scheduler.route_root_input(...)`。
- `build_agent` 必须传入稳定 `id`，使用 `id=id or config.id` 确保上下文延续。
- Agent registry 收口到 `console/server/services/agent_registry/` 包。

### Configuration

- SDK 配置入口在 `agiwo/config/settings.py`；Console 专属部署/渠道配置入口在 `console/server/config.py`。
- 项目自有环境变量命名空间保持 `AGIWO_*` 与 `AGIWO_CONSOLE_*`；外部 Provider 继续使用其标准变量名。
- 业务模块不要新增散落的 `os.getenv(...)`；环境变量读取应集中在配置层，当前唯一例外是 `agiwo.llm.factory` 根据 `api_key_env_name` 做运行时密钥解析。
- `skills_dirs` 是 SDK 全局配置，不是 Console Agent 配置字段。Console 侧的 Agent/default-agent 只配置 `allowed_skills`，不直接配置 skill 目录。
- Console runtime config API / 页面修改的是进程内 override；重启后仍以环境配置为准，当前不写回 `.env`。

### Construction & Layering

- `agiwo.agent` 是公共 facade，外部代码（Console、第三方集成）必须从此导入类型。只有 agent 内部模块（如 `models/`、`hooks/`、`runtime/`）可绕过 facade 直接依赖 `agiwo.agent.run_loop`、`agiwo.agent.tool_executor` 等内部模块。
- 不要在 Console 侧直读 `scheduler.store`；统一走 `Scheduler` facade 的查询 API。

## Lint & Guardrails

- Python 代码改动后，Agent 必须主动运行与改动范围匹配的检查；不要把 `pre-commit` 当作主要反馈回路。
- 开发中的低噪音快速检查继续使用：`uv run python scripts/lint.py changed`
- 与 CI Lint job 对齐的轻量本地门禁统一使用：`uv run python scripts/lint.py ci`
- 安装仓库内 git hooks：`uv run python scripts/install_git_hooks.py`
- `pre-commit` 会自动运行 `uv run python scripts/lint.py ci`，本地未通过时禁止提交。
- `pre-push` 会自动运行 `uv run python scripts/check.py pre-push`，作为推送前兜底门禁；不要等到 push 时才第一次发现问题。
- `scripts/lint.py changed` 不包含 `ruff format --check`，只能用于开发中的快速回路，不能视为 CI lint 全通过。
- `scripts/check.py console-tests` 会强制设置 `AGIWO_ROOT_PATH="$(pwd)/console/.agiwo"`，避免本地 `console/.env` 污染测试结果。
- Required checks by change type:
  - 只改 Python 代码：至少运行 `uv run python scripts/lint.py ci`，并运行受影响测试。
  - 改 `pyproject.toml`、workflow、打包、发布脚本：除上面外，还要跑构建与 smoke install。
  - 改 Console 后端：除上面外，还要跑 `uv run python scripts/check.py console-tests`
  - 改前端：运行 `cd console/web && npm run lint && npm test && npm run build`
- 不要写 schema migration；数据模型变更时直接让旧数据失败并通知用户清理。

## Build & Test Commands

```bash
# 安装依赖
uv sync
(cd console && uv sync)

# 安装仓库内 git hooks
uv run python scripts/install_git_hooks.py

# 开发中的低噪音快速检查
uv run python scripts/lint.py changed

# 提交前轻量门禁（与 CI Lint job 对齐）
uv run python scripts/lint.py ci

# SDK 测试
uv run pytest tests/ -v

# Console 后端测试
uv run python scripts/check.py console-tests

# 推送前完整本地门禁
uv run python scripts/check.py pre-push

# Console 前端检查
(cd console/web && npm run lint)
(cd console/web && npm test)
(cd console/web && npm run build)

# Console Docker smoke
uv run python scripts/smoke_console_docker.py

# 打包 / workflow / 发布脚本改动后追加
uv build
uv run python scripts/smoke_release_install.py dist/agiwo-0.1.0-py3-none-any.whl
(cd console && uv build)
uv run python scripts/smoke_release_install.py dist/agiwo-0.1.0-py3-none-any.whl console/dist/agiwo_console-0.1.0-py3-none-any.whl
```


## Maintaining AGENTS.md
- 当“目录职责”、“公开 API”、“机器护栏规则”、“标准开发流程”发生变化时更新本文件。
- 继续保持目录/包级别描述，不要退回到逐文件目录树。
- 如果某条说明开始频繁失真，优先上移抽象层级，而不是继续堆更多细节。
- Public repository overview generation consumes repository structure first and may also use `AGENTS.md` as a supporting source；请保持目录职责与稳定边界说明为最新状态。
