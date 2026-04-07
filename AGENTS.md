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
| `agiwo/agent/` | Canonical agent runtime。public API 只从 `agiwo.agent` 暴露；顶层只保留稳定入口与核心 orchestrator（如 `agent.py`、`definition.py`、`run_loop.py`、`llm_caller.py`、`tool_executor.py`、`prompt.py`、`trace_writer.py`）。纯数据模型收口在 `models/`，hook contract 收口在 `hooks/`，nested-agent adapter 收口在 `nested/`，run/session runtime context 与 state helper 收口在 `agiwo.agent.runtime`，termination logic 收口在 `termination/`，上下文回顾优化收口在 `retrospect/`，`storage/` 负责持久化。 |
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
| `console/server/services/` | 应用服务层。`runtime/`（agent factory、runtime cache、session runtime / session service、scheduler tree view）、`tool_catalog/`（tool reference / catalog / runtime builder）、`agent_registry/`（配置 CRUD + store 子包）、`runtime_config.py`（运行时全局配置查看/覆盖）、`storage_wiring.py`（存储 config builders）、`metrics.py`。 |
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
- 公开构造入口是 `Agent(config: AgentConfig, *, model: Model, tools: list[BaseTool] | None = None, hooks: AgentHooks | None = None, id: str | None = None)`；`AgentConfig` 只承载纯配置，不放 live object。
- **`id` 必须稳定**：在 Console 这类跨请求复用会话的场景里，每次构造 `Agent` 都必须传稳定 `id`（如 registry `config.id`），否则历史 steps 会丢。不传 id 时自动生成 `{name}-{hex[:6]}` 格式。
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

- Context Rollback 通过 `sleep_and_wait(no_progress=True)` 触发，删除空转轮次。
- Tool Result Retrospect 由 `agiwo/agent/retrospect/` 处理，`run_loop.py` 只通过 `RetrospectBatch` 交互。
- `StepRecord.condensed_content` 记录精简内容，加载历史时优先。

### Storage & Observability

- Run/Step 持久化通过 `AgentOptions.run_step_storage` 配置；Trace 持久化通过 `AgentOptions.trace_storage` 配置。
- Agent 运行记录通过 session runtime 统一提交；`handle.stream()` / `run_stream()` 对外暴露 `AgentStreamItem`。
- `BaseTraceStorage` 支持查询和实时订阅。

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
- 共享的 Console 数据模型统一放 `console/server/models/`。
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

- 不要在 agent runtime 外越权依赖 `agiwo.agent.run_loop`、`agiwo.agent.tool_executor` 等内部模块。
- `agiwo.agent.types` 已标记为 deprecated，优先从 `agiwo.agent` 直接导入。
- 不要在 Console 侧直读 `scheduler.store`；统一走 `Scheduler` facade 的查询 API。

## Lint & Guardrails

- Python 代码改动后默认执行：`uv run python scripts/lint.py changed`
- 提交前必须跑一遍与 CI 对齐的 lint：
  1. `uv run ruff check --ignore C901 --ignore PLR0911 --ignore PLR0912 agiwo/ console/server/ tests/ console/tests/ scripts/`
  2. `uv run ruff format --check agiwo/ console/server/ tests/ console/tests/ scripts/`
  3. `uv run python scripts/lint.py imports`
  4. `uv run python scripts/repo_guard.py`
- 不要写 schema migration；数据模型变更时直接让旧数据失败并通知用户清理。

## Build & Test Commands

```bash
# 安装依赖
uv sync

# 标准检查
uv run python scripts/lint.py changed

# 提交前 lint 四步
uv run ruff check --ignore C901 --ignore PLR0911 --ignore PLR0912 agiwo/ console/server/ tests/ console/tests/ scripts/
uv run ruff format --check agiwo/ console/server/ tests/ console/tests/ scripts/
uv run python scripts/lint.py imports
uv run python scripts/repo_guard.py

# SDK 测试
uv run pytest tests/ -v

# Console 后端测试
cd console && uv run pytest tests/ -v
```


## Maintaining AGENTS.md
- 当“目录职责”、“公开 API”、“机器护栏规则”、“标准开发流程”发生变化时更新本文件。
- 继续保持目录/包级别描述，不要退回到逐文件目录树。
- 如果某条说明开始频繁失真，优先上移抽象层级，而不是继续堆更多细节。