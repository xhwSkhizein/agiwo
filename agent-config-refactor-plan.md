# Agent 构造重构方案 — AgentConfig 分层

> 确认时间：2026-03-14
>
> 本文档记录已确认的重构方案，覆盖 AgentConfig 引入、derive_child 改造、storage 隔离、配置模型边界隔离等全部改动。

## 背景

`review_confirm.md` 中的 7 个问题，问题 1-4 已在当前代码中修复：

| 问题 | 状态 | 修复位置 |
|------|------|----------|
| #1 hooks 共享污染 | ✅ 已修复 | `derive_child()` 已用 `copy.deepcopy(self.hooks)` |
| #2 child session_id 不一致 | ✅ 已修复 | `AgentState.resolve_runtime_session_id()` 统一语义 |
| #3 并发安全契约未执行 | ✅ 已修复 | `ToolExecutor.execute_batch()` safe/unsafe 分组 |
| #4 step.tool_name 错误 | ✅ 已修复 | 改为 `step.name`，`except` 加 `logger.warning` |

本次重构聚焦于以下已确认的核心问题：

| 编号 | 问题 | 对应方案 |
|------|------|----------|
| B | `derive_child()` 共享父 agent 的 storage 对象实例 | 方案 1-3 |
| C | `create_agent()` 与 `Agent()` 双入口 | 方案 1-2 |
| D | `Agent.system_prompt` / `_system_prompt` 僵尸字段 | 方案 2 |
| E | `derive_child()` 共享 `_skill_manager` 实例 | 方案 1-3 |
| F | scheduler cleanup 只 unregister 不 close child | 方案 4 |
| G | 配置模型 Input/Patch 侵入 SDK 层 | 方案 5 |

## 核心设计：配置态与运行态分层

`Agent` 当前混合了三类语义：纯配置、agent-local 可变状态、可共享基础设施。重构后显式分为两层：

```
AgentConfig (纯配置，可复制/派生)
    ├── name, description, model
    ├── system_prompt (输入态)
    ├── options, hooks (模板)
    └── tools (用户指定的原始工具 / child 的完整工具)

Agent (运行时实例，由 AgentConfig 构建)
    ├── config: AgentConfig (持有原始配置引用)
    ├── tools (装配后的完整列表)
    ├── hooks (实例独立，deepcopy)
    ├── run_step_storage (实例独立 handle)
    ├── trace_storage (实例独立 handle)
    ├── session_storage (实例独立 handle)
    ├── _skill_manager (实例独立)
    └── _prompt_builder (实例独立)
```

---

## 方案 1：AgentConfig 数据模型

**文件**：`agiwo/agent/config.py`（新建）

```python
from dataclasses import dataclass

from agiwo.agent.hooks import AgentHooks
from agiwo.agent.options import AgentOptions
from agiwo.llm.base import Model
from agiwo.tool.base import BaseTool


@dataclass
class AgentConfig:
    """纯配置，可复制/序列化/派生，不持有运行时对象。"""

    name: str
    description: str
    model: Model
    id: str | None = None              # None → Agent.__init__ 自动生成
    system_prompt: str = ""
    tools: list[BaseTool] | None = None # 用户原始工具 / child 的完整工具
    options: AgentOptions | None = None
    hooks: AgentHooks | None = None
    skip_default_tools_merge: bool = False  # child 派生时设为 True
```

**设计决策**：

- `tools` 对于正常构造是用户原始输入；对于 child 派生（`skip_default_tools_merge=True`）是装配后的完整列表
- `model` 是已构建的 Model 实例（Model 本身无状态，可安全复用）
- `hooks` 是用户提供的模板，Agent 构造时会 `deepcopy` 后使用
- `id` 为 None 时 Agent 自动生成 `{name}-{hex[:6]}`

**导出**：从 `agiwo.agent` 和 `agiwo` 公开导出 `AgentConfig`。

---

## 方案 2：Agent.__init__ 重构

**文件**：`agiwo/agent/agent.py`

### 构造函数

```python
class Agent:
    def __init__(self, config: AgentConfig):
        # 1. 持有原始配置引用
        self.config = config

        # 2. 基础字段
        self.name = config.name
        self.id = config.id or _generate_default_id(config.name)
        self.description = config.description
        self.model = config.model
        self.options = config.options or AgentOptions()

        # 3. 实例独立 hooks（deepcopy 隔离）
        self.hooks = copy.deepcopy(config.hooks) if config.hooks else AgentHooks()

        # 4. 装配工具
        self.tools, self._skill_manager = self._assemble_tools()

        # 5. 注入默认 memory hook
        self._inject_default_memory_hook()

        # 6. 创建实例独立的 storage handles
        self.run_step_storage = StorageFactory.create_run_step_storage(
            self.options.run_step_storage
        )
        self.trace_storage = StorageFactory.create_trace_storage(
            self.options.trace_storage
        )
        self.session_storage = _create_session_storage(self.options)

        # 7. 创建 prompt builder
        self._prompt_builder = DefaultSystemPromptBuilder(
            base_prompt=config.system_prompt,
            agent_name=self.name,
            agent_id=self.id,
            options=self.options,
            tools=self.tools,
            skill_manager=self._skill_manager,
        )
```

### 工具装配

```python
def _assemble_tools(self) -> tuple[list[BaseTool], SkillManager | None]:
    config = self.config
    skill_manager = None

    if config.skip_default_tools_merge:
        # child 派生：tools 已是完整列表，直接使用
        resolved_tools = list(config.tools or [])
    else:
        # 正常装配：merge default tools + ensure bash pair
        resolved_tools = ensure_bash_tool_pair(config.tools or [])
        user_tool_names = {t.get_name() for t in resolved_tools}
        default_tools = [
            cls() for name, cls in DEFAULT_TOOLS.items()
            if name not in user_tool_names
        ]
        resolved_tools = resolved_tools + default_tools

    # Skill tool 注入（两种路径都需要检查）
    if not config.skip_default_tools_merge and self.options.enable_skill:
        skill_manager = _create_skill_manager(self.options, self.name)
        skill_tool = skill_manager.get_skill_tool()
        resolved_tools.append(skill_tool)

    return resolved_tools, skill_manager
```

### 删除内容

- **删除** `create_agent()` 模块级函数
- **删除** `Agent._system_prompt` 字段
- **删除** `Agent.system_prompt` property
- **保留** `Agent.get_effective_system_prompt()` 作为唯一获取 prompt 的方式

### 消费者迁移

| 消费者 | 当前 | 迁移后 |
|--------|------|--------|
| SDK 用户 | `create_agent(name=..., model=..., ...)` | `Agent(AgentConfig(name=..., model=..., ...))` |
| Console `build_agent()` | `create_agent(...)` | `Agent(AgentConfig(...))` |
| Console `rehydrate_agent()` | `build_agent()` + `agent.id = ...` | `Agent(AgentConfig(id=state.id, ...))` |
| 测试 | `create_agent(...)` 或 `Agent(name=..., id=..., tools=..., ...)` | `Agent(AgentConfig(...))` |

---

## 方案 3：derive_child() 基于 config 派生

**文件**：`agiwo/agent/agent.py`

```python
async def derive_child(
    self,
    *,
    child_id: str,
    instruction: str | None = None,
    system_prompt_override: str | None = None,
    exclude_tool_names: set[str] | None = None,
) -> "Agent":
    # 1. 构建 child system_prompt
    system_prompt = system_prompt_override or await self.get_effective_system_prompt()
    if instruction:
        system_prompt += f"\n\n<task-instruction>\n{instruction}\n</task-instruction>"

    # 2. 过滤 tools（基于装配后的完整列表）
    tools = list(self.tools)
    if exclude_tool_names:
        tools = [t for t in tools if t.get_name() not in exclude_tool_names]

    # 3. 派生 child options
    child_options = copy.deepcopy(self.options)
    child_options.enable_termination_summary = True

    # 4. 构造 child config
    child_config = AgentConfig(
        name=self.config.name,
        description=self.config.description,
        model=self.config.model,
        id=child_id,
        system_prompt=system_prompt,
        tools=tools,
        options=child_options,
        hooks=self.config.hooks,       # Agent.__init__ 会 deepcopy
        skip_default_tools_merge=True,  # 跳过重复装配
    )

    # 5. 全新 Agent（完整装配路径）
    return Agent(child_config)
```

**关键语义变化**：

- 每个 child 拥有独立的 storage handles、hooks、skill_manager、prompt builder
- tools 过滤基于父 agent 装配后的 `self.tools`，child 的 config 标记 `skip_default_tools_merge=True` 跳过重复装配
- hooks 传递的是原始 `self.config.hooks`（用户模板），`Agent.__init__` 会 deepcopy 后使用
- **不再存在共享 live objects 的问题**

---

## 方案 4：scheduler cleanup 补 close

**文件**：`agiwo/scheduler/runner.py`

```python
async def _maybe_cleanup_agent(self, state: AgentState) -> None:
    self._coordinator.release_state_dispatch(state.id)
    if state.parent_id is None:
        return
    refreshed = await self._store.get_state(state.id)
    if refreshed is None or refreshed.status in (
        AgentStateStatus.COMPLETED,
        AgentStateStatus.FAILED,
    ):
        # 先获取 agent 引用，再 unregister，最后 close
        agent = self._coordinator.get_registered_agent(state.id)
        self._coordinator.unregister_agent(state.id)
        if agent is not None:
            await agent.close()
```

**设计要点**：

- `close()` 在 `unregister` 之后调用，避免并发路径在 close 过程中拿到该 agent
- SLEEPING 状态的 child 不会被 close（需要保留以支持后续 wake）
- root agent 的 close 由调用者（Console / SDK 用户）负责

---

## 方案 5：配置模型边界隔离

### 核心方向

SDK 只保留纯运行时配置类。Console API 层的 Input/Patch DTO 全部迁移到 Console 层，Patch 类彻底删除（web 端只传全量参数做全量 update）。

### SDK 层变更

**`agiwo/agent/options.py`**：
- 保留：`AgentOptions`、`RunStepStorageConfig`、`TraceStorageConfig`
- 删除：`AgentOptionsInput`、`AgentOptionsPatch`、`sanitize_agent_options_data()`

**`agiwo/llm/factory.py`**：
- 保留：`ModelConfig`
- 删除：`ModelParamsInput`、`ModelParamsPatch`

**`agiwo/llm/config_policy.py`**：
- `sanitize_model_params_data()` 如果只服务于 Input/Patch validator，迁移到 Console 层

### Console 层变更

**`console/server/domain/`**：
- 迁入 `AgentOptionsInput`、`ModelParamsInput`（Console 自主管理验证逻辑）
- **彻底删除** `AgentOptionsPatch`、`ModelParamsPatch`

**`console/server/routers/agents.py`**：
- 删除 PATCH 语义，改为全量 update

**`console/server/services/agent_lifecycle.py`**：
- import 路径更新

**`console/server/services/agent_config_policy.py`**：
- import 路径更新

### 好处

- SDK `options.py` 从 197 行缩减到约 100 行，只剩纯配置
- SDK 不再承担 API 验证/清洗职责
- Console 可自由演进 DTO 不影响 SDK
- 删除 Patch 类后 Console 逻辑更简单

---

## 受影响文件汇总

### SDK 新增

| 文件 | 内容 |
|------|------|
| `agiwo/agent/config.py` | `AgentConfig` dataclass |

### SDK 修改

| 文件 | 改动 |
|------|------|
| `agiwo/agent/agent.py` | Agent.__init__ 接收 AgentConfig，删除 create_agent()，删除 _system_prompt / system_prompt property |
| `agiwo/agent/__init__.py` | 导出 AgentConfig，删除 create_agent 导出 |
| `agiwo/__init__.py` | 导出 AgentConfig，删除 create_agent 导出 |
| `agiwo/agent/options.py` | 删除 AgentOptionsInput, AgentOptionsPatch, sanitize_agent_options_data |
| `agiwo/llm/factory.py` | 删除 ModelParamsInput, ModelParamsPatch |
| `agiwo/llm/config_policy.py` | 评估 sanitize_model_params_data 是否需要迁移 |
| `agiwo/scheduler/runner.py` | _maybe_cleanup_agent 补 close |

### Console 修改

| 文件 | 改动 |
|------|------|
| `console/server/domain/` | 迁入 AgentOptionsInput, ModelParamsInput |
| `console/server/services/agent_lifecycle.py` | import 路径更新，build_agent 改用 AgentConfig |
| `console/server/services/agent_config_policy.py` | import 路径更新 |
| `console/server/routers/agents.py` | 删除 patch 语义，改全量 update |
| `console/server/channels/runtime_agent_pool.py` | build_agent 调用方式更新 |

### 测试

| 文件 | 改动 |
|------|------|
| `tests/agent/` | 所有测试从 create_agent / Agent(name=...) 迁移到 Agent(AgentConfig(...)) |
| `tests/scheduler/` | 同上 |
| `console/tests/` | import 路径更新 + 构造方式更新 |

---

## 执行顺序建议

1. **Phase 0**：新建 `AgentConfig`，从 `agiwo.agent` 和 `agiwo` 导出
2. **Phase 1**：重构 `Agent.__init__` 接收 AgentConfig，删除 `create_agent()`、`_system_prompt`、`system_prompt` property
3. **Phase 2**：重构 `derive_child()` 基于 config 派生（storage/hooks/skill_manager 实例隔离自然生效）
4. **Phase 3**：scheduler cleanup 补 close
5. **Phase 4**：配置模型边界隔离（迁移 Input 到 Console，删除 Patch）
6. **Phase 5**：更新所有测试
7. **Phase 6**：lint + 全量测试验证

---

## 语义变化说明

### memory backend 下 child 的行为

child 拥有独立的内存 storage 后：
- child 运行期间历史仍然连续
- child 完成并被 cleanup 后，内存数据随对象消失
- 这是预期行为：memory backend 下 child 的详细 step/compact/trace 是临时态

### close() 语义

- 每个 Agent 只关闭自己持有的 storage handles
- root/child 互不影响
- scheduler 对 completed/failed child 显式 close
- root agent 的 close 由外部调用者负责
