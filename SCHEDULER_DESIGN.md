# Agent 调度系统设计方案

## 1. 概述

### 目标

为 Agiwo Agent SDK 增加调度能力：

- Agent 在运行中派生独立子 Agent 执行不同任务（独立 Session/Run/Step）
- Agent 可挂起自身，在条件满足时被调度器唤醒恢复执行
- 通过持久化状态实现跨时间、跨进程的 Agent 生命周期管理

### 设计原则

- **Scheduler 无状态**：只依赖 Storage，不持有 Agent 实例或进程内状态，不耦合 Agent 逻辑
- **Agent 不感知调度**：Agent 通过工具（spawn / sleep / query）与调度系统交互，Agent 核心代码无需知道调度器存在
- **进程透明**：同进程或跨进程部署使用相同协议，仅 Storage 连接方式不同
- **Session 连续性复用**：Agent 唤醒 = 对已有 session 发起新 `agent.run(wake_message, session_id=same_session)`，利用现有的 `_execute_workflow` 从 `RunStepStorage` 加载对话历史，LLM 看到完整上下文后自然接续
- **V1 最小化**：建设核心框架和协议，不做过度设计

### V1 范围

**包含：**
- 3 个调度工具：`spawn_agent`、`sleep_and_wait`、`query_spawned_agent`
- `AgentState` 持久化（SQLite 实现）
- `ModelRegistry` + `ToolRegistry`（名称 → 实例的解析）
- `SchedulerLoop`（单进程后台运行）
- `AgentExecutor` 最小改动（支持 tool 驱动的执行终止）

**不包含（记录为 Future Work）：**
- Hooks 恢复
- 跨进程部署 / 多 Scheduler 竞态处理
- Blueprint 版本兼容
- Agent 最大唤醒次数限制
- Scheduler 高可用

---

## 2. 模块结构

```
agiwo/
├── scheduler/
│   ├── __init__.py
│   ├── models.py          # AgentState, WakeCondition, ModelRef, AgentBlueprint
│   ├── store.py           # AgentStateStorage ABC + SQLite 实现
│   ├── registry.py        # ModelRegistry + ToolRegistry
│   ├── builder.py         # AgentBuilder: Blueprint + Registry → Agent 实例
│   └── loop.py            # SchedulerLoop: 检查唤醒 + 信号传播 + Agent 重建执行
├── tool/
│   └── scheduling/
│       ├── __init__.py
│       ├── spawn_tool.py   # SpawnAgentTool: 派生子 Agent
│       ├── sleep_tool.py   # SleepAndWaitTool: 挂起并注册唤醒条件
│       └── query_tool.py   # QuerySpawnedAgentTool: 查询子 Agent 状态/结果
```

---

## 3. 核心数据模型

> 所有模型位于 `agiwo/scheduler/models.py`

### 3.1 ModelRef

可序列化的 Model 引用，用于通过 `ModelRegistry` 重建 `Model` 实例。

```python
@dataclass
class ModelRef:
    provider: str              # "deepseek" | "openai" | "anthropic" | "nvidia"
    model_id: str              # "deepseek-chat" | "gpt-4o" | ...
    params: dict[str, Any]     # temperature, max_tokens, top_p 等可覆盖参数

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelRef": ...
```

### 3.2 AgentBlueprint

可序列化的 Agent 构建描述，包含重建一个等价 Agent 实例所需的全部信息（Hooks 除外）。

```python
@dataclass
class AgentBlueprint:
    agent_id: str
    description: str
    model_ref: ModelRef
    tool_names: list[str]          # 通过 ToolRegistry 解析为 BaseTool 实例
    system_prompt: str
    options: dict[str, Any]        # AgentOptions 各字段

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentBlueprint": ...

    @classmethod
    def from_agent(cls, agent: "Agent") -> "AgentBlueprint":
        """从 Agent 实例提取 Blueprint。
        
        依赖：
        - agent.model.provider (需要 Model 新增 provider 属性)
        - agent.tools[*].get_name()
        - agent.options (dataclass → dict)
        """
        ...
```

**Blueprint 获取**：Agent 在执行开始时将 `SchedulingMeta`（含 Blueprint + RunStepStorage）存入 `context.metadata["_scheduling"]`，调度工具从中读取。详见 6.3。

### 3.3 WakeCondition

简化设计：去掉 `at_least`，semaphore 由 Scheduler 根据实际 spawn 的子 Agent 数量自动管理。

```python
class TimeUnit(str, Enum):
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"

class WakeType(str, Enum):
    CHILDREN_COMPLETE = "children_complete"   # 所有子 Agent 完成后唤醒
    INTERVAL = "interval"                     # 每隔 N 秒唤醒
    DELAY = "delay"                           # 延迟一段时间后唤醒（一次性）

@dataclass
class WakeCondition:
    type: WakeType

    # INTERVAL 模式
    interval_seconds: int | None = None

    # DELAY 模式
    # LLM 输出 delay_value + delay_unit（如 "2 hours"），程序换算为秒并计算 wakeup_at
    delay_value: int | None = None
    delay_unit: TimeUnit | None = None
    wakeup_at: float | None = None      # 由程序自动计算，LLM 不直接设置

    # CHILDREN_COMPLETE 模式 (Scheduler 自动维护以下两个字段)
    total_children: int = 0
    completed_children: int = 0

    # 可选：超时保护（所有模式通用）
    timeout_seconds: int | None = None

    # Scheduler 内部追踪
    last_wake_at: float | None = None

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WakeCondition": ...
```

**组合使用**：Agent 可以 sleep 时同时指定 `CHILDREN_COMPLETE` 和 `interval_seconds`。Scheduler 任一条件满足即唤醒：
- 所有子 Agent 完成 → 唤醒（最终唤醒）
- 每隔 N 秒 → 唤醒（中间检查，Agent 可用 `query_spawned_agent` 查看进度后决定是否再次 sleep）

**DELAY vs TIMESTAMP**：LLM 难以输出准确的 Unix 时间戳。LLM 输出 `delay_value`（数值）+ `delay_unit`（固定枚举: seconds/minutes/hours/days），`SleepAndWaitTool.execute()` 换算为秒并计算 `wakeup_at = time.time() + total_seconds`。

### 3.4 AgentState

调度系统的核心状态记录。一个 AgentState 代表一个可被调度的 Agent 生命周期，跨 sleep/wake 周期稳定。

```python
class AgentStateStatus(str, Enum):
    PENDING = "pending"         # 刚 spawn，等待 Scheduler 启动
    RUNNING = "running"         # 正在执行中
    SLEEPING = "sleeping"       # 挂起，等待唤醒条件
    COMPLETED = "completed"     # 执行完成
    FAILED = "failed"           # 执行失败

@dataclass
class AgentState:
    id: str                            # 稳定标识，跨 sleep/wake 不变
    session_id: str                    # 会话 ID，跨 sleep/wake 不变
    agent_id: str                      # Agent 逻辑 ID

    parent_state_id: str | None        # 父 Agent 的 state_id（信号传播用）
    status: AgentStateStatus

    blueprint: dict[str, Any]          # AgentBlueprint.to_dict()
    wake_condition: dict[str, Any] | None

    task: str                          # spawn 时的任务描述
    last_run_id: str | None            # 最近一次执行的 run_id
    result_summary: str | None         # 完成后的最终响应文本

    created_at: datetime
    updated_at: datetime
```

**生命周期状态转换：**

```
PENDING ──→ RUNNING ──→ COMPLETED
                │
                ├──→ SLEEPING ──→ RUNNING ──→ COMPLETED
                │                    │
                │                    ├──→ SLEEPING ──→ ...
                │                    │
                │                    └──→ FAILED
                │
                └──→ FAILED
```

---

## 4. AgentStateStorage

> 位于 `agiwo/scheduler/store.py`

### 4.1 ABC 接口

```python
class AgentStateStorage(ABC):

    async def close(self) -> None: ...

    # --- CRUD ---

    @abstractmethod
    async def save_state(self, state: AgentState) -> None:
        """创建或更新 AgentState"""

    @abstractmethod
    async def get_state(self, state_id: str) -> AgentState | None:
        """按 ID 获取"""

    @abstractmethod
    async def get_states_by_parent(self, parent_state_id: str) -> list[AgentState]:
        """获取某个父 Agent 的所有子 Agent 状态"""

    # --- 调度查询 ---

    @abstractmethod
    async def find_pending(self) -> list[AgentState]:
        """查找所有 PENDING 状态的 Agent（需要启动）"""

    @abstractmethod
    async def find_wakeable(self, now: float) -> list[AgentState]:
        """查找所有满足唤醒条件的 SLEEPING Agent。
        
        条件判断：
        - CHILDREN_COMPLETE: completed_children >= total_children
        - INTERVAL: now - last_wake_at >= interval_seconds
        - TIMESTAMP: now >= wakeup_at
        - timeout: now - created_at >= timeout_seconds
        """

    # --- 信号 ---

    @abstractmethod
    async def increment_completed_children(self, parent_state_id: str) -> None:
        """子 Agent 完成时，递增父 Agent 的 completed_children 计数"""

    # --- 状态更新 ---

    @abstractmethod
    async def update_status(
        self, state_id: str, status: AgentStateStatus, **updates
    ) -> None:
        """更新状态及附加字段（last_run_id, result_summary 等）"""
```

### 4.2 SQLite 实现

V1 使用 SQLite，表结构：

```sql
CREATE TABLE IF NOT EXISTS agent_states (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    parent_state_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    blueprint TEXT NOT NULL,          -- JSON
    wake_condition TEXT,              -- JSON
    task TEXT NOT NULL,
    last_run_id TEXT,
    result_summary TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_agent_states_status ON agent_states(status);
CREATE INDEX idx_agent_states_parent ON agent_states(parent_state_id);
```

---

## 5. Registry

> 位于 `agiwo/scheduler/registry.py`

Registry 负责**名称 → 实例**的解析，用于从 Blueprint 重建 Agent。Scheduler 启动时由用户注册所有可用的 Model 和 Tool。

### 5.1 ModelRegistry

```python
class ModelRegistry:
    """Model 工厂注册表。"""

    def register(
        self, provider: str, model_id: str, factory: Callable[..., Model]
    ) -> None:
        """注册 Model 工厂函数。
        
        factory 签名: (params: dict) -> Model
        例: registry.register("deepseek", "deepseek-chat", 
                lambda params: DeepseekModel(id="deepseek-chat", name="deepseek-chat", **params))
        """

    def resolve(self, ref: ModelRef) -> Model:
        """从 ModelRef 解析出 Model 实例。找不到则 raise KeyError。"""

    def get_ref(self, model: Model) -> ModelRef:
        """从 Model 实例反向生成 ModelRef。
        依赖 model.provider 属性。"""
```

### 5.2 ToolRegistry

```python
class ToolRegistry:
    """Tool 工厂注册表。"""

    def register(self, name: str, factory: Callable[[], BaseTool]) -> None:
        """注册 Tool 工厂函数。
        
        例: registry.register("calculator", lambda: Calculator())
            registry.register("http_request", lambda: HttpRequestTool(api_key=os.getenv("KEY")))
        """

    def resolve(self, name: str) -> BaseTool:
        """按名称解析 Tool 实例。找不到则 raise KeyError。"""

    def resolve_many(self, names: list[str]) -> list[BaseTool]:
        """批量解析。"""
```

**内置工具自动注册**：`ToolRegistry` 初始化时自动从 `DEFAULT_TOOLS` 注册所有内置工具，用户只需注册自定义工具。

---

## 6. Agent 改动

> 改动范围极小，仅增加 Blueprint 支持、provider 字段、bind_agent 机制。

### 6.1 Model 新增 `provider` 字段

`agiwo/llm/base.py` — Model ABC 新增显式字段：

```python
@dataclass
class Model(ABC):
    id: str
    name: str
    provider: str          # NEW: 显式字段，由构造时传入
    # ... 现有字段 (temperature, max_tokens, top_p ...) ...
```

`OpenAIModel` 代表的是 OpenAI-compatible API，不是 OpenAI 本身。provider 由用户在构造时指定，不由子类硬编码：

```python
# 用户代码
model = DeepseekModel(id="deepseek-chat", name="deepseek-chat", provider="deepseek")
model = OpenAIModel(id="gpt-4o", name="gpt-4o", provider="openai")
model = AnthropicModel(id="claude-3", name="claude-3", provider="anthropic")

# 同一个 OpenAIModel 子类可有不同 provider
model = NvidiaModel(id="llama-3", name="llama-3", provider="nvidia")
```

### 6.2 Agent 新增 `to_blueprint()` 方法

`agiwo/agent/agent.py`：

```python
class Agent:
    def to_blueprint(self) -> "AgentBlueprint":
        """生成可序列化的 Blueprint，包含重建此 Agent 所需的全部配置。"""
        from agiwo.scheduler.models import AgentBlueprint, ModelRef
        return AgentBlueprint(
            agent_id=self.id,
            description=self.description,
            model_ref=ModelRef(
                provider=self.model.provider,
                model_id=self.model.id,
                params={
                    "temperature": self.model.temperature,
                    "max_tokens": self.model.max_tokens,
                    "top_p": self.model.top_p,
                },
            ),
            tool_names=[t.get_name() for t in (self.tools or [])],
            system_prompt=self.system_prompt,
            options=self._serialize_options(),
        )
```

### 6.3 SchedulingMeta — 通过 context.metadata 传递调度上下文

调度工具需要在 `execute()` 时访问当前 Agent 的 Blueprint 和 RunStepStorage。工具实例是无状态的全局单例，不能持有 Agent 引用。因此通过 per-execution 的 `context.metadata` 传递一个类型化对象。

**新增数据模型**（`agiwo/scheduler/models.py`）：

```python
@dataclass
class SchedulingMeta:
    blueprint: dict[str, Any]          # AgentBlueprint.to_dict()
    run_step_storage: RunStepStorage   # 当前 Agent 的存储实例
    agent_state_id: str | None = None  # Scheduler 唤醒时注入，跨 sleep/wake 周期稳定

SCHEDULING_META_KEY = "_scheduling"
```

**Agent 注入时机**（`Agent._execute_workflow` 开头）：

```python
SCHEDULING_STATE_ID_KEY = "_scheduling_state_id"

async def _execute_workflow(self, user_input, context, abort_signal):
    # 注入调度上下文供 scheduling tools 使用
    context.metadata[SCHEDULING_META_KEY] = SchedulingMeta(
        blueprint=self.to_blueprint().to_dict(),
        run_step_storage=self.run_step_storage,
        # Scheduler 唤醒时通过 run(metadata={...}) 预注入 state_id
        agent_state_id=context.metadata.get(SCHEDULING_STATE_ID_KEY),
    )

    emitter = EventEmitter(context)
    # ... 后续不变 ...
```

**state_id 注入链路**：
```
Scheduler._wake_agent:
  agent.run(wake_msg, session_id=..., metadata={"_scheduling_state_id": state.id})
    → Agent._create_context: context.metadata = {"_scheduling_state_id": state.id}
    → Agent._execute_workflow: 读取 context.metadata["_scheduling_state_id"]
       → SchedulingMeta(agent_state_id=state.id, ...)
       → 写入 context.metadata["_scheduling"]
    → SleepAndWaitTool.execute:
       meta = context.metadata["_scheduling"]
       meta.agent_state_id  # 有值 → 更新已有 state
```

首次执行（非 Scheduler 唤醒）时 `_scheduling_state_id` 不存在 → `agent_state_id=None` → sleep_tool 创建新 state。

**工具读取方式**：

```python
from agiwo.scheduler.models import SchedulingMeta, SCHEDULING_META_KEY

meta: SchedulingMeta = context.metadata[SCHEDULING_META_KEY]
blueprint = meta.blueprint
storage = meta.run_step_storage
state_id = meta.agent_state_id   # None if first run, set if woken by Scheduler
```

**设计要点**：
- `context.metadata` 是 per-execution 的 dict，天然隔离多 Agent 并发
- 工具实例完全无状态，可被多个 Agent 安全共享
- `ExecutionContext` 核心定义不变，调度信息作为扩展数据存在于 metadata 中
- state_id 注入链路清晰：Scheduler → run(metadata) → context.metadata → SchedulingMeta → tool 读取

### 6.4 TerminationReason 新增 SLEEPING

`agiwo/agent/schema.py`：

```python
class TerminationReason(str, Enum):
    # ... 现有值 ...
    SLEEPING = "sleeping"      # Agent 主动挂起，等待调度器唤醒
```

---

## 7. Executor 改动

> 改动量：~10 行

### 7.1 ToolResult 新增终止标志

`agiwo/tool/base.py`：

```python
@dataclass
class ToolResult:
    # ... 现有字段 ...
    terminate_execution: bool = False    # 工具请求终止当前执行循环
```

### 7.2 AgentExecutor 检查终止标志

`agiwo/agent/inner/executor.py` — `_execute_tools` 末尾新增：

```python
async def _execute_tools(self, state, tool_calls, abort_signal):
    # ... 现有逻辑：execute_batch + 记录 steps ...

    # NEW: 检查工具请求的执行终止
    for result in results:
        if result.terminate_execution:
            state.termination_reason = TerminationReason.SLEEPING
            return
```

`_run_loop` 中 `_execute_tools` 调用后新增检查：

```python
async def _run_loop(self, state, pending_tool_calls, abort_signal):
    # ...
    while True:
        # ... LLM call ...
        if not step.tool_calls:
            state.termination_reason = TerminationReason.COMPLETED
            return

        await self._execute_tools(state, step.tool_calls, abort_signal)

        # NEW: 工具请求终止（如 sleep_and_wait）
        if state.termination_reason is not None:
            return
```

---

## 8. 调度工具

> 位于 `agiwo/tool/scheduling/`
>
> 三个工具均通过构造函数注入 `AgentStateStorage` 等依赖，无全局状态。

### 8.1 SpawnAgentTool

**职责**：在 `AgentStateStorage` 中创建一条 `PENDING` 状态的 `AgentState` 记录。Scheduler 在下一次检查时发现并启动它。

```python
class SpawnAgentTool(BaseTool):

    def __init__(self, state_store: AgentStateStorage) -> None:
        self.state_store = state_store

    def get_name(self) -> str:
        return "spawn_agent"

    def get_description(self) -> str:
        return (
            "Spawn an independent child agent to execute a task. "
            "The child runs in a separate session. Returns a state_id for tracking."
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Task description for the child agent. Be specific and complete.",
                },
                "config_overrides": {
                    "type": "object",
                    "description": "Optional overrides for child agent config.",
                    "properties": {
                        "system_prompt": {"type": "string"},
                        "description": {"type": "string"},
                        "max_steps": {"type": "integer"},
                        "max_tokens": {
                            "type": "integer",
                            "description": "Max token budget for child agent. Default: 100000.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Max execution time in seconds. Default: 300.",
                        },
                    },
                },
            },
            "required": ["task"],
        }
```

**execute 逻辑**：

1. 从 `context.metadata[SCHEDULING_META_KEY]` 读取 `SchedulingMeta`，获取父 Agent Blueprint 作为模板
2. Deep-copy + 应用 `config_overrides`（未指定 `max_tokens` 默认 100000，`timeout` 默认 300s）
3. 创建 `AgentState(status=PENDING, task=task, blueprint=child_blueprint)`
4. 设置 `parent_state_id`（从 `meta.agent_state_id` 读取，如果父 Agent 也在调度系统中）
5. 写入 `AgentStateStorage`
6. 返回 `ToolResult(content="Spawned child agent. state_id={state.id}")`

### 8.2 SleepAndWaitTool

**职责**：将当前 Agent 的状态持久化到 `AgentStateStorage`，设置唤醒条件，并通过 `terminate_execution=True` 终止当前执行。

```python
class SleepAndWaitTool(BaseTool):

    def __init__(self, state_store: AgentStateStorage) -> None:
        self.state_store = state_store

    def get_name(self) -> str:
        return "sleep_and_wait"

    def get_description(self) -> str:
        return (
            "Suspend the current agent and wait for a wake condition. "
            "Use after spawning child agents to wait for their completion, "
            "or to schedule a future wake-up."
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "wake_type": {
                    "type": "string",
                    "enum": ["children_complete", "interval", "delay"],
                    "description": (
                        "children_complete: wake when all spawned children finish. "
                        "interval: wake every N seconds. "
                        "delay: wake after a specified duration (one-shot)."
                    ),
                },
                "interval_seconds": {
                    "type": "integer",
                    "description": "For interval mode: seconds between wake-ups.",
                },
                "delay_value": {
                    "type": "integer",
                    "description": "For delay mode: amount of time to wait.",
                },
                "delay_unit": {
                    "type": "string",
                    "enum": ["seconds", "minutes", "hours", "days"],
                    "description": "For delay mode: time unit.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Optional: maximum wait time before forced wake.",
                },
            },
            "required": ["wake_type"],
        }
```

**execute 逻辑**：

1. 从 `context.metadata[SCHEDULING_META_KEY]` 读取 `SchedulingMeta`，获取 Blueprint
2. 构建 `WakeCondition`；如果 `DELAY` 模式，换算为秒并计算 `wakeup_at = time.time() + total_seconds`
3. 如果 `meta.agent_state_id` 已存在（之前被唤醒过的 Agent）：
   - 更新已有 `AgentState`：status=SLEEPING，更新 wake_condition
4. 否则（首次 sleep）：
   - 创建新 `AgentState(status=SLEEPING, ...)`
   - 将 `state.id` 写回 `meta.agent_state_id`
5. 对于 `CHILDREN_COMPLETE` 模式：查询 `AgentStateStorage` 中 `parent_state_id=this_state_id` 的记录数，设置 `total_children`
6. 写入 Storage
7. 返回 `ToolResult(terminate_execution=True, content="Agent sleeping. Wake condition: {type}. state_id={state.id}")`

### 8.3 QuerySpawnedAgentTool

**职责**：查询子 Agent 的状态和执行结果。wake_message 只通知完成状态，Agent 通过此工具主动决定是否读取子 Agent 产出。

```python
class QuerySpawnedAgentTool(BaseTool):

    def __init__(self, state_store: AgentStateStorage) -> None:
        self.state_store = state_store
        # run_step_storage 通过 context.metadata[SCHEDULING_META_KEY].run_step_storage 访问

    def get_name(self) -> str:
        return "query_spawned_agent"

    def get_description(self) -> str:
        return (
            "Query the status and results of a spawned child agent. "
            "Returns status and optionally the full result."
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "state_id": {
                    "type": "string",
                    "description": "The state_id of the spawned agent to query.",
                },
                "include_result": {
                    "type": "boolean",
                    "description": "If true, include the agent's full final response. Default false.",
                    "default": False,
                },
                "include_steps": {
                    "type": "boolean",
                    "description": "If true, include recent execution steps. Default false.",
                    "default": False,
                },
            },
            "required": ["state_id"],
        }
```

**execute 逻辑**：

1. 从 `AgentStateStorage` 读取子 Agent 的 `AgentState`
2. 基础返回：`status`, `agent_id`, `task`
3. `include_result=True` 且 `status=COMPLETED`：返回 `state.result_summary`（即子 Agent 的最终响应）
4. `include_steps=True`：从 `RunStepStorage` 按 `session_id + agent_id` 查询最近 N 条步骤，返回简化摘要
5. 如果子 Agent 仍在运行：返回 `status=RUNNING` + 已完成步骤数

---

## 9. Scheduler Loop

> 位于 `agiwo/scheduler/loop.py`
>
> Scheduler 是一个持续运行的异步循环，无状态，只依赖 Storage 和 Registry。
> 同进程部署时作为 `asyncio.Task` 运行，跨进程部署时作为独立服务运行。

### 9.1 SchedulerLoop 接口

```python
class SchedulerLoop:

    def __init__(
        self,
        state_store: AgentStateStorage,
        run_step_storage: RunStepStorage,
        model_registry: ModelRegistry,
        tool_registry: ToolRegistry,
        check_interval: float = 5.0,
        max_concurrent: int = 10,
    ) -> None:
        self.state_store = state_store
        self.run_step_storage = run_step_storage
        self.model_registry = model_registry
        self.tool_registry = tool_registry
        self.check_interval = check_interval
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running = False

    async def start(self) -> None:
        """启动调度循环（阻塞）。"""
        self._running = True
        while self._running:
            await self._tick()
            await asyncio.sleep(self.check_interval)

    async def stop(self) -> None:
        """停止调度循环。"""
        self._running = False

    async def _tick(self) -> None:
        """单次调度检查。"""
        await self._propagate_signals()
        await self._start_pending_agents()
        await self._wake_sleeping_agents()
```

### 9.2 `_start_pending_agents`

处理 `PENDING` 状态的 Agent（新 spawn 的子 Agent），**并行执行**：

```python
async def _start_pending_agents(self) -> None:
    pending = await self.state_store.find_pending()
    if not pending:
        return
    tasks = [self._run_agent(state) for state in pending]
    await asyncio.gather(*tasks, return_exceptions=True)

async def _run_agent(self, state: AgentState) -> None:
    async with self._semaphore:          # 受 max_concurrent 控制
        await self.state_store.update_status(state.id, AgentStateStatus.RUNNING)
        try:
            agent = self.builder.build(state.blueprint)
            output = await agent.run(state.task, session_id=state.session_id)
            await self.state_store.update_status(
                state.id, AgentStateStatus.COMPLETED,
                result_summary=output.response,
                last_run_id=output.run_id,
            )
        except Exception as e:
            await self.state_store.update_status(
                state.id, AgentStateStatus.FAILED,
                result_summary=str(e),
            )
```

**并发控制**：`asyncio.Semaphore(max_concurrent)` 限制同时运行的 Agent 数量，防止 LLM API 超限。默认 `max_concurrent=10`。

### 9.3 `_wake_sleeping_agents`

处理满足唤醒条件的 `SLEEPING` Agent：

```python
async def _wake_sleeping_agents(self) -> None:
    now = time.time()
    wakeable = await self.state_store.find_wakeable(now)
    if not wakeable:
        return
    tasks = [self._wake_agent(state, now) for state in wakeable]
    await asyncio.gather(*tasks, return_exceptions=True)

async def _wake_agent(self, state: AgentState, now: float) -> None:
    async with self._semaphore:
        await self.state_store.update_status(
            state.id, AgentStateStatus.RUNNING, last_wake_at=now
        )
        agent = self.builder.build(state.blueprint)
        wake_message = self._build_wake_message(state)
        # 通过 run(metadata=...) 注入 state_id，Agent._execute_workflow 会将其合并到 SchedulingMeta
        try:
            output = await agent.run(
                wake_message,
                session_id=state.session_id,
                metadata={SCHEDULING_STATE_ID_KEY: state.id},
            )
            if output.termination_reason == TerminationReason.SLEEPING:
                pass  # sleep_tool 已更新 state
            elif output.termination_reason == TerminationReason.COMPLETED:
                await self.state_store.update_status(
                    state.id, AgentStateStatus.COMPLETED,
                    result_summary=output.response,
                )
            else:
                await self.state_store.update_status(
                    state.id, AgentStateStatus.FAILED,
                    result_summary=f"Unexpected termination: {output.termination_reason}",
                )
        except Exception as e:
            await self.state_store.update_status(
                state.id, AgentStateStatus.FAILED, result_summary=str(e),
            )
```

### 9.4 `_propagate_signals`

子 Agent 完成后向父 Agent 传播信号：

```
1. 查询所有刚完成的 Agent (status=COMPLETED 且 parent_state_id 不为空)
   - 需要一个标记区分 "已传播" 和 "未传播" 的完成事件
   - V1 方案：AgentState 增加 signal_propagated: bool 字段
2. 对每个未传播的完成状态:
   a. state_store.increment_completed_children(parent_state_id)
   b. 标记 signal_propagated = True
```

信号传播后，父 Agent 的 `wake_condition.completed_children` 递增。
下一次 `_wake_sleeping_agents` 检查时，如果 `completed_children >= total_children`，父 Agent 被唤醒。

### 9.5 Wake Message 构建

Wake message 只通知状态，**不包含子 Agent 的具体产出内容**。Agent 通过 `query_spawned_agent` 工具主动决定是否读取。

```python
def _build_wake_message(self, state: AgentState) -> str:
    wake_condition = WakeCondition.from_dict(state.wake_condition)

    if wake_condition.type == WakeType.CHILDREN_COMPLETE:
        children = await self.state_store.get_states_by_parent(state.id)
        child_summary = "\n".join(
            f"- {c.id}: status={c.status.value}, task=\"{c.task[:80]}\""
            for c in children
        )
        return (
            f"<wake_signal>\n"
            f"All {len(children)} spawned child agents have completed.\n"
            f"Children:\n{child_summary}\n"
            f"Use query_spawned_agent tool to read specific results.\n"
            f"</wake_signal>"
        )

    elif wake_condition.type == WakeType.INTERVAL:
        return (
            f"<wake_signal>\n"
            f"Periodic wake-up (interval: {wake_condition.interval_seconds}s).\n"
            f"Use query_spawned_agent tool to check child agent progress.\n"
            f"</wake_signal>"
        )

    elif wake_condition.type == WakeType.DELAY:
        return (
            f"<wake_signal>\n"
            f"Scheduled wake-up reached (after {wake_condition.delay_value} {wake_condition.delay_unit}).\n"
            f"</wake_signal>"
        )
```

### 9.6 AgentBuilder

> 位于 `agiwo/scheduler/builder.py`

从 Blueprint + Registry 重建 Agent 实例：

```python
class AgentBuilder:
    def __init__(
        self,
        model_registry: ModelRegistry,
        tool_registry: ToolRegistry,
        run_step_storage: RunStepStorage,
    ) -> None: ...

    def build(self, blueprint: AgentBlueprint) -> Agent:
        """从 Blueprint 重建 Agent。

        1. model_registry.resolve(blueprint.model_ref) → Model
        2. tool_registry.resolve_many(blueprint.tool_names) → list[BaseTool]
        3. AgentOptions(**blueprint.options) → options
        4. Agent(id=..., model=..., tools=..., system_prompt=..., options=...)
        
        注意：
        - Hooks 不恢复（V1 限制）
        - 重建的 Agent 使用 Scheduler 提供的 run_step_storage（共享实例）
        - 调度工具（spawn/sleep/query）需要重新注入到 tools 中
        """
```

---

## 10. 执行流程

### 10.1 Spawn 流程

```
┌─────────────────────────────────────────────────────┐
│ Main Agent 执行中                                    │
│                                                     │
│  LLM → tool_call: spawn_agent(task="Research X")    │
│         │                                           │
│  SpawnAgentTool.execute():                          │
│    1. meta = context.metadata["_scheduling"]         │
│       blueprint = meta.blueprint (父 Agent 模板)      │
│    2. Deep-copy Blueprint, 应用 config_overrides     │
│    3. AgentState(                                   │
│         status=PENDING,                             │
│         task="Research X",                          │
│         parent_state_id=my_state_id,                │
│         blueprint=child_blueprint                   │
│       )                                             │
│    4. state_store.save_state(state)                  │
│    5. return ToolResult(                            │
│         content="Spawned. state_id=abc123"          │
│       )                                             │
│                                                     │
│  LLM 继续运行（可 spawn 更多 / sleep / 正常完成）     │
└─────────────────────────────────────────────────────┘

        ↓ 秒级延迟

┌─────────────────────────────────────────────────────┐
│ Scheduler._tick()                                   │
│                                                     │
│  _start_pending_agents():                           │
│    1. find_pending() → [state_abc123]               │
│    2. status = RUNNING                              │
│    3. AgentBuilder.build(state.blueprint) → Agent   │
│    4. agent.run("Research X",                       │
│         session_id=state.session_id)                │
│    5. → RunOutput                                   │
│    6. status = COMPLETED                            │
│       result_summary = output.response              │
└─────────────────────────────────────────────────────┘
```

### 10.2 Sleep 流程

```
┌─────────────────────────────────────────────────────┐
│ Main Agent 执行中 (已 spawn 了 3 个子 Agent)         │
│                                                     │
│  LLM → tool_call: sleep_and_wait(                   │
│           wake_type="children_complete"              │
│         )                                           │
│         │                                           │
│  SleepAndWaitTool.execute():                        │
│    1. 读取 Blueprint from context.metadata           │
│    2. 查询子 Agent 数量 → total_children = 3         │
│    3. AgentState(                                   │
│         status=SLEEPING,                            │
│         wake_condition={                            │
│           type: "children_complete",                │
│           total_children: 3,                        │
│           completed_children: 0                     │
│         }                                           │
│       )                                             │
│    4. state_store.save_state(state)                  │
│    5. return ToolResult(                            │
│         terminate_execution=True,                   │
│         content="Sleeping. state_id=parent123"      │
│       )                                             │
│                                                     │
│  Executor 检测 terminate_execution=True              │
│  → termination_reason = SLEEPING                    │
│  → Agent 执行结束                                   │
└─────────────────────────────────────────────────────┘
```

### 10.3 Wake 流程

```
┌─────────────────────────────────────────────────────┐
│ Scheduler._tick() (持续运行)                         │
│                                                     │
│  _propagate_signals():                              │
│    子 Agent abc 完成 → parent.completed_children += 1│
│    子 Agent def 完成 → parent.completed_children += 1│
│    子 Agent ghi 完成 → parent.completed_children += 1│
│                                                     │
│  _wake_sleeping_agents():                           │
│    find_wakeable() → [parent123]                    │
│      (completed_children=3 >= total_children=3)     │
│                                                     │
│    1. status = RUNNING                              │
│    2. AgentBuilder.build(blueprint) → Agent         │
│    3. wake_message = (                              │
│         "<wake_signal>                              │
│          All 3 children completed.                  │
│          - abc: status=completed, task=..."         │
│          - def: status=completed, task=..."         │
│          - ghi: status=completed, task=..."         │
│          Use query_spawned_agent to read results.   │
│          </wake_signal>"                            │
│       )                                             │
│    4. context.metadata["_agent_state_id"] = parent123│
│    5. agent.run(wake_message,                       │
│         session_id=state.session_id)                │
│                                                     │
│  Agent 恢复执行：                                    │
│    _execute_workflow:                               │
│      existing_steps = storage.get_steps(session_id) │
│      → 加载完整对话历史 (spawn + sleep 记录)         │
│      messages = assemble(system_prompt,              │
│                          existing_steps,             │
│                          ...)                        │
│      → LLM 看到完整上下文 + wake_signal              │
│      → LLM 使用 query_spawned_agent 读取子结果       │
│      → LLM 综合处理并生成最终响应                     │
│                                                     │
│    结果：                                            │
│      termination_reason = COMPLETED                 │
│      → Scheduler 更新 status = COMPLETED            │
│      → 如果有 parent，传播信号给更上层                │
└─────────────────────────────────────────────────────┘
```

### 10.4 Interval + Children Complete 组合流程

```
Agent spawn 3 children → sleep(wake_type="children_complete", interval_seconds=60)

Scheduler 行为：
  t+60s:  interval 满足 → 唤醒 Agent
          Agent 用 query_spawned_agent 查看进度
          → 发现还有 2 个在运行
          → 调用 sleep_and_wait(wake_type="children_complete")  再次 sleep
          → Scheduler 更新 state (同一个 state_id)

  t+120s: interval 满足 → 唤醒 Agent
          → 发现还有 1 个在运行
          → 再次 sleep

  t+150s: 最后一个子 Agent 完成
          → _propagate_signals → completed_children=3
          → children_complete 满足 → 唤醒 Agent（不等 interval）
          → Agent 读取所有结果，完成工作
```

---

## 11. 用户使用示例

```python
from agiwo.agent.agent import Agent
from agiwo.llm.deepseek import DeepseekModel
from agiwo.scheduler.loop import SchedulerLoop
from agiwo.scheduler.registry import ModelRegistry, ToolRegistry
from agiwo.scheduler.store import SQLiteAgentStateStorage
from agiwo.agent.storage.sqlite import SQLiteRunStepStorage
from agiwo.tool.scheduling import SpawnAgentTool, SleepAndWaitTool, QuerySpawnedAgentTool

# --- 初始化 Storage ---
run_step_storage_config = RunStepStorageConfig(storage_type="sqlite", config={"db_path": "agent.db"})
state_store = SQLiteAgentStateStorage(db_path="agent_states.db")

# --- 初始化 Registry ---
model_registry = ModelRegistry()
model_registry.register("deepseek", "deepseek-chat",
    lambda params: DeepseekModel(id="deepseek-chat", name="deepseek-chat", provider="deepseek", **params))

tool_registry = ToolRegistry()  # 内置工具自动注册

# --- 创建调度工具 ---
spawn_tool = SpawnAgentTool(state_store=state_store)
sleep_tool = SleepAndWaitTool(state_store=state_store)
query_tool = QuerySpawnedAgentTool(state_store=state_store)

# --- 创建 Agent ---
agent = Agent(
    id="orchestrator",
    description="An agent that can spawn sub-agents and coordinate their work",
    model=DeepseekModel(id="deepseek-chat", name="deepseek-chat", provider="deepseek"),
    tools=[spawn_tool, sleep_tool, query_tool],
    system_prompt="You are a task coordinator. Break complex tasks into subtasks, "
                  "spawn child agents, and synthesize their results.",
    options=AgentOptions(
        max_steps=20,
        run_step_storage=run_step_storage_config,
    ),
)

# --- 启动 Scheduler (后台) ---
scheduler = SchedulerLoop(
    state_store=state_store,
    run_step_storage=run_step_storage,
    model_registry=model_registry,
    tool_registry=tool_registry,
    check_interval=5.0,
    max_concurrent=10,
)
scheduler_task = asyncio.create_task(scheduler.start())

# --- 执行 Agent ---
result = await agent.run("Research and write a report about AI agents in 2026")
# Agent 可能会:
# 1. spawn_agent(task="Research latest AI agent papers")
# 2. spawn_agent(task="Analyze current AI agent frameworks")
# 3. spawn_agent(task="Survey enterprise AI agent adoption")
# 4. sleep_and_wait(wake_type="children_complete")
# → Agent 挂起，Scheduler 启动子 Agent 并等待完成
# → 全部完成后 Agent 被唤醒，读取结果，综合生成报告
```

---

## 12. 实现顺序

| Phase | 内容 | 依赖 |
|-------|------|------|
| **P1** | `scheduler/models.py` — 数据模型 | 无 |
| **P2** | `scheduler/store.py` — AgentStateStorage + SQLite | P1 |
| **P3** | `scheduler/registry.py` — ModelRegistry + ToolRegistry | 无 |
| **P4** | `llm/base.py` — Model.provider + `agent/schema.py` — SLEEPING | 无 |
| **P5** | `tool/base.py` — terminate_execution + `executor.py` — 终止检查 | P4 |
| **P6** | `agent/agent.py` — to_blueprint() + context metadata 注入 | P1, P4 |
| **P7** | `tool/scheduling/` — 3 个调度工具 | P1, P2, P5, P6 |
| **P8** | `scheduler/builder.py` — AgentBuilder | P3 |
| **P9** | `scheduler/loop.py` — SchedulerLoop | P2, P3, P8 |
| **P10** | 端到端测试 | All |

P1-P5 可并行，P6-P7 依赖前序，P8-P9 依赖 Registry。

---

## 13. Future Work

以下内容不纳入 V1，记录待后续版本完善：

1. **Hooks 恢复**：Agent 从 Blueprint 重建后无 Hooks。方案：Scheduler 支持 `hook_factory: Callable[[str], AgentHooks]` 注册，按 agent_id 返回对应 Hooks。
2. **跨进程部署 / 多 Scheduler 竞态**：多个 Scheduler 实例可能同时唤醒同一个 Agent。方案：AgentStateStorage 层面做 atomic claim（`UPDATE ... WHERE status='sleeping' RETURNING ...`）。
3. **Blueprint 版本兼容**：代码更新后旧 Blueprint 可能无法反序列化。方案：Blueprint 增加 `version` 字段，AgentBuilder 做版本迁移。
4. **Interval 唤醒无限循环**：Agent 被 interval 唤醒后可能无限次 sleep → wake。方案：增加 `max_wake_count` 限制或 TTL。
5. **Scheduler 高可用**：Scheduler 挂掉后所有 sleeping Agent 无人唤醒。方案：健康检查 + 自动重启 + 持久化 last_check_at。
6. **并发 Agent 执行资源控制**：多个 spawned Agent 同时 LLM 调用可能超限。方案：Scheduler 支持 `max_concurrent_agents` 配置，用 semaphore 控制。
