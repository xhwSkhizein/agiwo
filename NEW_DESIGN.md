# Agent 调度系统设计方案 v2

## 1. 概述

### 与 v1 的核心区别

v1 要求用户手动管理 StateStore、ModelRegistry、ToolRegistry、Blueprint、SchedulerLoop、3 个 Tool 实例共 8+ 个组件。v2 将所有复杂性封装进 `Scheduler` 类：

| | v1 | v2 |
|---|---|---|
| 用户管理的组件 | StateStore, Registry×2, Tools×3, SchedulerLoop, Agent | **Scheduler, Agent** |
| 用户代码行数 | ~30 行 | **~10 行** |
| Tool 注入 | 手动创建并传入 | **自动注入** |
| Agent 重建 | Blueprint + Registry 反序列化 | **直引用（V1 同进程）** |
| 子 Agent 创建 | Blueprint deep-copy + Builder | **复用父 Agent 的 model/tools 引用** |

### 设计原则

- **用户 API 极简**：创建 Scheduler → 传给 Agent → async with 运行，3 步完成
- **同进程直引用**：V1 不做序列化，Scheduler 直接持有 Agent 实例，子 Agent 共享父 Agent 的 model/tools（它们是无状态的）
- **工具自动注入**：Agent 构造时传入 scheduler，自动获得 spawn/sleep/query 三个工具
- **无状态工具**：工具通过 per-execution 的 `context.metadata` 获取调度上下文，可被多 Agent 安全共享
- **进程透明准备**：数据模型预留跨进程扩展能力，V1 不实现

### V1 范围

**包含：**
- `Scheduler` 类（封装 state store + loop + child agent 创建 + tool 管理）
- 3 个自动注入的调度工具（spawn_agent / sleep_and_wait / query_spawned_agent）
- AgentState 持久化（SQLite）
- 并行 Agent 执行（`asyncio.Semaphore`）

**不含（Future Work）：**
- 跨进程 Blueprint 序列化 + Registry 重建
- Hooks 恢复
- 进程重启后 sleeping Agent 恢复
- Scheduler 高可用 / 多实例竞态

---

## 2. 用户 API

```python
from agiwo import Agent, Scheduler
from agiwo.llm.deepseek import DeepseekModel

# 1. 创建 Scheduler
scheduler = Scheduler(db_path="scheduler.db")

# 2. 创建 Agent（自动注入 spawn/sleep/query 工具）
agent = Agent(
    id="orchestrator",
    model=DeepseekModel(id="deepseek-chat", name="deepseek-chat", provider="deepseek"),
    system_prompt="You are a task coordinator. Break complex tasks into subtasks, "
                  "spawn child agents, and synthesize their results.",
    scheduler=scheduler,
)

# 3. 在 Scheduler 上下文中运行
async with scheduler:
    result = await agent.run("Research and write a report about AI agents in 2026")
    # Agent 自动获得 spawn_agent / sleep_and_wait / query_spawned_agent 工具
    # 可能的执行过程：
    # → spawn_agent(task="Research latest AI agent papers")
    # → spawn_agent(task="Analyze current AI agent frameworks")
    # → sleep_and_wait(wake_type="children_complete")
    # → [Scheduler 并行启动子 Agent，等待完成，唤醒父 Agent]
    # → query_spawned_agent(state_id="...", include_result=True)
    # → 综合生成最终报告
```

---

## 3. Scheduler 类

> 位于 `agiwo/scheduler/scheduler.py`，是整个调度系统的**唯一用户入口**。

```python
class Scheduler:

    def __init__(
        self,
        db_path: str | None = None,     # None = in-memory
        check_interval: float = 5.0,
        max_concurrent: int = 10,
    ) -> None:
        self._store: AgentStateStorage = (
            SQLiteAgentStateStorage(db_path) if db_path
            else InMemoryAgentStateStorage()
        )
        self._agents: dict[str, "Agent"] = {}   # agent_id → Agent 实例
        self._check_interval = check_interval
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tools: list[BaseTool] | None = None
        self._running = False
        self._task: asyncio.Task | None = None

    # ── Agent 注册 ──────────────────────────────────────

    def register(self, agent: "Agent") -> None:
        """注册 Agent 实例。由 Agent.__init__ 自动调用。"""
        self._agents[agent.id] = agent

    def get_agent(self, agent_id: str) -> "Agent":
        """按 ID 获取已注册的 Agent 实例。"""
        return self._agents[agent_id]

    # ── 工具管理 ──────────────────────────────────────

    def get_tools(self) -> list[BaseTool]:
        """返回调度工具集。延迟创建，所有 Agent 共享同一组实例（无状态）。"""
        if self._tools is None:
            self._tools = [
                SpawnAgentTool(self._store),
                SleepAndWaitTool(self._store),
                QuerySpawnedAgentTool(self._store),
            ]
        return self._tools

    # ── 子 Agent 创建 ──────────────────────────────────

    def create_child_agent(
        self,
        parent_id: str,
        child_id: str,
        overrides: dict[str, Any],
    ) -> "Agent":
        """从父 Agent 配置 + overrides 创建子 Agent。

        子 Agent 共享父 Agent 的 model 和 tools 引用（它们是无状态的）。
        子 Agent 自动注册到 Scheduler 并获得调度工具。
        """
        parent = self._agents[parent_id]
        return Agent(
            id=child_id,
            description=overrides.get("description", parent.description),
            model=parent.model,                    # 共享引用
            tools=list(parent.tools),              # 浅拷贝，共享实例
            system_prompt=overrides.get("system_prompt", parent.system_prompt),
            options=self._merge_options(parent.options, overrides),
            scheduler=self,                        # 子 Agent 也注册到同一 Scheduler
        )

    def _merge_options(
        self, parent_opts: "AgentOptions", overrides: dict
    ) -> "AgentOptions":
        """合并父 Agent options 与 overrides。"""
        # max_steps, run_timeout, max_tokens 等可被覆盖
        # 未指定的字段继承父 Agent
        ...

    # ── Context Manager ──────────────────────────────────

    async def __aenter__(self) -> "Scheduler":
        self._running = True
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *args) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
        await self._store.close()

    # ── 调度循环 ──────────────────────────────────────

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._tick()
            except Exception:
                pass  # log error, continue loop
            await asyncio.sleep(self._check_interval)

    async def _tick(self) -> None:
        await self._propagate_signals()
        await self._start_pending()
        await self._wake_sleeping()

    # ── 启动 PENDING Agent ──────────────────────────────

    async def _start_pending(self) -> None:
        pending = await self._store.find_pending()
        if not pending:
            return
        tasks = [self._run_agent(s) for s in pending]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_agent(self, state: AgentState) -> None:
        async with self._semaphore:
            await self._store.update_status(state.id, AgentStateStatus.RUNNING)
            try:
                child = self.create_child_agent(
                    state.parent_agent_id, state.agent_id, state.config_overrides
                )
                output = await child.run(state.task, session_id=state.session_id)
                if output.termination_reason == TerminationReason.SLEEPING:
                    pass  # sleep_tool 已更新 state
                else:
                    await self._store.update_status(
                        state.id, AgentStateStatus.COMPLETED,
                        result_summary=output.response,
                    )
            except Exception as e:
                await self._store.update_status(
                    state.id, AgentStateStatus.FAILED, result_summary=str(e),
                )

    # ── 唤醒 SLEEPING Agent ──────────────────────────────

    async def _wake_sleeping(self) -> None:
        now = time.time()
        wakeable = await self._store.find_wakeable(now)
        if not wakeable:
            return
        tasks = [self._wake_agent(s, now) for s in wakeable]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _wake_agent(self, state: AgentState, now: float) -> None:
        async with self._semaphore:
            await self._store.update_status(
                state.id, AgentStateStatus.RUNNING, last_wake_at=now,
            )
            agent = self._agents.get(state.agent_id)
            if not agent:
                await self._store.update_status(
                    state.id, AgentStateStatus.FAILED,
                    result_summary=f"Agent {state.agent_id} not found in registry",
                )
                return

            wake_message = self._build_wake_message(state)
            try:
                output = await agent.run(
                    wake_message,
                    session_id=state.session_id,
                    metadata={SCHEDULING_STATE_ID_KEY: state.id},
                )
                if output.termination_reason == TerminationReason.SLEEPING:
                    pass
                elif output.termination_reason == TerminationReason.COMPLETED:
                    await self._store.update_status(
                        state.id, AgentStateStatus.COMPLETED,
                        result_summary=output.response,
                    )
                else:
                    await self._store.update_status(
                        state.id, AgentStateStatus.FAILED,
                        result_summary=f"Unexpected: {output.termination_reason}",
                    )
            except Exception as e:
                await self._store.update_status(
                    state.id, AgentStateStatus.FAILED, result_summary=str(e),
                )

    # ── 信号传播 ──────────────────────────────────────

    async def _propagate_signals(self) -> None:
        """子 Agent 完成 → 递增父 Agent 的 completed_children。"""
        completed = await self._store.find_unpropagated_completed()
        for state in completed:
            if state.parent_state_id:
                await self._store.increment_completed_children(state.parent_state_id)
            await self._store.mark_propagated(state.id)

    # ── Wake Message ──────────────────────────────────

    def _build_wake_message(self, state: AgentState) -> str:
        """构建唤醒消息。只通知状态，不含子 Agent 产出内容。"""
        wc = WakeCondition.from_dict(state.wake_condition)
        if wc.type == WakeType.CHILDREN_COMPLETE:
            return (
                "<wake_signal>\n"
                "All spawned child agents have completed.\n"
                "Use query_spawned_agent to read specific results.\n"
                "</wake_signal>"
            )
        elif wc.type == WakeType.INTERVAL:
            return (
                f"<wake_signal>\n"
                f"Periodic wake-up (interval: {wc.interval_seconds}s).\n"
                f"Use query_spawned_agent to check child agent progress.\n"
                f"</wake_signal>"
            )
        elif wc.type == WakeType.DELAY:
            return (
                f"<wake_signal>\n"
                f"Scheduled wake-up reached (after {wc.delay_value} {wc.delay_unit}).\n"
                f"</wake_signal>"
            )
        return "<wake_signal>Wake-up triggered.</wake_signal>"
```

**关键设计点：**

- **`_agents` dict 是 V1 的核心**：直接持有 Agent 实例引用，创建子 Agent 时复用父 Agent 的 model/tools，零序列化开销
- **`get_tools()` 延迟单例**：所有 Agent 共享同一组工具实例，工具无状态，通过 context.metadata 获取 per-execution 上下文
- **`create_child_agent()` 共享引用**：子 Agent 的 model 和 tools 是父 Agent 的同一实例（无状态，线程安全）
- **Context Manager**：`async with scheduler:` 自动启动/停止调度循环，生命周期明确
- **`_run_agent` 兼容 sleep**：子 Agent 运行后如果 `termination_reason == SLEEPING`，说明子 Agent 也调了 sleep，不标记为 COMPLETED

---

## 4. 数据模型

> 位于 `agiwo/scheduler/models.py`

### 4.1 SchedulingMeta

per-execution 调度上下文，通过 `context.metadata["_scheduling"]` 传递给工具。

```python
@dataclass
class SchedulingMeta:
    scheduler: "Scheduler"              # Scheduler 实例引用
    agent_id: str                       # 当前执行的 Agent ID
    agent_state_id: str | None = None   # Scheduler 唤醒时注入

SCHEDULING_META_KEY = "_scheduling"
SCHEDULING_STATE_ID_KEY = "_scheduling_state_id"
```

### 4.2 WakeCondition

```python
class TimeUnit(str, Enum):
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"

class WakeType(str, Enum):
    CHILDREN_COMPLETE = "children_complete"
    INTERVAL = "interval"
    DELAY = "delay"

@dataclass
class WakeCondition:
    type: WakeType

    # INTERVAL
    interval_seconds: int | None = None

    # DELAY (LLM 输出 value + unit，程序换算)
    delay_value: int | None = None
    delay_unit: TimeUnit | None = None
    wakeup_at: float | None = None       # 程序计算，LLM 不设置

    # CHILDREN_COMPLETE (Scheduler 自动维护)
    total_children: int = 0
    completed_children: int = 0

    # 通用
    timeout_seconds: int | None = None
    last_wake_at: float | None = None

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WakeCondition": ...
```

### 4.3 AgentState

```python
class AgentStateStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SLEEPING = "sleeping"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AgentState:
    id: str                            # 稳定标识，跨 sleep/wake 不变
    session_id: str                    # 会话 ID，跨 sleep/wake 不变
    agent_id: str                      # 此 state 所属的 Agent ID
    parent_agent_id: str               # 父 Agent ID（用于 create_child_agent）
    parent_state_id: str | None        # 父 state ID（信号传播用）

    status: AgentStateStatus
    task: str                          # 任务描述
    config_overrides: dict[str, Any]   # spawn 时的覆盖配置（无 blueprint）

    wake_condition: dict[str, Any] | None
    last_run_id: str | None
    result_summary: str | None
    signal_propagated: bool            # 完成信号是否已传播给父 Agent

    created_at: datetime
    updated_at: datetime
```

**与 v1 的区别**：没有 `blueprint` 字段。V1 同进程通过 `parent_agent_id` → `Scheduler._agents` 查找父 Agent 实例 → 复用其 model/tools 创建子 Agent。

---

## 5. AgentStateStorage

> 位于 `agiwo/scheduler/store.py`，用户不直接接触，由 Scheduler 内部管理。

```python
class AgentStateStorage(ABC):

    async def close(self) -> None: ...

    @abstractmethod
    async def save_state(self, state: AgentState) -> None: ...

    @abstractmethod
    async def get_state(self, state_id: str) -> AgentState | None: ...

    @abstractmethod
    async def get_states_by_parent(self, parent_state_id: str) -> list[AgentState]: ...

    @abstractmethod
    async def find_pending(self) -> list[AgentState]: ...

    @abstractmethod
    async def find_wakeable(self, now: float) -> list[AgentState]: ...

    @abstractmethod
    async def find_unpropagated_completed(self) -> list[AgentState]: ...

    @abstractmethod
    async def increment_completed_children(self, parent_state_id: str) -> None: ...

    @abstractmethod
    async def mark_propagated(self, state_id: str) -> None: ...

    @abstractmethod
    async def update_status(
        self, state_id: str, status: AgentStateStatus, **updates
    ) -> None: ...
```

SQLite 表结构：

```sql
CREATE TABLE IF NOT EXISTS agent_states (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    parent_agent_id TEXT NOT NULL,
    parent_state_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    task TEXT NOT NULL,
    config_overrides TEXT NOT NULL DEFAULT '{}',   -- JSON
    wake_condition TEXT,                            -- JSON
    last_run_id TEXT,
    result_summary TEXT,
    signal_propagated INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_agent_states_status ON agent_states(status);
CREATE INDEX idx_agent_states_parent_state ON agent_states(parent_state_id);
CREATE INDEX idx_agent_states_propagated ON agent_states(status, signal_propagated);
```

---

## 6. 调度工具

> 位于 `agiwo/scheduler/tools.py`，由 Scheduler 内部创建，用户不直接接触。
>
> 三个工具共享同一组实例，完全无状态，通过 `context.metadata["_scheduling"]` 获取 per-execution 上下文。

### 6.1 SpawnAgentTool

```python
class SpawnAgentTool(BaseTool):

    def __init__(self, state_store: AgentStateStorage) -> None:
        self._store = state_store

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
                    "description": "Task description for the child agent.",
                },
                "config_overrides": {
                    "type": "object",
                    "description": "Optional overrides for child agent config.",
                    "properties": {
                        "system_prompt": {"type": "string"},
                        "description": {"type": "string"},
                        "max_steps": {
                            "type": "integer",
                            "description": "Max steps for the child agent. Default: 30."
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Max token budget. Default: 100000.",
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

    async def execute(self, context, parameters) -> ToolResult:
        meta: SchedulingMeta = context.metadata[SCHEDULING_META_KEY]
        child_id = f"{meta.agent_id}_{uuid4().hex[:8]}"
        state = AgentState(
            id=uuid4().hex,
            session_id=uuid4().hex,
            agent_id=child_id,
            parent_agent_id=meta.agent_id,
            parent_state_id=meta.agent_state_id,
            status=AgentStateStatus.PENDING,
            task=parameters["task"],
            config_overrides=parameters.get("config_overrides", {}),
            ...
        )
        await self._store.save_state(state)
        return ToolResult(content=f"Spawned child agent. state_id={state.id}")
```

### 6.2 SleepAndWaitTool

```python
class SleepAndWaitTool(BaseTool):

    def __init__(self, state_store: AgentStateStorage) -> None:
        self._store = state_store

    def get_name(self) -> str:
        return "sleep_and_wait"

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "wake_type": {
                    "type": "string",
                    "enum": ["children_complete", "interval", "delay"],
                },
                "interval_seconds": {"type": "integer"},
                "delay_value": {"type": "integer"},
                "delay_unit": {
                    "type": "string",
                    "enum": ["seconds", "minutes", "hours", "days"],
                },
                "timeout_seconds": {"type": "integer"},
            },
            "required": ["wake_type"],
        }

    async def execute(self, context, parameters) -> ToolResult:
        meta: SchedulingMeta = context.metadata[SCHEDULING_META_KEY]

        # 构建 WakeCondition
        wc = WakeCondition(type=WakeType(parameters["wake_type"]))
        if wc.type == WakeType.DELAY:
            unit_map = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}
            total = parameters["delay_value"] * unit_map[parameters["delay_unit"]]
            wc.delay_value = parameters["delay_value"]
            wc.delay_unit = TimeUnit(parameters["delay_unit"])
            wc.wakeup_at = time.time() + total
        elif wc.type == WakeType.INTERVAL:
            wc.interval_seconds = parameters["interval_seconds"]

        if parameters.get("timeout_seconds"):
            wc.timeout_seconds = parameters["timeout_seconds"]

        # 创建或更新 AgentState
        if meta.agent_state_id:
            # 之前被唤醒过 → 更新已有 state
            await self._store.update_status(
                meta.agent_state_id,
                AgentStateStatus.SLEEPING,
                wake_condition=wc.to_dict(),
            )
            state_id = meta.agent_state_id
        else:
            # 首次 sleep → 创建新 state
            state = AgentState(
                id=uuid4().hex,
                session_id=context.session_id,
                agent_id=meta.agent_id,
                parent_agent_id=meta.agent_id,  # self
                parent_state_id=None,
                status=AgentStateStatus.SLEEPING,
                task="[self-sleeping]",
                config_overrides={},
                wake_condition=wc.to_dict(),
                ...
            )
            # CHILDREN_COMPLETE：自动统计子 Agent 数量
            if wc.type == WakeType.CHILDREN_COMPLETE:
                children = await self._store.get_states_by_parent(state.id)
                wc.total_children = len(children)
                state.wake_condition = wc.to_dict()

            await self._store.save_state(state)
            state_id = state.id
            meta.agent_state_id = state_id  # 写回，供同次执行中的 spawn_tool 使用

        return ToolResult(
            content=f"Agent sleeping. state_id={state_id}. Wake: {wc.type.value}",
            terminate_execution=True,
        )
```

**注意**：`meta.agent_state_id = state_id` 写回 SchedulingMeta —— 如果 Agent 在同一次执行中先 sleep 再被唤醒再 sleep，state_id 保持一致。

### 6.3 QuerySpawnedAgentTool

```python
class QuerySpawnedAgentTool(BaseTool):

    def __init__(self, state_store: AgentStateStorage) -> None:
        self._store = state_store

    def get_name(self) -> str:
        return "query_spawned_agent"

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "state_id": {"type": "string"},
                "include_result": {"type": "boolean", "default": False},
            },
            "required": ["state_id"],
        }

    async def execute(self, context, parameters) -> ToolResult:
        state = await self._store.get_state(parameters["state_id"])
        if not state:
            return ToolResult(content="Agent state not found.")

        info = {
            "state_id": state.id,
            "status": state.status.value,
            "task": state.task,
        }

        if parameters.get("include_result") and state.status == AgentStateStatus.COMPLETED:
            info["result"] = state.result_summary

        return ToolResult(content=json.dumps(info, ensure_ascii=False))
```

---

## 7. Agent / Executor 改动

### 7.1 Agent 新增 `scheduler` 参数

`agiwo/agent/agent.py`：

```python
class Agent:
    def __init__(
        self,
        id: str,
        model: Model,
        # ... 现有参数 ...
        scheduler: "Scheduler | None" = None,
    ):
        # ... 现有初始化 ...
        self.scheduler = scheduler

        if scheduler is not None:
            scheduler.register(self)
            # 自动注入调度工具（去重）
            existing_names = {t.get_name() for t in self.tools}
            for tool in scheduler.get_tools():
                if tool.get_name() not in existing_names:
                    self.tools.append(tool)
```

### 7.2 执行时注入 SchedulingMeta

`Agent._execute_workflow` 开头：

```python
async def _execute_workflow(self, user_input, context, abort_signal):
    # 注入调度上下文
    if self.scheduler is not None:
        context.metadata[SCHEDULING_META_KEY] = SchedulingMeta(
            scheduler=self.scheduler,
            agent_id=self.id,
            agent_state_id=context.metadata.get(SCHEDULING_STATE_ID_KEY),
        )

    emitter = EventEmitter(context)
    # ... 后续不变 ...
```

### 7.3 TerminationReason.SLEEPING

`agiwo/agent/schema.py`：

```python
class TerminationReason(str, Enum):
    # ... 现有值 ...
    SLEEPING = "sleeping"
```

### 7.4 ToolResult.terminate_execution

`agiwo/tool/base.py`：

```python
@dataclass
class ToolResult:
    # ... 现有字段 ...
    terminate_execution: bool = False
```

### 7.5 Executor 检查终止标志

`agiwo/agent/inner/executor.py` — `_execute_tools` 末尾新增 (~5 行)：

```python
async def _execute_tools(self, state, tool_calls, abort_signal):
    # ... 现有逻辑 ...

    for result in results:
        if result.terminate_execution:
            state.termination_reason = TerminationReason.SLEEPING
            return
```

`_run_loop` 中 `_execute_tools` 调用后新增 (~2 行)：

```python
await self._execute_tools(state, step.tool_calls, abort_signal)
if state.termination_reason is not None:
    return
```

### 7.6 Model.provider 字段

`agiwo/llm/base.py`（为未来 Blueprint 序列化预留）：

```python
@dataclass
class Model(ABC):
    id: str
    name: str
    provider: str          # 显式字段，构造时传入
    # ... 现有字段 ...
```

---

## 8. 执行流程

### 8.1 Spawn + Sleep + Wake 完整流程

```
用户代码:
  scheduler = Scheduler(db_path="scheduler.db")
  agent = Agent(id="orch", ..., scheduler=scheduler)
  async with scheduler:
      result = await agent.run("Complex task")

Agent 执行:
  _execute_workflow:
    context.metadata["_scheduling"] = SchedulingMeta(scheduler, "orch", None)

  LLM → spawn_agent(task="Sub task A")
    SpawnAgentTool.execute:
      meta = context.metadata["_scheduling"]
      child_id = "orch_a1b2c3d4"
      AgentState(agent_id=child_id, parent_agent_id="orch", status=PENDING)
      → state_store.save_state()
      → return "Spawned. state_id=xxx"

  LLM → spawn_agent(task="Sub task B")
    → 同上，另一条 PENDING 记录

  LLM → sleep_and_wait(wake_type="children_complete")
    SleepAndWaitTool.execute:
      AgentState(agent_id="orch", status=SLEEPING, wake_condition={...})
      → total_children = 2
      → return ToolResult(terminate_execution=True)
    Executor: termination_reason = SLEEPING → 退出循环
  → agent.run() 返回 RunOutput(termination_reason=SLEEPING)

Scheduler._tick() (后台循环, 每 5 秒):
  _propagate_signals: (无完成的子 Agent)
  _start_pending:
    find_pending() → [state_A, state_B]
    并行:
      _run_agent(state_A):
        create_child_agent("orch", "orch_a1b2c3d4", overrides)
          → Agent(id="orch_a1b2c3d4", model=parent.model, tools=parent.tools, scheduler=self)
        child.run("Sub task A", session_id=state_A.session_id)
        → COMPLETED, result_summary = output.response
      _run_agent(state_B):
        → 同上

后续 _tick():
  _propagate_signals:
    state_A completed → parent.completed_children += 1
    state_B completed → parent.completed_children += 1
  _wake_sleeping:
    find_wakeable() → [parent_state]  (completed_children=2 >= total_children=2)
    _wake_agent:
      agent = _agents["orch"]  ← 直接从内存获取，无需重建
      wake_message = "<wake_signal>All children completed...</wake_signal>"
      agent.run(wake_message, session_id=same_session,
                metadata={"_scheduling_state_id": parent_state.id})
        → _execute_workflow:
            SchedulingMeta(scheduler, "orch", agent_state_id=parent_state.id)
            加载历史 steps (spawn + sleep 记录)
            LLM 看到完整上下文 + wake_signal
            → query_spawned_agent(state_id=..., include_result=True)
            → 综合生成最终报告
        → COMPLETED
```

### 8.2 Interval 周期检查流程

```
Agent spawn 3 children → sleep(wake_type="children_complete", interval_seconds=60)

t+60s:  interval 满足 → 唤醒
        Agent 用 query_spawned_agent 查看进度
        → 2 个还在运行
        → sleep_and_wait(wake_type="children_complete")  再次 sleep
        → meta.agent_state_id 已有值 → 更新已有 state（不创建新的）

t+120s: interval 满足 → 唤醒
        → 1 个还在运行 → 再次 sleep

t+140s: 最后一个子 Agent 完成
        → _propagate_signals → completed_children=3
        → children_complete 满足 → 立即唤醒（不等 interval）
        → Agent 读取所有结果，完成工作
```

---

## 9. 模块结构

```
agiwo/
├── scheduler/
│   ├── __init__.py           # 导出 Scheduler
│   ├── scheduler.py          # Scheduler 类（唯一用户入口）
│   ├── models.py             # SchedulingMeta, AgentState, WakeCondition, TimeUnit, WakeType
│   ├── store.py              # AgentStateStorage ABC + SQLiteAgentStateStorage
│   └── tools.py              # SpawnAgentTool, SleepAndWaitTool, QuerySpawnedAgentTool
```

**5 个文件**，用户只需导入 `Scheduler`。

---

## 10. 实现顺序

| Phase | 内容 | 改动量 |
|-------|------|--------|
| **P1** | `scheduler/models.py` — 数据模型 | 新文件 |
| **P2** | `scheduler/store.py` — AgentStateStorage + SQLite | 新文件 |
| **P3** | `scheduler/tools.py` — 3 个调度工具 | 新文件 |
| **P4** | `tool/base.py` — ToolResult.terminate_execution | +1 行 |
| **P5** | `agent/schema.py` — TerminationReason.SLEEPING | +1 行 |
| **P6** | `agent/inner/executor.py` — 终止检查 | +7 行 |
| **P7** | `llm/base.py` — Model.provider 字段 | +1 行 |
| **P8** | `agent/agent.py` — scheduler 参数 + SchedulingMeta 注入 | ~15 行 |
| **P9** | `scheduler/scheduler.py` — Scheduler 类 | 新文件 |
| **P10** | `scheduler/__init__.py` + `agiwo/__init__.py` 导出 | ~5 行 |
| **P11** | 端到端测试 | 新文件 |

P1-P3 可并行（新文件），P4-P7 可并行（最小改动），P8-P9 依赖前序。

---

## 11. Future Work

V1 不含，记录待后续版本：

1. **跨进程 Blueprint 重建**：AgentState 增加 `blueprint` 字段，新增 ModelRegistry + ToolRegistry + AgentBuilder。Scheduler 检测 `_agents` 中找不到 agent_id 时 fallback 到 Blueprint 重建
2. **进程重启恢复**：Scheduler 启动时扫描 SLEEPING/RUNNING 状态，重建 Agent 并恢复执行
3. **Hooks 恢复**：Scheduler 支持 `hook_factory` 注册，按 agent_id 返回 Hooks
4. **多 Scheduler 竞态**：AgentStateStorage 层 atomic claim（`UPDATE ... WHERE status='sleeping' RETURNING ...`）
5. **最大唤醒次数**：`WakeCondition.max_wake_count` 防止 interval 无限循环
6. **Scheduler 高可用**：健康检查 + 自动重启
