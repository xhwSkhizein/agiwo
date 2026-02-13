# Scheduler 设计文档

> Agiwo Agent 调度系统的架构设计与实现细节。

## 1. 概述

### 目标

为 Agiwo Agent SDK 提供异步编排能力：

- Agent 在运行中派生独立子 Agent 执行不同任务
- Agent 可挂起自身，在条件满足时被调度器唤醒恢复执行
- 通过持久化状态实现跨时间的 Agent 生命周期管理
- Root Agent 可持久化存活，持续接收新任务
- 集中式护栏防止资源失控（深度、子 Agent 数量、唤醒次数、超时）
- 子 Agent 即使超时也能产出摘要报告

### 设计原则

- **Agent 不感知调度**：Agent 通过工具（spawn / sleep / query）与调度系统交互，Agent 核心代码无需知道调度器存在
- **单向依赖**：`scheduler/ → agent/`，无循环依赖
- **Session 连续性复用**：Agent 唤醒 = 对已有 session 发起新 `agent.run(wake_message, session_id=same_session)`
- **集中式护栏**：所有限制检查收敛到 `TaskGuard`，工具和 Scheduler 不自行检查
- **Guaranteed Summary**：`enable_termination_summary` 默认 True + 超时唤醒机制，确保产出

---

## 2. 模块结构

```
agiwo/scheduler/
├── __init__.py        # 公开导出
├── models.py          # 数据模型: AgentState, WakeCondition, WakeType, WaitMode,
│                      #           TimeUnit, TaskLimits, SchedulerConfig, AgentStateStorageConfig
├── store.py           # AgentStateStorage ABC + InMemoryAgentStateStorage + SQLiteAgentStateStorage
├── guard.py           # TaskGuard: 集中式护栏
├── tools.py           # SpawnAgentTool, SleepAndWaitTool, QuerySpawnedAgentTool
└── scheduler.py       # Scheduler: 编排层入口
```

---

## 3. 核心数据模型

> 所有模型位于 `agiwo/scheduler/models.py`

### 3.1 WakeType & WaitMode

```python
class WakeType(str, Enum):
    WAITSET = "waitset"              # 等待指定子 Agent 完成
    TIMER = "timer"                  # 一次性延迟唤醒
    PERIODIC = "periodic"            # 周期性唤醒
    TASK_SUBMITTED = "task_submitted" # 新任务提交（persistent agent 专用）

class WaitMode(str, Enum):
    ALL = "all"   # 等待所有指定子 Agent 完成
    ANY = "any"   # 任一指定子 Agent 完成即唤醒
```

### 3.2 WakeCondition

唤醒条件，描述 SLEEPING Agent 何时应被唤醒。

```python
@dataclass
class WakeCondition:
    type: WakeType

    # WAITSET 字段
    wait_for: list[str]          # 等待的子 Agent state_id 列表
    wait_mode: WaitMode          # ALL 或 ANY
    completed_ids: list[str]     # 已完成的子 Agent ID（Scheduler 自动维护）

    # TIMER / PERIODIC 字段
    time_value: float | None     # 时间数值
    time_unit: TimeUnit | None   # 时间单位 (seconds/minutes/hours)
    wakeup_at: datetime | None   # 计算后的唤醒时间点

    # TASK_SUBMITTED 字段
    submitted_task: str | None   # 提交的新任务内容

    # 超时保护（WAITSET / PERIODIC 通用）
    timeout_at: datetime | None  # 超时时间点，防止永久 sleep
```

**满足条件判断** (`is_satisfied`):

| WakeType | 满足条件 |
|----------|----------|
| WAITSET (ALL) | `set(wait_for) ⊆ set(completed_ids)` |
| WAITSET (ANY) | `set(wait_for) ∩ set(completed_ids) ≠ ∅` |
| TIMER / PERIODIC | `now >= wakeup_at` |
| TASK_SUBMITTED | `submitted_task is not None` |

**超时判断** (`is_timed_out`): `timeout_at is not None and now >= timeout_at`

### 3.3 AgentState

调度系统的核心状态记录。一个 AgentState 代表一个可被调度的 Agent 生命周期。

```python
@dataclass
class AgentState:
    id: str                            # 主键 (= agent_id)
    session_id: str                    # 会话 ID，跨 sleep/wake 不变
    agent_id: str                      # Agent 逻辑 ID
    parent_agent_id: str               # 父 Agent ID
    status: AgentStateStatus           # PENDING/RUNNING/SLEEPING/COMPLETED/FAILED
    task: str                          # 任务描述

    parent_state_id: str | None        # 父 Agent 的 state_id（信号传播用）
    config_overrides: dict             # 子 Agent 配置覆盖 (system_prompt 等)
    wake_condition: WakeCondition | None
    result_summary: str | None         # 完成后的最终响应文本
    signal_propagated: bool            # 完成信号是否已传播给父 Agent

    is_persistent: bool                # 是否为持久化 root agent
    depth: int                         # spawn 深度 (root=0)
    wake_count: int                    # 已唤醒次数（防活锁）

    created_at: datetime
    updated_at: datetime
```

**生命周期状态转换：**

```
                                    ┌─────────────────────────────────┐
                                    │         Persistent Agent        │
                                    │  SLEEPING ←──→ RUNNING ──→ ... │
                                    │     ↑ submit_task()             │
                                    └─────────────────────────────────┘

PENDING ──→ RUNNING ──→ COMPLETED
                │
                ├──→ SLEEPING ──→ RUNNING ──→ COMPLETED
                │        │            │
                │        │            ├──→ SLEEPING ──→ ...
                │        │            │
                │        │            └──→ FAILED
                │        │
                │        └──→ (timeout) ──→ RUNNING ──→ COMPLETED
                │
                └──→ FAILED
```

### 3.4 TaskLimits

运行时限制，由 `TaskGuard` 强制执行。

```python
@dataclass
class TaskLimits:
    max_depth: int = 5                    # 最大 spawn 深度
    max_children_per_agent: int = 10      # 每个 Agent 最大活跃子 Agent 数
    default_wait_timeout: float = 600.0   # 默认等待超时 (秒)
    max_wake_count: int = 20              # 最大唤醒次数 (防活锁)
```

### 3.5 SchedulerConfig

```python
@dataclass
class SchedulerConfig:
    state_storage: AgentStateStorageConfig   # 存储配置 (memory/sqlite)
    check_interval: float = 5.0             # 调度循环间隔 (秒)
    max_concurrent: int = 10                # 最大并发 Agent 数
    graceful_shutdown_wait_seconds: int = 30 # 优雅关闭等待时间
    task_limits: TaskLimits                  # 运行时限制
```

---

## 4. AgentStateStorage

> 位于 `agiwo/scheduler/store.py`

### 4.1 ABC 接口

```python
class AgentStateStorage(ABC):
    # CRUD
    async def save_state(state: AgentState) -> None
    async def get_state(state_id: str) -> AgentState | None
    async def get_states_by_parent(parent_state_id: str) -> list[AgentState]
    async def update_status(state_id, status, *, wake_condition?, result_summary?) -> None

    # 调度查询
    async def find_pending() -> list[AgentState]
    async def find_wakeable(now: datetime) -> list[AgentState]

    # 信号传播
    async def find_unpropagated_completed() -> list[AgentState]  # 含 FAILED
    async def mark_child_completed(parent_state_id, child_id) -> None
    async def mark_propagated(state_id) -> None

    # 超时 & 唤醒计数
    async def find_timed_out(now: datetime) -> list[AgentState]
    async def increment_wake_count(state_id) -> None

    # Console 用
    async def list_all(*, status?, limit?, offset?) -> list[AgentState]
```

### 4.2 实现

- **InMemoryAgentStateStorage**：内存字典，测试和开发用
- **SQLiteAgentStateStorage**：SQLite 持久化，生产用

SQLite 表结构核心列：

```sql
CREATE TABLE agent_states (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    parent_agent_id TEXT NOT NULL,
    parent_state_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    task TEXT NOT NULL,
    config_overrides TEXT,          -- JSON
    wake_type TEXT,
    wake_wait_for TEXT,             -- JSON array
    wake_wait_mode TEXT DEFAULT 'all',
    wake_completed_ids TEXT,        -- JSON array
    wake_time_value REAL,
    wake_time_unit TEXT,
    wake_wakeup_at TEXT,
    wake_submitted_task TEXT,
    wake_timeout_at TEXT,
    result_summary TEXT,
    signal_propagated INTEGER DEFAULT 0,
    is_persistent INTEGER DEFAULT 0,
    depth INTEGER DEFAULT 0,
    wake_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

---

## 5. TaskGuard — 集中式护栏

> 位于 `agiwo/scheduler/guard.py`

`TaskGuard` 是所有调度限制检查的唯一入口。Tools 和 Scheduler 不自行实现限制逻辑。

```python
class TaskGuard:
    def __init__(self, limits: TaskLimits, store: AgentStateStorage): ...

    async def check_spawn(parent_state) -> str | None
        # 检查 depth < max_depth
        # 检查 active_children < max_children_per_agent
        # 返回 None=允许, str=拒绝原因

    async def check_wake(state) -> str | None
        # 检查 wake_count < max_wake_count
        # 返回 None=允许, str=拒绝原因

    async def find_timed_out(now) -> list[AgentState]
        # 委托给 store.find_timed_out()
```

**设计要点**：
- `check_spawn` 只计算活跃子 Agent（PENDING/RUNNING/SLEEPING），已完成的不算
- `check_wake` 防止活锁：Agent 反复 sleep/wake 超过阈值时强制 FAILED
- `find_timed_out` 在每次 tick 中调用，发现超时的 SLEEPING Agent

---

## 6. 调度工具

> 位于 `agiwo/scheduler/tools.py`
>
> 三个工具由 Scheduler 通过 `_prepare_agent()` 注入，Agent 无需手动添加。

### 6.1 SpawnAgentTool

**职责**：在 `AgentStateStorage` 中创建一条 `PENDING` 状态的 `AgentState`。

**执行流程**：
1. 从 `context.agent_id` 获取父 Agent ID
2. 从 Store 读取父 Agent 的 `AgentState`
3. 调用 `guard.check_spawn(parent_state)` — 检查深度和子 Agent 数量限制
4. 创建子 `AgentState`，`depth = parent.depth + 1`
5. 写入 Store，返回 `child_id`

**参数**：
- `task` (required): 子 Agent 任务描述
- `system_prompt` (optional): 覆盖子 Agent 的 system prompt
- `child_id` (optional): 显式指定子 Agent ID，默认自动生成

### 6.2 SleepAndWaitTool

**职责**：将当前 Agent 状态设为 SLEEPING，设置唤醒条件，通过 `TerminationReason.SLEEPING` 终止执行循环。

**参数**：
- `wake_type` (required): `"waitset"` / `"timer"` / `"periodic"`
- `wait_mode` (optional): `"all"` / `"any"`，仅 waitset 模式
- `wait_for` (optional): 显式指定等待的子 Agent ID 列表，省略则等待所有子 Agent
- `delay_seconds` (required for timer/periodic): 延迟/间隔秒数
- `time_unit` (optional): `"seconds"` / `"minutes"` / `"hours"`
- `timeout` (optional): 超时秒数，waitset 默认使用 `TaskLimits.default_wait_timeout`

**WAITSET 模式执行流程**：
1. 确定 `wait_for` 列表（显式指定或查询所有子 Agent）
2. 扫描已完成的子 Agent，填充 `completed_ids`
3. 计算 `timeout_at = now + timeout`
4. 更新 AgentState 为 SLEEPING + WakeCondition
5. 返回 `TerminationReason.SLEEPING` 终止执行

### 6.3 QuerySpawnedAgentTool

**职责**：查询子 Agent 的状态和执行结果。

**参数**：
- `agent_id` (required): 要查询的子 Agent ID

**返回**：Agent ID、状态、任务描述、结果摘要。

---

## 7. Scheduler — 编排层

> 位于 `agiwo/scheduler/scheduler.py`

### 7.1 公开 API

| 方法 | 说明 |
|------|------|
| `run(agent, input, persistent?)` | 阻塞执行 = submit + wait_for |
| `submit(agent, input, persistent?)` | 非阻塞提交，返回 state_id |
| `submit_task(state_id, task)` | 向持久化 Agent 提交新任务 |
| `wait_for(state_id, timeout?)` | 阻塞等待完成 |
| `get_state(state_id)` | 查询状态 |
| `cancel(state_id, reason?)` | 硬取消（递归） |
| `shutdown(state_id)` | 优雅关闭（递归，让 Agent 生成摘要） |

### 7.2 Agent 准备

`_prepare_agent(agent)`:
1. 注入三个调度工具（幂等）
2. 强制 `agent.options.enable_termination_summary = True`
3. 注册到 `_agents` 字典

`_create_child_agent(state)`:
1. 从 `_agents` 查找父 Agent
2. 复制父 Agent 配置（model、tools、hooks）
3. 应用 `config_overrides`（如 system_prompt）
4. 强制 `enable_termination_summary = True`

### 7.3 调度循环

```
_loop() → 每 check_interval 秒执行一次 _tick()

_tick():
  1. _propagate_signals()    — 子 Agent 完成/失败 → 父 Agent.completed_ids
  2. _enforce_timeouts()     — 超时的 SLEEPING Agent → 唤醒生成摘要
  3. _start_pending()        — PENDING Agent → 创建子 Agent 实例并执行
  4. _wake_sleeping()        — 满足条件的 SLEEPING Agent → 唤醒并注入结果
```

### 7.4 信号传播

```
子 Agent COMPLETED/FAILED
  → find_unpropagated_completed()
  → mark_child_completed(parent_state_id, child_id)  // 追加到 completed_ids
  → mark_propagated(child_id)
  → 下一次 tick 中 find_wakeable() 检查 is_satisfied()
```

### 7.5 Wake Message 构建

`_build_wake_message(state)` 根据 WakeType 构建不同的唤醒消息：

| WakeType | 消息内容 |
|----------|----------|
| WAITSET | 自动收集子 Agent 结果，注入 `## Child Agent Results` |
| TIMER | "The scheduled delay has elapsed." |
| PERIODIC | "A scheduled periodic check has triggered." |
| TASK_SUBMITTED | 注入 `submitted_task` 内容 |

**WAITSET 自动结果注入**：唤醒时调用 `_collect_child_results(state)` 从 Store 读取所有 `wait_for` 中子 Agent 的 `result_summary`，直接注入 wake message。Agent 无需手动调用 `query_spawned_agent`。

### 7.6 Agent 输出处理

`_handle_agent_output(state, output)` 决定 Agent 执行后的状态转换：

```
output.termination_reason == SLEEPING?
  → 返回（sleep_tool 已更新状态）

原始 WakeCondition 是 PERIODIC?
  → 计算下一次 wakeup_at，重新进入 SLEEPING

Agent 是 persistent?
  → 进入 SLEEPING + WakeCondition(TASK_SUBMITTED)，等待新任务

否则 → COMPLETED
```

### 7.7 超时处理

```
_enforce_timeouts():
  → guard.find_timed_out(now)
  → 对每个超时 Agent 调用 _wake_for_timeout(state)

_wake_for_timeout(state):
  1. 收集已有的子 Agent 部分结果
  2. 构建超时唤醒消息: "Wait timeout reached. Completed: N/M. [partial results]"
  3. 清除 wake_condition，设为 RUNNING
  4. 执行 agent.run(timeout_message)
  5. Agent 在 enable_termination_summary=True 下产出摘要
```

### 7.8 Persistent Root Agent

```
submit(agent, input, persistent=True):
  → AgentState(is_persistent=True, depth=0)
  → Agent 执行完成后 → _handle_agent_output → SLEEPING + TASK_SUBMITTED
  → wait_for() 检测到 persistent + SLEEPING → 返回当前结果

submit_task(state_id, task):
  → 验证: is_persistent=True, status=SLEEPING
  → 更新 WakeCondition(TASK_SUBMITTED, submitted_task=task)
  → 下一次 tick 中 find_wakeable() 发现并唤醒

shutdown(state_id):
  → 递归关闭所有后代
  → SLEEPING Agent: 注入 shutdown 消息唤醒，让其生成最终报告
  → PENDING Agent: 直接标记 FAILED
```

### 7.9 Cancel & Shutdown

**cancel(state_id)** — 硬取消：
1. 触发 AbortSignal
2. 递归取消所有活跃后代（DFS）
3. 所有节点标记 FAILED + reason

**shutdown(state_id)** — 优雅关闭：
1. 递归关闭所有活跃后代（DFS，先子后父）
2. SLEEPING Agent → 注入 shutdown 消息唤醒，让其生成最终摘要
3. PENDING Agent → 直接标记 FAILED
4. RUNNING Agent → 等待当前执行完成（通过 enable_termination_summary 保证摘要）

---

## 8. 与 Agent 核心的交互

### 8.1 ToolResult.termination_reason

`ToolResult` 包含 `termination_reason: TerminationReason | None` 字段。`SleepAndWaitTool` 返回 `TerminationReason.SLEEPING`。

`AgentExecutor._execute_tools()` 执行后检查：如果任何 ToolResult 包含 `termination_reason`，立即终止执行循环。

### 8.2 enable_termination_summary

`AgentOptions.enable_termination_summary` 默认 `True`。当 Agent 因 TIMEOUT、MAX_STEPS 等原因终止时，`AgentExecutor` 在 `finally` 块中调用 `_maybe_generate_summary()` 生成摘要。

Scheduler 强制所有管理的 Agent 开启此选项，确保：
- 子 Agent 超时 → 产出摘要 → 父 Agent 可读取
- 优雅关闭 → Agent 产出最终报告

### 8.3 AbortSignal

每个 Agent 执行关联一个 `AbortSignal`。Scheduler 在 `_abort_signals` 字典中追踪。

- `cancel()` 触发 signal → Agent 执行被取消
- 子 Agent 启动前检查父 Agent 的 signal，如已取消则不启动

---

## 9. Console 集成

Console（FastAPI + Next.js）通过以下方式集成 Scheduler：

### 9.1 API 层

- `GET /api/scheduler/states` — 列表（支持 status 过滤 + 分页）
- `GET /api/scheduler/states/{id}` — 详情
- `GET /api/scheduler/states/{id}/children` — 子 Agent 列表
- `GET /api/scheduler/stats` — 统计
- `POST /api/scheduler/chat/{agent_id}` — SSE 实时对话
- `POST /api/scheduler/chat/{agent_id}/cancel` — 取消

### 9.2 响应模型

`WakeConditionResponse` 包含所有 V3 字段：`wait_for`, `wait_mode`, `completed_ids`, `submitted_task`, `timeout_at`。

`AgentStateResponse` / `AgentStateListItem` 包含：`is_persistent`, `depth`, `wake_count`。

---

## 10. 测试覆盖

| 测试文件 | 覆盖范围 |
|----------|----------|
| `test_models.py` | WakeCondition 满足/超时判断、序列化、TaskLimits 默认值 |
| `test_store.py` | InMemory 存储的 CRUD、查询、信号传播、超时、唤醒计数 |
| `test_guard.py` | TaskGuard 的 spawn/wake 限制检查、超时发现 |
| `test_tools.py` | SpawnAgentTool 限制检查、SleepAndWaitTool 各模式、QuerySpawnedAgentTool |
| `test_scheduler.py` | Scheduler 生命周期、Agent 准备、提交、唤醒消息、信号传播、submit_task、cancel、shutdown |
| `test_scheduler_api.py` | Console API 端点（列表、详情、子 Agent、统计） |
| `test_scheduler_chat_api.py` | Console Chat SSE、取消级联 |

**总计**：SDK 134 tests + Console 18 tests = **152 tests**
