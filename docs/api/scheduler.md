# Scheduler API Reference

## `Scheduler`

```python
class Scheduler:
    def __init__(self, config: SchedulerConfig | None = None) -> None
    async def start(self) -> None
    async def stop(self) -> None
    async def __aenter__(self) -> Scheduler
    async def __aexit__(self, *args) -> None
```

### Lifecycle

```python
async with Scheduler() as scheduler:
    ...
```

or:

```python
scheduler = Scheduler()
await scheduler.start()
...
await scheduler.stop()
```

`Scheduler` 不再暴露 `store` property。查询和控制统一走 facade API。

## Core Orchestration Methods

### `enqueue_input()`

```python
async def enqueue_input(
    self,
    state_id: str,
    user_input: UserInput,
    *,
    agent: Agent | None = None,
) -> None
```

给 persistent root 的下一轮输入赋值。当前只接受 `IDLE` 或 `FAILED` 的 persistent root；它不是消息队列，也不会并发堆积多条待处理输入。

### `route_root_input()`

```python
async def route_root_input(
    self,
    user_input: UserInput,
    *,
    agent: Agent,
    state_id: str | None = None,
    session_id: str | None = None,
    abort_signal: AbortSignal | None = None,
    persistent: bool = True,
    agent_config_id: str | None = None,
    timeout: float | None = None,
    include_child_events: bool = True,
    stream_mode: RouteStreamMode = RouteStreamMode.RUN_END,
) -> RouteResult
```

集成侧 canonical 高层入口。它会根据当前 root state 自动决定：

- `submitted`
- `enqueued`
- `steered`

返回值里的 `RouteResult.stream` 是对应 root 的流式事件迭代器。对同一个 root `state_id`，同时只允许一个活跃 subscriber。

`stream_mode` 控制 stream 生命周期：

- `RouteStreamMode.RUN_END`：root 当前这次 run 结束就关闭 stream
- `RouteStreamMode.UNTIL_SETTLED`：持续到 root 收敛到 `IDLE` / `COMPLETED` / `FAILED`

### `wait_for()`

```python
async def wait_for(
    self,
    state_id: str,
    timeout: float | None = None,
) -> RunOutput
```

等待 state 收敛到：

- `IDLE`
- `COMPLETED`
- `FAILED`

### `get_state()`

```python
async def get_state(self, state_id: str) -> AgentState | None
```

### `list_states()`

```python
async def list_states(
    self,
    *,
    statuses=None,
    parent_id: str | None = None,
    session_id: str | None = None,
    signal_propagated: bool | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[AgentState]
```

### `list_events()`

```python
async def list_events(
    self,
    *,
    target_agent_id: str | None = None,
    session_id: str | None = None,
) -> list[PendingEvent]
```

### `get_stats()`

```python
async def get_stats(self) -> dict[str, int]
```

返回各状态数量统计。

## Control Methods

### `steer()`

```python
async def steer(
    self,
    state_id: str,
    user_input: UserInput,
    *,
    urgent: bool = False,
) -> bool
```

对 `RUNNING` state 直接转给 live handle；否则会把输入记录成 `USER_HINT` event 并唤醒 scheduler loop。

### `cancel()`

```python
async def cancel(self, state_id: str, reason: str = "Cancelled by user") -> bool
```

递归取消 state 及其 active subtree。

### `shutdown()`

```python
async def shutdown(self, state_id: str) -> bool
```

对 active subtree 发起 shutdown。对 persistent root，`RUNNING/WAITING/IDLE` 会收敛到一次 `_SHUTDOWN_SUMMARY_TASK` 的 queued rerun。

### `rebind_agent()`

```python
async def rebind_agent(self, state_id: str, agent: Agent) -> bool
```

替换某个 root 的 live runtime agent。当前只接受：

- state 不存在
- `IDLE`
- `COMPLETED`
- `FAILED`

### `get_registered_agent()`

```python
def get_registered_agent(self, state_id: str) -> Agent | None
```

Returns the live Agent instance bound to `state_id`, or `None` if not found.

## Core Models

### `SchedulerConfig`

```python
@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    state_storage: AgentStateStorageConfig = ...
    check_interval: float = 1.0
    max_concurrent: int = 20
    graceful_shutdown_wait_seconds: float = 10.0
    task_limits: TaskLimits = ...
    event_debounce_min_count: int = 3
    event_debounce_max_wait_seconds: float = 10.0
    state_list_page_size: int = 1000
```

`state_storage.storage_type` 当前支持：

- `memory`
- `sqlite`
- `mongodb`

### `AgentState`

```python
@dataclass(frozen=True, slots=True)
class AgentState:
    id: str
    session_id: str
    status: AgentStateStatus
    task: UserInput
    parent_id: str | None = None
    pending_input: UserInput | None = None
    wake_condition: WakeCondition | None = None
    result_summary: str | None = None
    signal_propagated: bool = False
    agent_config_id: str | None = None
    is_persistent: bool = False
    depth: int = 0
    wake_count: int = 0
    explain: str | None = None
    created_at: datetime = ...
    updated_at: datetime = ...
```

### `RouteResult`

```python
@dataclass(frozen=True, slots=True)
class RouteResult:
    action: Literal["submitted", "enqueued", "steered"]
    state_id: str
    stream: AsyncIterator[AgentStreamItem] | None = None
```

`route_root_input()` 在所有非 `RUNNING` 路径都会返回非空 `stream`。只有 `steered + RUNNING` 场景下，`stream` 为 `None`，因为原有 live subscriber 会继续消费同一 root 的流。

### `RouteStreamMode`

```python
class RouteStreamMode(str, Enum):
    RUN_END = "run_end"
    UNTIL_SETTLED = "until_settled"
```

### `DispatchAction`

```python
@dataclass(frozen=True, slots=True)
class DispatchAction:
    state: AgentState
    reason: DispatchReason
    input_override: UserInput | None = None
    events: tuple[PendingEvent, ...] = ()
```

`DispatchAction` 是 scheduler 内部“为什么这轮要跑它”的统一表达。
