# Scheduler API Reference

## `Scheduler`

```python
class Scheduler:
    def __init__(self, config: SchedulerConfig | None = None) -> None

    async def __aenter__(self) -> Scheduler
    async def __aexit__(self, *args) -> None
```

### Lifecycle

```python
async with Scheduler() as scheduler:
    # Use scheduler
    pass
# Automatically stops on exit
```

Or manually:

```python
scheduler = Scheduler()
await scheduler.start()
# ... use scheduler ...
await scheduler.stop()
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `store` | `AgentStateStorage` | Underlying state storage (read-only access) |

### Methods

#### `run()`

```python
async def run(
    self,
    agent: Agent,
    user_input: UserInput,
    *,
    session_id: str | None = None,
    timeout: float | None = None,
    abort_signal: AbortSignal | None = None,
    persistent: bool = False,
) -> RunOutput
```

Submit an agent and wait for completion.

#### `submit()`

```python
async def submit(
    self,
    agent: Agent,
    user_input: UserInput,
    *,
    session_id: str | None = None,
    abort_signal: AbortSignal | None = None,
    persistent: bool = False,
    agent_config_id: str | None = None,
) -> str
```

Submit an agent and return the state ID immediately.

#### `enqueue_input()`

```python
async def enqueue_input(
    self,
    state_id: str,
    user_input: UserInput,
    *,
    agent: Agent | None = None,
) -> None
```

Add input to a running or idle agent.

#### `stream()`

```python
async def stream(
    self,
    user_input: UserInput,
    *,
    agent: Agent | None = None,
    state_id: str | None = None,
    session_id: str | None = None,
    abort_signal: AbortSignal | None = None,
    persistent: bool = False,
    agent_config_id: str | None = None,
    timeout: float | None = None,
    include_child_events: bool = True,
) -> AsyncIterator[AgentStreamItem]
```

Stream events from an agent execution.

#### `wait_for()`

```python
async def wait_for(
    self,
    state_id: str,
    timeout: float | None = None,
) -> RunOutput
```

Wait for an agent to complete.

#### `get_state()`

```python
async def get_state(self, state_id: str) -> AgentState | None
```

Get the current state of an agent.

#### `steer()`

```python
async def steer(
    self,
    state_id: str,
    user_input: UserInput,
    *,
    urgent: bool = False,
) -> bool
```

Send steering input to a running agent.

#### `cancel()`

```python
async def cancel(self, state_id: str, reason: str = "Cancelled by user") -> bool
```

Cancel a running agent.

#### `shutdown()`

```python
async def shutdown(self, state_id: str) -> bool
```

Terminate and clean up an agent.

#### `get_registered_agent()`

```python
def get_registered_agent(self, state_id: str) -> Agent | None
```

Get the Agent instance associated with a state ID.

---

## `SchedulerConfig`

```python
@dataclass
class SchedulerConfig:
    check_interval: float = 1.0
    max_concurrent: int = 10
    state_storage: str = "memory"  # "memory" or "sqlite"
    task_limits: dict = field(default_factory=dict)
    graceful_shutdown_wait_seconds: float = 30.0
```

## `AgentState`

```python
@dataclass
class AgentState:
    state_id: str
    agent_id: str
    status: AgentStateStatus  # IDLE, RUNNING, WAITING, QUEUED, COMPLETED, FAILED
    created_at: float
    last_activity_at: float | None = None
    # ... other fields
```

### Agent States

| State | Description |
|-------|-------------|
| `IDLE` | Agent is alive and waiting for input |
| `RUNNING` | Agent is actively executing |
| `WAITING` | Agent is waiting for a child or event |
| `QUEUED` | Agent is queued for execution |
| `COMPLETED` | Agent finished successfully |
| `FAILED` | Agent terminated with an error |
