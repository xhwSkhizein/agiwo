# Agent API Reference

## `Agent`

```python
class Agent:
    def __init__(
        self,
        config: AgentConfig,
        *,
        model: Model,
        tools: list[RuntimeToolLike] | None = None,
        hooks: AgentHooks | None = None,
        id: str | None = None,
    ) -> None
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Unique instance ID |
| `name` | `str` | Config name |
| `description` | `str` | Config description |
| `config` | `AgentConfig` | Deep copy of configuration |
| `model` | `Model` | LLM model instance |
| `tools` | `tuple[RuntimeToolLike, ...]` | All registered tools |
| `hooks` | `AgentHooks` | Lifecycle hooks (settable) |
| `options` | `AgentOptions` | Runtime options |
| `run_step_storage` | `RunStepStorage` | Run/step persistence |
| `trace_storage` | `BaseTraceStorage \| None` | Trace storage |
| `session_storage` | `SessionStorage` | Session metadata storage |

### Methods

#### `run()`

```python
async def run(
    self,
    user_input: UserInput,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict | None = None,
    abort_signal: AbortSignal | None = None,
) -> RunOutput
```

Execute and wait for completion. Convenience wrapper over `start()` + `handle.wait()`.

#### `run_stream()`

```python
async def run_stream(
    self,
    user_input: UserInput,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict | None = None,
    abort_signal: AbortSignal | None = None,
) -> AsyncIterator[AgentStreamItem]
```

Execute and stream events.

#### `start()`

```python
def start(
    self,
    user_input: UserInput,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict | None = None,
    abort_signal: AbortSignal | None = None,
) -> AgentExecutionHandle
```

Start execution and return a handle for fine-grained control.

#### `derive_child_spec()`

```python
def derive_child_spec(
    self,
    *,
    child_id: str,
    instruction: str | None = None,
    system_prompt_override: str | None = None,
    exclude_tool_names: set[str] | None = None,
    metadata_overrides: dict | None = None,
) -> ChildAgentSpec
```

Create a child agent specification for nested execution.

#### `install_runtime_tools()`

```python
def install_runtime_tools(self, tools: list[RuntimeToolLike]) -> None
```

Add tools after agent creation.

#### `add_step_observer()` / `remove_step_observer()`

```python
def add_step_observer(self, observer: StepObserver) -> None
def remove_step_observer(self, observer: StepObserver) -> None
```

Register/unregister step observers for monitoring execution.

#### `close()`

```python
async def close(self) -> None
```

Release resources and cancel active executions.

---

## `AgentConfig`

```python
@dataclass
class AgentConfig:
    name: str
    description: str
    system_prompt: str
    options: AgentOptions = field(default_factory=AgentOptions)
```

## `AgentExecutionHandle`

```python
class AgentExecutionHandle:
    run_id: str
    session_id: str

    async def wait(self) -> RunOutput
    async def stream(self) -> AsyncIterator[AgentStreamItem]
    async def steer(self, user_input: str | UserInput) -> bool
    async def cancel(self) -> bool
```

## `RunOutput`

```python
@dataclass
class RunOutput:
    response: str                    # Final text response
    tool_results: list[ToolResult]   # All tool results from the run
    usage: dict[str, Any]            # Token usage stats
    run_id: str                      # Unique run identifier
    session_id: str                  # Session identifier
```

## `AgentStreamItem`

```python
@dataclass
class AgentStreamItem:
    delta: StreamDelta | None        # Content delta
    run_id: str                      # Run identifier
    session_id: str                  # Session identifier
    step: int                        # Step number
    type: str                        # Event type
```
