# Tool API Reference

## `BaseTool`

Abstract base class for all tools.

```python
class BaseTool(ABC):
    cacheable: bool = False
    timeout_seconds: int = 30
    concurrency_safe: bool = True

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    def get_short_description(self) -> str: ...

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]: ...

    @abstractmethod
    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult: ...

    def get_definition(self) -> ToolDefinition: ...
    def to_openai_schema(self) -> dict[str, object]: ...
```

### What to Implement

| Member | Type | Description |
|--------|------|-------------|
| `name` | `str` (class attr or property) | Tool identifier (must be unique per agent) |
| `description` | `str` (class attr or property) | Description shown to the LLM |
| `get_parameters()` | `dict` | JSON Schema for parameters |
| `execute()` | `ToolResult` | Execute the tool |

### Class Attributes

| Attribute | Default | Description |
|-----------|---------|-------------|
| `cacheable` | `False` | Enable session-scoped result caching |
| `timeout_seconds` | `30` | Execution timeout |
| `concurrency_safe` | `True` | Can run in parallel with other tools |

### Additional Methods

| Method | Description |
|--------|-------------|
| `gate(parameters, context) -> ToolGateDecision` | Preflight safety check before execution |
| `build_context(run_context, tool_call_id) -> ToolContext` | Builds `ToolContext` from agent run context |

---

## `ToolResult`

```python
@dataclass
class ToolResult:
    tool_name: str
    tool_call_id: str
    input_args: dict[str, Any]
    content: str                          # Text for LLM
    output: Any                           # Raw structured output
    start_time: float
    end_time: float
    duration: float
    content_for_user: str | None = None   # Display text for UI
    error: str | None = None
    is_success: bool = True
```

### Factory Methods

#### `ToolResult.success()`

```python
@classmethod
def success(
    cls,
    tool_name: str,
    content: str,
    tool_call_id: str = "",
    input_args: dict[str, Any] | None = None,
    start_time: float | None = None,
    output: Any = None,
    content_for_user: str | None = None,
) -> ToolResult
```

#### `ToolResult.failed()`

```python
@classmethod
def failed(
    cls,
    tool_name: str,
    error: str,
    tool_call_id: str = "",
    input_args: dict[str, Any] | None = None,
    start_time: float | None = None,
    content: str | None = None,
    output: Any = None,
) -> ToolResult
```

#### `ToolResult.aborted()`

```python
@classmethod
def aborted(
    cls,
    tool_name: str,
    tool_call_id: str = "",
    input_args: dict[str, Any] | None = None,
    start_time: float | None = None,
) -> ToolResult
```

#### `ToolResult.denied()`

```python
@classmethod
def denied(
    cls,
    tool_name: str,
    reason: str,
    tool_call_id: str = "",
    input_args: dict[str, Any] | None = None,
    start_time: float | None = None,
    content: str | None = None,
    output: Any = None,
) -> ToolResult
```

---

## `ToolDefinition`

```python
@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    is_concurrency_safe: bool = True
    timeout_seconds: int = 30
    cacheable: bool = False
```

---

## `ToolContext`

```python
@dataclass(frozen=True)
class ToolContext:
    session_id: str
    agent_id: str | None = None
    agent_name: str | None = None
    user_id: str | None = None
    timeout_at: float | None = None
    depth: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    gate_checked: bool = False
    tool_call_id: str = ""
```

---

## `AbortSignal`

```python
class AbortSignal:
    def is_aborted(self) -> bool: ...
```

Pass to long-running tools and check periodically for cancellation.
