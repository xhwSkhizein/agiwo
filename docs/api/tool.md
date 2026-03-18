# Tool API Reference

## `BaseTool`

Abstract base class for all tools.

```python
class BaseTool(ABC):
    cacheable: bool = False
    timeout_seconds: int = 30

    @abstractmethod
    def get_name(self) -> str: ...

    @abstractmethod
    def get_description(self) -> str: ...

    def get_short_description(self) -> str: ...

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]: ...

    @abstractmethod
    def is_concurrency_safe(self) -> bool: ...

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

### Methods to Implement

| Method | Returns | Description |
|--------|---------|-------------|
| `get_name()` | `str` | Tool identifier (must be unique per agent) |
| `get_description()` | `str` | Description shown to the LLM |
| `get_parameters()` | `dict` | JSON Schema for parameters |
| `is_concurrency_safe()` | `bool` | Can run in parallel with other tools |
| `execute()` | `ToolResult` | Execute the tool |

### Properties

| Property | Default | Description |
|----------|---------|-------------|
| `cacheable` | `False` | Enable session-scoped result caching |
| `timeout_seconds` | `30` | Execution timeout |

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
class ToolContext:
    """Runtime context provided to tool execution."""
    # Access workspace paths, agent identity, session info, storage
```

---

## `AbortSignal`

```python
class AbortSignal:
    def is_cancelled(self) -> bool: ...
```

Pass to long-running tools and check periodically for cancellation.
