# Tool

Tools give agents the ability to interact with the external world — run shell commands, search the web, read files, call APIs, or execute custom logic.

## Tool Interface

Every tool extends `BaseTool` and implements a few attributes and methods:

```python
from agiwo.tool import BaseTool, ToolResult
from agiwo.tool import ToolContext


class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string"},
            },
            "required": ["input"],
        }

    async def execute(
        self,
        parameters: dict,
        context: ToolContext,
        abort_signal=None,
    ) -> ToolResult:
        # Your logic here
        return ToolResult.success(
            tool_name=self.name,
            content="Result for LLM",
        )
```

## ToolResult

Always use factory methods to construct results:

| Method | When to use |
|--------|-------------|
| `ToolResult.success(...)` | Tool completed successfully |
| `ToolResult.failed(...)` | Tool encountered an error |
| `ToolResult.aborted(...)` | Tool was cancelled via abort_signal |
| `ToolResult.denied(...)` | Tool execution was denied (e.g., permissions) |

### Key fields

| Field | Description |
|-------|-------------|
| `content` | Text returned to the LLM for its next reasoning step |
| `content_for_user` | Optional display text for the frontend/console |
| `output` | Raw structured output (dict, list, etc.) |
| `error` | Error message if failed |

## Tool Properties

### Caching

Set `cacheable = True` to enable session-scoped result caching. When the same tool is called with identical arguments within a session, the cached result is returned without re-execution:

```python
class ExpensiveTool(BaseTool):
    cacheable = True
    # ...
```

### Timeout

Override the default 30-second timeout:

```python
class SlowTool(BaseTool):
    timeout_seconds = 120
    # ...
```

### Concurrency

`concurrency_safe` tells the runtime whether this tool can run alongside other tools in the same batch. Set to `False` if your tool has side effects that conflict with concurrent execution (e.g., writing to a shared file).

## ToolContext

The `ToolContext` provides runtime information to tool execution:

- Workspace paths
- Agent identity
- Session information
- Storage access

## Builtin Tools

Agiwo includes these tools out of the box:

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands with security sandboxing |
| `bash_process` | Manage long-running background processes (inspect logs, stop, send input) |
| `web_search` | Search the web via multiple search engines |
| `web_reader` | Fetch and extract readable content from URLs (supports curl and Playwright) |
| `memory_retrieval` | Hybrid BM25 + vector search over MEMORY/ files |

Builtin tools are automatically included when you create an Agent. They live in `agiwo/tool/builtin/`.

## Agent-as-Tool

Agents can be wrapped as tools for composition. See [Multi-Agent Guide](../guides/multi-agent.md).

```python
researcher_tool = researcher_agent.as_tool()
```

This is different from `BaseTool` — it's an `AgentTool` adapter provided by the agent runtime, not part of `agiwo.tool` core.
