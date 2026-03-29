# Custom Tools

This guide covers how to build tools with advanced features like caching, authorization, and structured output.

## Basic Tool

The simplest tool:

```python
from agiwo.tool import BaseTool, ToolContext, ToolResult


class GreetTool(BaseTool):
    name = "greet"
    description = "Greet a person by name"

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
            },
            "required": ["name"],
        }

    async def execute(
        self,
        parameters: dict,
        context: ToolContext,
        abort_signal=None,
    ) -> ToolResult:
        name = parameters["name"]
        return ToolResult.success(
            tool_name=self.name,
            content=f"Hello, {name}!",
            content_for_user=f"Greeted {name}",
        )
```

## Handling Errors

Use `ToolResult.failed()` for expected errors:

```python
async def execute(self, parameters, context, abort_signal) -> ToolResult:
    try:
        result = await self._do_work(parameters)
        return ToolResult.success(tool_name=self.name, content=result)
    except ExternalAPIError as e:
        return ToolResult.failed(
            tool_name=self.name,
            error=f"API error: {e}",
        )
```

The LLM sees the error message and can decide to retry or use a different approach.

## Structured Output

Pass structured data via the `output` field:

```python
return ToolResult.success(
    tool_name=self.name,
    content=f"Found {len(results)} results",
    output={"results": results, "count": len(results)},
)
```

## Caching

Enable session-scoped caching for expensive operations:

```python
class EmbeddingTool(BaseTool):
    cacheable = True  # Results cached per session for identical arguments
    timeout_seconds = 60

    # ...
```

When `cacheable = True`, the runtime stores results keyed by tool name + arguments hash. Subsequent calls with the same arguments return the cached result without re-execution.

## Abort Signals

Check `abort_signal` for cancellation in long-running tools:

```python
async def execute(self, parameters, context, abort_signal) -> ToolResult:
    for item in large_dataset:
        if abort_signal and abort_signal.is_cancelled():
            return ToolResult.aborted(
                tool_name=self.name,
            )
        await process(item)
    return ToolResult.success(tool_name=self.name, content="Done")
```

## Concurrency Safety

Set `concurrency_safe` based on your tool's behavior:

- **`True`** (default): Tool can run concurrently with other tools. Good for read-only operations, API calls, pure computations.
- **`False`**: Tool must run alone. Use for tools that modify shared state, write to files, or have ordering dependencies.

The runtime batches concurrent-safe tools together for parallel execution.

## Using ToolContext

`ToolContext` provides runtime information:

```python
async def execute(self, parameters, context, abort_signal) -> ToolResult:
    # Access workspace path
    workspace = context.workspace_dir

    # Access agent identity
    agent_name = context.agent_name

    # ...
```

## Denial Pattern

For tools that need permission gating:

```python
async def execute(self, parameters, context, abort_signal) -> ToolResult:
    if not self._check_permission(parameters, context):
        return ToolResult.denied(
            tool_name=self.name,
            reason="Requires admin access",
        )
    # ...
```

The denial message guides the LLM to ask the user for permission or try a different approach.

## Registering Tools

Pass tools when creating an Agent:

```python
agent = Agent(
    AgentConfig(name="assistant", description="...", system_prompt="..."),
    model=model,
    tools=[GreetTool(), WeatherTool(), CalculatorTool()],
)
```

Prefer registering the full tool set up front so the agent definition stays explicit.
