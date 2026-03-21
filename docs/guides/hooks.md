# Hooks

Hooks let you observe and intercept agent lifecycle events. They're optional callbacks organized in a dataclass.

## Hook Types

```python
from agiwo.agent.hooks import AgentHooks

hooks = AgentHooks(
    on_before_run=my_before_run_hook,
    on_after_run=my_after_run_hook,
    on_before_tool_call=my_before_tool_call_hook,
    on_after_tool_call=my_after_tool_call_hook,
    on_before_llm_call=my_before_llm_call_hook,
    on_after_llm_call=my_after_llm_call_hook,
    on_step=my_on_step_hook,
    on_memory_write=my_memory_write_hook,
    on_memory_retrieve=my_memory_retrieve_hook,
)
```

## Available Hooks

### Run Lifecycle

```python
async def on_before_run(user_input, context) -> str | None:
    """Called before a run starts."""
    print(f"Run {context.run_id} starting")
    return None

async def on_after_run(result, context) -> None:
    """Called after a run completes."""
    preview = (result.response or "")[:100]
    print(f"Run {context.run_id} ended: {preview}")
```

### Tool Execution

```python
async def on_before_tool_call(tool_call_id, tool_name, parameters) -> dict | None:
    """Called before a tool executes."""
    print(f"Tool {tool_name} starting with {parameters}")
    return None

async def on_after_tool_call(tool_call_id, tool_name, parameters, result) -> None:
    """Called after a tool completes."""
    print(f"Tool {result.tool_name} finished in {result.duration:.2f}s")
```

### LLM Calls

```python
async def on_before_llm_call(messages) -> list[dict] | None:
    """Called before an LLM request."""
    print(f"LLM call with {len(messages)} messages")
    return None

async def on_after_llm_call(step) -> None:
    """Called after an assistant step is committed."""
    print(f"Assistant step finished: {step.id}")
```

### Steps

```python
async def on_step(step) -> None:
    """Called when any step (user/assistant/tool) is committed."""
    print(f"Committed {step.role.value} step {step.sequence}")
```

### Memory

```python
async def on_memory_write(user_input, result, context) -> None:
    """Called after a successful run to persist external memory."""
    print(f"Persisting memory for run {context.run_id}")

async def on_memory_retrieve(user_input, context) -> list:
    """Called before execution to fetch memories."""
    print(f"Retrieving memories for run {context.run_id}")
    return []
```

## Using Hooks

Pass hooks when creating an Agent:

```python
agent = Agent(
    AgentConfig(name="assistant", description="...", system_prompt="..."),
    model=model,
    hooks=my_hooks,
)
```

Or update hooks after creation:

```python
agent.hooks = new_hooks
```

## Use Cases

### Logging and Tracing

```python
async def tracing_on_after_tool_call(tool_call_id, tool_name, parameters, result):
    logger.info(
        "tool_executed",
        tool_call_id=tool_call_id,
        tool=tool_name,
        parameters=parameters,
        duration=result.duration,
        success=result.is_success,
    )
```

### Cost Tracking

```python
total_tokens = 0

async def cost_on_after_llm(step):
    global total_tokens
    if step.metrics and step.metrics.total_tokens:
        total_tokens += step.metrics.total_tokens
        print(f"Total tokens so far: {total_tokens}")
```

### Rate Limiting

```python
import asyncio
import time

last_call = 0

async def rate_limit_on_before_llm(messages):
    global last_call
    elapsed = time.time() - last_call
    if elapsed < 1.0:
        await asyncio.sleep(1.0 - elapsed)
    last_call = time.time()
```

### Debugging

```python
async def debug_on_step(step):
    print(f"=== Step {step.sequence} ({step.role.value}) ===")
    print(f"Content: {step.content}")
    print(f"Tool calls: {step.tool_calls}")
    print()
```

## Notes

- All hooks are async functions
- Each lifecycle point has a single callback slot on `AgentHooks`
- Hook exceptions propagate and will fail the run unless you handle them yourself
- Hook context provides run_id, session_id, agent_id, and other runtime metadata
