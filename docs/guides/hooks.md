# Hooks

Hooks let you observe and intercept agent lifecycle events. They're optional callbacks organized in a dataclass.

## Hook Types

```python
from agiwo.agent.hooks import AgentHooks

hooks = AgentHooks(
    on_run_start=my_run_start_hook,
    on_run_end=my_run_end_hook,
    on_tool_start=my_tool_start_hook,
    on_tool_end=my_tool_end_hook,
    on_llm_start=my_llm_start_hook,
    on_llm_end=my_llm_end_hook,
    on_step_start=my_step_start_hook,
    on_step_end=my_step_end_hook,
    on_memory_write=my_memory_write_hook,
    on_memory_retrieve=my_memory_retrieve_hook,
)
```

## Available Hooks

### Run Lifecycle

```python
async def on_run_start(context) -> None:
    """Called when a run begins."""
    print(f"Run {context.run_id} started")

async def on_run_end(context, result) -> None:
    """Called when a run completes (success or failure)."""
    print(f"Run {context.run_id} ended: {result.response[:100]}")
```

### Tool Execution

```python
async def on_tool_start(context, tool_name, parameters) -> None:
    """Called before a tool executes."""
    print(f"Tool {tool_name} starting with {parameters}")

async def on_tool_end(context, result) -> None:
    """Called after a tool completes."""
    print(f"Tool {result.tool_name} finished in {result.duration:.2f}s")
```

### LLM Calls

```python
async def on_llm_start(context, messages) -> None:
    """Called before an LLM request."""
    print(f"LLM call with {len(messages)} messages")

async def on_llm_end(context, response) -> None:
    """Called after an LLM response."""
    print(f"LLM responded")
```

### Steps

```python
async def on_step_start(context, step) -> None:
    """Called at the start of each reasoning step."""
    print(f"Step {step} starting")

async def on_step_end(context, step, record) -> None:
    """Called at the end of each reasoning step."""
    print(f"Step {step} completed")
```

### Memory

```python
async def on_memory_write(context, path, content) -> None:
    """Called when memory is written to a file."""
    print(f"Memory written to {path}")

async def on_memory_retrieve(context, query, results) -> None:
    """Called when memory retrieval completes."""
    print(f"Memory search for '{query}' returned {len(results)} results")
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
async def tracing_on_tool_end(context, result):
    logger.info(
        "tool_executed",
        tool=result.tool_name,
        duration=result.duration,
        success=result.is_success,
    )
```

### Cost Tracking

```python
total_tokens = 0

async def cost_on_llm_end(context, response):
    global total_tokens
    if response.usage:
        total_tokens += response.usage.get("total_tokens", 0)
        print(f"Total tokens so far: {total_tokens}")
```

### Rate Limiting

```python
import time

last_call = 0

async def rate_limit_on_llm_start(context, messages):
    global last_call
    elapsed = time.time() - last_call
    if elapsed < 1.0:
        await asyncio.sleep(1.0 - elapsed)
    last_call = time.time()
```

### Debugging

```python
async def debug_on_step_end(context, step, record):
    print(f"=== Step {step} ===")
    print(f"LLM response: {record.llm_response[:200]}")
    print(f"Tool calls: {record.tool_calls}")
    print()
```

## Notes

- All hooks are async functions
- Hooks run sequentially in registration order
- A hook exception does not abort the run (it's logged and swallowed)
- Hook context provides run_id, session_id, agent_id, and other runtime metadata
