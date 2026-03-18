# Streaming

Agiwo is streaming-first. All LLM responses flow through the same streaming pipeline, whether you use `run()`, `run_stream()`, or `start()`.

## Streaming with `run_stream()`

The simplest way to get real-time output:

```python
async for event in agent.run_stream("Tell me about Python"):
    if event.delta and event.delta.content:
        print(event.delta.content, end="", flush=True)
```

## Event Structure

Each event is an `AgentStreamItem` containing:

| Field | Description |
|-------|-------------|
| `delta` | Content delta (text, tool calls, reasoning) |
| `run_id` | Unique run identifier |
| `session_id` | Session identifier |
| `step` | Current step number |
| `type` | Event type |

### Delta Fields

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str \| None` | Text content chunk |
| `reasoning_content` | `str \| None` | Thinking/reasoning content |
| `tool_calls` | `list[dict] \| None` | Tool call information |
| `finish_reason` | `str \| None` | Why this step ended |

## Handle-based Streaming

For more control, use `start()` to get a handle:

```python
handle = agent.start("Write a poem about code")

async for event in handle.stream():
    if event.delta and event.delta.content:
        sys.stdout.write(event.delta.content)
        sys.stdout.flush()

    # Check for tool calls
    if event.delta and event.delta.tool_calls:
        print(f"\n[Using tool: {event.delta.tool_calls}]")

# Or get the final result
result = await handle.wait()
```

## Scheduler Streaming

The Scheduler exposes the same streaming interface:

```python
async with Scheduler() as scheduler:
    async for event in scheduler.stream("Research topic X", agent=agent):
        if event.delta and event.delta.content:
            print(event.delta.content, end="", flush=True)
```

For an existing agent state:

```python
async for event in scheduler.stream(
    "Continue the analysis",
    state_id=existing_state_id,
):
    process(event)
```

## Stream Consumption

The `consume_execution_stream()` helper handles common patterns:

```python
from agiwo.agent.streaming import consume_execution_stream

handle = agent.start("Do something")

# Automatically handles cleanup on early exit
async for event in consume_execution_stream(
    handle,
    cancel_reason="consumer closed",
):
    process(event)
```

## Under the Hood

```
Agent.run_stream()
  └─► Agent.start() → AgentExecutionHandle
       └─► consume_execution_stream()
            └─► handle.stream()
                 └─► AgentRunner → AgentExecutor
                      └─► LLM Model.arun_stream()
                           └─► StreamChunk (provider-specific)
                                └─► AgentStreamItem (normalized)
```

All execution paths — `run()`, `run_stream()`, Scheduler — share this pipeline. The difference is only in how the consumer processes events.
