# Streaming

Agiwo is streaming-first. All LLM responses flow through the same streaming pipeline, whether you use `run()`, `run_stream()`, `start()`, or scheduler `route_root_input()`.

## Streaming with `run_stream()`

The simplest way to get real-time output:

```python
async for event in agent.run_stream("Tell me about Python"):
    if event.type == "step_delta" and event.delta.content:
        print(event.delta.content, end="", flush=True)
```

## Event Structure

`AgentStreamItem` is a tagged union of:

| Event Type | Payload |
|------------|---------|
| `run_started` | run/session metadata |
| `step_delta` | incremental assistant deltas via `event.delta` |
| `step_completed` | committed `StepView` via `event.step` |
| `messages_rebuilt` | rebuilt prompt messages after runtime mutation |
| `context_steps_hidden` | step ids that should be removed from public transcript |
| `compaction_applied` | committed compaction summary + transcript range |
| `compaction_failed` | committed compaction failure fact |
| `step_back_applied` | committed step-back rewrite fact |
| `termination_decided` | committed termination decision fact |
| `run_rolled_back` | committed rollback range |
| `run_completed` | final response, metrics, termination reason |
| `run_failed` | final error |

## Handle-based Streaming

For more control, use `start()` to get a handle:

```python
handle = agent.start("Write a poem about code")

async for event in handle.stream():
    if event.type == "step_delta" and event.delta.content:
        sys.stdout.write(event.delta.content)
        sys.stdout.flush()

    # Check for tool calls
    if event.type == "step_delta" and event.delta.tool_calls:
        print(f"\n[Using tool: {event.delta.tool_calls}]")

# Or get the final result
result = await handle.wait()
```

## Scheduler Streaming

The Scheduler reuses the same stream protocol, but the public entrypoint is
`route_root_input()`. Consume `RouteResult.stream`:

```python
async with Scheduler() as scheduler:
    route = await scheduler.route_root_input(
        "Research topic X",
        agent=agent,
        persistent=False,
    )
    assert route.stream is not None

    async for event in route.stream:
        if event.type == "step_delta" and event.delta.content:
            print(event.delta.content, end="", flush=True)
```

For an existing agent state:

```python
route = await scheduler.route_root_input(
    "Continue the analysis",
    agent=agent,
    state_id=existing_state_id,
)
assert route.stream is not None

async for event in route.stream:
    process(event)
```

Only one live stream subscriber is allowed per root `state_id`. If you steer a
currently `RUNNING` root, `RouteResult.stream` is `None` because the existing
subscriber continues consuming that root's stream.

## Stream Consumption

`run_stream()` is the high-level streaming API. If you need more control, call `start()` and consume `handle.stream()` directly:

```python
handle = agent.start("Do something")

try:
    async for event in handle.stream():
        process(event)
finally:
    handle.cancel("consumer closed")
```

## Under the Hood

```
Agent.run_stream()
  └─► Agent.start() → execution handle
       └─► handle.stream()
            └─► session runtime → run loop
                 └─► LLM Model.arun_stream()
                      └─► StreamChunk (provider-specific)
                           └─► AgentStreamItem (normalized)
```

All execution paths — `run()`, `run_stream()`, `start()`, and scheduler `route_root_input()` — share this pipeline. The difference is only in how the consumer processes events.

## `context_steps_hidden`

When the runtime later decides that some committed steps were only temporary
introspection metadata, it emits a dedicated stream event:

```json
{
  "type": "context_steps_hidden",
  "session_id": "sess-1",
  "run_id": "run-1",
  "agent_id": "agent-1",
  "step_ids": ["step-review-call", "step-review-result"],
  "reason": "introspection_metadata"
}
```

Clients should treat this as a reconciliation signal:

- remove or fold any already-rendered transcript entries whose `step_id` is in
  `step_ids`
- keep the rest of the conversation intact

This keeps the live transcript aligned with historical step queries, which
already hide steps that were marked `hidden_from_context`.
