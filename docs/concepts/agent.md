# Agent

The `Agent` class is the primary entry point for the Agiwo SDK. It wraps an LLM model, a set of tools, and optional hooks into a single executable unit.

## Creating an Agent

```python
from agiwo import Agent, AgentConfig
from agiwo.llm import OpenAIModel

agent = Agent(
    AgentConfig(
        name="researcher",
        description="A research assistant",
        system_prompt="You are thorough and cite sources.",
    ),
    model=OpenAIModel(id="gpt-4o", name="gpt-4o"),
)
```

### AgentConfig

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Agent identifier (used in logs, traces, workspace paths) |
| `description` | `str` | Human-readable description |
| `system_prompt` | `str` | System prompt injected at the start of every conversation |
| `options` | `AgentOptions` | Runtime options (termination, memory, compaction) |

### AgentOptions

Key options you can configure:

| Option | Default | Description |
|--------|---------|-------------|
| `max_iterations` | `10` | Maximum tool-calling iterations per run |
| `enable_termination_summary` | `False` | Generate a summary when the run ends |
| `relevant_memory_max_token` | `2048` | Token budget for auto-injected memories |

## Execution Methods

### `run()` — One-shot execution

```python
result = await agent.run("What is 2 + 2?")
print(result.response)          # The final text response
print(result.tool_results)      # List of ToolResult objects
print(result.usage)             # Token usage stats
```

### `run_stream()` — Streaming execution

```python
async for event in agent.run_stream("Tell me a story"):
    if event.delta and event.delta.content:
        print(event.delta.content, end="", flush=True)
    if event.delta and event.delta.tool_calls:
        print(f"\n[Tool call: {event.delta.tool_calls}]")
```

### `start()` — Low-level handle

Returns an `AgentExecutionHandle` for fine-grained control:

```python
handle = agent.start("Research topic X")

# Stream events
async for event in handle.stream():
    process(event)

# Or wait for completion
result = await handle.wait()

# Or steer mid-execution
await handle.steer("Focus on the technical details")

# Or cancel
await handle.cancel()
```

## Agent Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Unique agent instance ID |
| `name` | `str` | Config name |
| `description` | `str` | Config description |
| `config` | `AgentConfig` | Deep copy of config |
| `model` | `Model` | The LLM model |
| `tools` | `tuple` | All registered tools |
| `hooks` | `AgentHooks` | Lifecycle hooks |

## Cleanup

Always close the agent when done to release resources:

```python
await agent.close()
```

Or use the handle pattern — closing the agent cancels active executions and flushes storage.

## Agent-as-Tool (Composition)

Agents can be composed as tools for other agents:

```python
from agiwo.agent.runtime_tools import as_tool

researcher = Agent(
    AgentConfig(name="researcher", description="Research specialist", system_prompt="..."),
    model=OpenAIModel(id="gpt-4o", name="gpt-4o"),
)

orchestrator = Agent(
    AgentConfig(name="orchestrator", description="Delegates research", system_prompt="..."),
    model=OpenAIModel(id="gpt-4o", name="gpt-4o"),
    tools=[as_tool(researcher)],
)
```

See [Multi-Agent Guide](../guides/multi-agent.md) for details.

## Internal Architecture

The Agent class is a thin facade over several internal components:

```
Agent (facade)
├── AgentDefinitionRuntime   — tools, hooks, prompt, skills (lives with Agent instance)
├── AgentResourceOwner       — storage, active executions (resource lifecycle)
├── AgentRunner              — execution loop (stateless)
└── AgentExecutionHandle     — per-run control surface
```

- `inner/` directory contains the execution pipeline internals — do not import from outside the `agiwo.agent` package.
- Definition-scoped objects (tools, hooks) are separated from resource-scoped objects (storage, active runs).
