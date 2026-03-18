# Scheduler

The `Scheduler` is an orchestration layer built on top of agents. It provides multi-agent coordination, long-running task management, sleep/wake cycles, and steering capabilities.

## When to Use the Scheduler

Use the Scheduler when you need:

- **Long-running agents** that persist across multiple interactions
- **Multi-agent orchestration** with spawn/cancel/steer
- **Periodic or scheduled tasks** with sleep/wake
- **Persistent agent state** that survives individual runs

For simple one-shot or streaming interactions, use `Agent.run()` / `Agent.run_stream()` directly.

## Quick Start

```python
import asyncio
from agiwo import Agent, AgentConfig, Scheduler
from agiwo.llm import OpenAIModel


async def main() -> None:
    agent = Agent(
        AgentConfig(
            name="orchestrator",
            description="Can delegate and coordinate",
            system_prompt="Break complex tasks into sub-tasks.",
        ),
        model=OpenAIModel(id="gpt-4o", name="gpt-4o"),
    )

    async with Scheduler() as scheduler:
        result = await scheduler.run(
            agent,
            "Research two approaches to X and compare them.",
        )
        print(result.response)


asyncio.run(main())
```

## Core Methods

### `run()` — Run and wait

Submits an agent and waits for completion:

```python
result = await scheduler.run(agent, "Do something complex")
```

| Parameter | Description |
|-----------|-------------|
| `agent` | The Agent instance |
| `user_input` | Input text or `UserInput` |
| `session_id` | Optional session ID for continuity |
| `timeout` | Optional timeout in seconds |
| `persistent` | If `True`, agent stays alive after completion |
| `abort_signal` | Cancellation signal |

### `submit()` — Fire and forget

Submits an agent and returns a state ID immediately:

```python
state_id = await scheduler.submit(agent, "Background task")
# Do other work...
result = await scheduler.wait_for(state_id)
```

### `stream()` — Stream events

```python
async for event in scheduler.stream("Do something", agent=agent):
    if event.delta and event.delta.content:
        print(event.delta.content, end="", flush=True)
```

### `enqueue_input()` — Feed more input

Add input to an already-running or idle agent:

```python
await scheduler.enqueue_input(state_id, "Now focus on the details")
```

### `steer()` — Guide a running agent

```python
await scheduler.steer(state_id, "Change direction: focus on cost instead")
```

### `cancel()` — Stop an agent

```python
await scheduler.cancel(state_id, reason="No longer needed")
```

### `shutdown()` — Terminate and clean up

```python
await scheduler.shutdown(state_id)
```

### `get_state()` — Check agent state

```python
state = await scheduler.get_state(state_id)
print(state.status)  # IDLE, RUNNING, WAITING, QUEUED, COMPLETED, FAILED
```

## Agent States

| State | Description |
|-------|-------------|
| `IDLE` | Agent is alive and waiting for input |
| `RUNNING` | Agent is actively executing |
| `WAITING` | Agent is waiting for a child or event |
| `QUEUED` | Agent is queued for execution |
| `COMPLETED` | Agent finished successfully |
| `FAILED` | Agent terminated with an error |

## Scheduling Tools

When an agent runs under the scheduler, it gets access to additional tools for orchestrating sub-tasks:

| Tool | Description |
|------|-------------|
| `spawn_agent` | Spawn a child agent for a sub-task |
| `sleep_and_wait` | Sleep until a condition is met |
| `query_spawned_agent` | Check on a spawned agent's status |
| `cancel_agent` | Cancel a spawned agent |
| `list_agents` | List all spawned agents |

These tools are injected automatically — you don't need to register them.

## SchedulerConfig

| Option | Default | Description |
|--------|---------|-------------|
| `check_interval` | `1.0` | Seconds between scheduler ticks |
| `max_concurrent` | `10` | Max concurrent agent executions |
| `state_storage` | `"memory"` | `"memory"` or `"sqlite"` |
| `task_limits` | `{}` | Task guard limits |
| `graceful_shutdown_wait_seconds` | `30` | Wait time for active tasks on shutdown |

### Persistent State Storage

For production, use SQLite to persist scheduler state across restarts:

```python
scheduler = Scheduler(SchedulerConfig(state_storage="sqlite"))
```

## Architecture

```
Scheduler (facade)
├── SchedulerEngine      — Public orchestration API
├── SchedulerStateOps    — State transitions
├── SchedulerTickOps     — Tick phases (check wake, timers, etc.)
├── SchedulerTreeOps     — Tree cancel/shutdown
├── SchedulerControlOps  — Tool-facing control helpers
├── SchedulerRunner      — Single agent execution
├── SchedulerCoordinator — In-process state (agents, handles, tasks)
└── Store                — Persistence (memory or sqlite)
```

Dependency direction: `Scheduler → Agent`, never the reverse.
