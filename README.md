<h1 align="center">Agiwo</h1>

<p align="center">
  <em>A lightweight, streaming-first AI Agent SDK for Python</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/status-experimental-orange" alt="Status">
</p>

> **Warning**
> Agiwo is a **personal toy project** for exploring AI Agent architecture patterns. It is NOT production-ready. APIs may change without notice. Use at your own risk.

## What is Agiwo?

Agiwo is an async-first Python SDK for building LLM-powered agents with tool calling, streaming output, observability, and nested agent composition. It prioritizes simplicity and explicitness over magic.

## Highlights

- **Streaming-first** -- Both `run()` and `run_stream()` use the same streaming pipeline internally. Real-time token-by-token output is a first-class citizen, not an afterthought.
- **Agent as Tools** -- No complex multi-agent orchestration framework. Simply wrap any Agent as a Tool with `as_tool()`, and let the parent Agent call it naturally. Depth limits and circular reference detection are built in.
- **Lifecycle Hooks** -- Inject custom behavior at 10 execution points (before/after run, LLM call, tool call, etc.) without subclassing. Just pass plain async functions.
- **Pluggable Storage** -- Run/Step history and Trace data are persisted through abstract interfaces. Ships with InMemory, SQLite, and MongoDB implementations. Configure via `AgentOptions`, no global state.
- **Built-in Observability** -- `TraceCollector` automatically builds OpenTelemetry-compatible Traces and Spans from the event stream as middleware. Optional OTLP export.
- **Multi-Provider LLM Support** -- OpenAI, Anthropic, DeepSeek, NVIDIA out of the box. Adding an OpenAI-compatible provider is ~20 lines of code.
- **Scheduler** -- Orchestration layer for long-running, sleep/wake agent workflows. Spawn child agents, sleep until conditions are met, and resume automatically. Blocking and non-blocking APIs.
- **Skill System** -- Optional file-based skill discovery. Agents can activate domain-specific skills at runtime via SKILL.md definitions.

## Quick Start

### Installation

```bash
# Clone and install with uv
git clone https://github.com/xhwSkhizein/agiwo.git
cd agiwo
uv sync

# Copy env file and add your API keys
cp .env.example .env
```

### Minimal Example

```python
import asyncio
from agiwo import Agent
from agiwo.llm import DeepseekModel

async def main():
    agent = Agent(
        id="my-agent",
        description="A helpful assistant",
        model=DeepseekModel(id="deepseek-chat", name="deepseek-chat"),
        system_prompt="You are a helpful assistant.",
    )

    # Blocking execution
    result = await agent.run("What is 2 + 2?")
    print(result.response)

    # Streaming execution
    async for event in agent.run_stream("Tell me a joke"):
        if event.delta and event.delta.content:
            print(event.delta.content, end="", flush=True)

    await agent.close()

asyncio.run(main())
```

## Core API

### Agent

The single entry point. No subclassing needed.

```python
from agiwo import Agent, AgentOptions, AgentHooks
from agiwo.llm import OpenAIModel

agent = Agent(
    id="assistant",
    description="A helpful assistant",
    model=OpenAIModel(id="gpt-4o-mini", name="gpt-4o-mini"),
    tools=[MyTool()],               # list[BaseTool], optional
    system_prompt="You are helpful.",
    options=AgentOptions(            # execution config, optional
        max_steps=10,
        run_timeout=600,
    ),
    hooks=AgentHooks(                # lifecycle hooks, optional
        on_before_run=my_hook,
        on_after_tool_call=my_callback,
    ),
)
```

### Custom Tools

Implement `BaseTool` to create tools the agent can call:

```python
from agiwo import BaseTool, ToolResult
from agiwo.agent import ExecutionContext

class WeatherTool(BaseTool):
    def get_name(self) -> str:
        return "get_weather"

    def get_description(self) -> str:
        return "Get current weather for a city"

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        }

    def is_concurrency_safe(self) -> bool:
        return True

    async def execute(self, parameters, context, abort_signal=None) -> ToolResult:
        city = parameters["city"]
        # ... fetch weather data ...
        return ToolResult(
            tool_name=self.name,
            tool_call_id=parameters.get("tool_call_id", ""),
            input_args=parameters,
            content=f"Weather in {city}: 25C, sunny",       # for LLM
            content_for_user=f"It's 25C and sunny in {city}", # for UI
            output={"temp": 25, "condition": "sunny"},
            start_time=0, end_time=0, duration=0,
        )
```

### Agent as Tool (Nested Agents)

```python
from agiwo import Agent, as_tool
from agiwo.llm import DeepseekModel

researcher = Agent(
    id="researcher",
    description="Expert at research tasks",
    model=DeepseekModel(id="deepseek-chat", name="deepseek-chat"),
    system_prompt="You are a research expert.",
)

orchestrator = Agent(
    id="orchestrator",
    description="Main agent that delegates tasks",
    model=DeepseekModel(id="deepseek-chat", name="deepseek-chat"),
    tools=[as_tool(researcher)],  # researcher becomes a callable tool
    system_prompt="Delegate research tasks to the researcher tool.",
)

result = await orchestrator.run("Research quantum computing trends")
```

### Scheduler (Agent Orchestration)

The `Scheduler` sits above `Agent` and manages spawn, sleep, wake, and completion lifecycles. Agents have **zero knowledge** of the Scheduler -- scheduling tools are injected externally.

```python
import asyncio
from agiwo import Agent, Scheduler, SchedulerConfig
from agiwo.llm import DeepseekModel

async def main():
    model = DeepseekModel(id="deepseek-chat", name="deepseek-chat")
    agent = Agent(
        id="orchestrator",
        description="An orchestrator that can spawn sub-agents",
        model=model,
        system_prompt="You can spawn child agents and sleep until they finish.",
    )

    # Blocking mode (simplest)
    async with Scheduler() as scheduler:
        result = await scheduler.run(agent, "Research and summarize AI trends")
        print(result.response)

asyncio.run(main())
```

With persistent state storage and task limits:

```python
from agiwo import Scheduler, SchedulerConfig
from agiwo.scheduler import AgentStateStorageConfig, TaskLimits

config = SchedulerConfig(
    state_storage=AgentStateStorageConfig(
        storage_type="sqlite",
        config={"db_path": "scheduler.db"},
    ),
    check_interval=5.0,
    max_concurrent=10,
    task_limits=TaskLimits(
        max_depth=5,
        max_children_per_agent=10,
        default_wait_timeout=600,
        max_wake_count=20,
    ),
)
async with Scheduler(config) as scheduler:
    # Non-blocking mode
    state_id = await scheduler.submit(agent, "Long running task")
    result = await scheduler.wait_for(state_id, timeout=300)
```

Persistent root agents stay alive after completing a task and accept new tasks:

```python
async with Scheduler(config) as scheduler:
    # Root agent persists after task completion
    state_id = await scheduler.submit(agent, "Initial task", persistent=True)
    result = await scheduler.wait_for(state_id)

    # Submit a new task to the same persistent agent
    await scheduler.submit_task(state_id, "Follow-up task")
    result2 = await scheduler.wait_for(state_id)

    # Graceful shutdown — agent produces a final summary
    await scheduler.shutdown(state_id)
```

The Scheduler automatically injects three tools into managed agents:

- **`spawn_agent`** -- Create child agents for sub-tasks (with depth and children limits)
- **`sleep_and_wait`** -- Sleep until a waitset of children completes, a timer fires, or periodically
- **`query_spawned_agent`** -- Check the status/result of spawned agents

### Persistent Storage

```python
from agiwo import Agent, AgentOptions
from agiwo.agent import RunStepStorageConfig, TraceStorageConfig

agent = Agent(
    id="persistent-agent",
    description="Agent with SQLite storage",
    model=my_model,
    options=AgentOptions(
        run_step_storage=RunStepStorageConfig(
            storage_type="sqlite",
            config={"db_path": "~/.agiwo/data.db"},
        ),
        trace_storage=TraceStorageConfig(
            storage_type="sqlite",
            config={"db_path": "~/.agiwo/data.db"},
        ),
    ),
)
```

### LLM Providers

```python
from agiwo.llm import OpenAIModel, AnthropicModel, DeepseekModel, NvidiaModel

# OpenAI
model = OpenAIModel(id="gpt-4o-mini", name="gpt-4o-mini")

# Anthropic (independent implementation, not OpenAI-compatible)
model = AnthropicModel(id="claude-3-5-sonnet-20240620", name="claude-3-5-sonnet")

# DeepSeek (extends OpenAIModel, with thinking mode support)
model = DeepseekModel(id="deepseek-chat", name="deepseek-chat")

# NVIDIA (extends OpenAIModel)
model = NvidiaModel(id="moonshotai/kimi-k2.5", name="kimi-k2.5")
```

API keys are resolved in order: constructor argument > `AgiwoSettings` > environment variable.

## Architecture

```
User Input
    |
    v
  Agent.run() / run_stream()
    |
    v
  ExecutionContext + StreamChannel
    |
    v
  AgentExecutor._run_loop()
    |
    +---> LLMStreamHandler.stream_assistant_step()
    |         |
    |         v
    |     Model.arun_stream()  -->  StreamChunk
    |
    +---> ToolExecutor.execute_batch()  (parallel)
    |         |
    |         v
    |     BaseTool.execute()  -->  ToolResult
    |
    +---> Loop until: completed / max_steps / timeout / sleeping
    |
    v
  RunOutput (response + metrics + termination_reason)
    |
    v
  [Optional] TraceCollector wraps event stream --> Trace + Spans
```

```
Scheduler (orchestration layer, sits above Agent)
    |
    +---> submit(agent, input, persistent?) --> AgentState (RUNNING)
    +---> submit_task(state_id, task) --> wake persistent agent
    +---> _scheduling_loop:
    |         |
    |         +---> _propagate_signals: child COMPLETED/FAILED --> parent.completed_ids
    |         +---> _enforce_timeouts: timed-out SLEEPING --> wake for summary
    |         +---> _start_pending: PENDING --> run Agent
    |         +---> _wake_sleeping: satisfied WakeCondition --> resume with results
    |
    +---> TaskGuard: centralized limit checks (depth, children, wake count, timeout)
    +---> Injected tools: spawn_agent / sleep_and_wait / query_spawned_agent
    +---> cancel(state_id) / shutdown(state_id): recursive tree operations
```

Key principle: **Agent** owns the lifecycle, **AgentExecutor** owns the loop, **Model** and **Tools** are stateless collaborators. **Scheduler** orchestrates multi-agent sleep/wake workflows from the outside. **TaskGuard** centralizes all limit enforcement.

## Known Limitations

This project is under active exploration. Known gaps include:

- **No production hardening** -- Error recovery, rate limiting, and circuit breakers are minimal
- **Limited test coverage** -- Unit tests exist but integration test coverage needs improvement
- **Tool cache incomplete** -- `ToolExecutor` cache is declared but not fully wired (see FIXME in code)
- **No structured output** -- No built-in JSON mode or schema-validated output from LLM
- **Single-consumer streaming** -- `StreamChannel.read()` can only be claimed once per execution
- **No conversation branching** -- Session history is append-only, no fork/rewind support yet
- **Documentation sparse** -- Inline docstrings exist but no standalone user guide

Contributions, ideas, and feedback are welcome.

## Project Structure

```
agiwo/
  agent/          # Agent core (Agent, ExecutionContext, Hooks, Options, Schema)
  llm/            # LLM providers (OpenAI, Anthropic, DeepSeek, NVIDIA)
  tool/           # Tool system (BaseTool, AgentTool, ToolExecutor, builtins)
  scheduler/      # Agent orchestration (Scheduler, AgentState, sleep/wake tools)
  observability/  # Tracing (TraceCollector, OTLP exporter, storage)
  skill/          # Optional skill system (SKILL.md discovery, hot-reload)
  config/         # Settings (pydantic-settings)
  utils/          # Shared utilities (logging, retry, abort signal)
tests/            # Unit tests (mock-based)
console/          # Control plane UI (FastAPI + Next.js)
pyproject.toml    # Project metadata and dependencies
```

## License

MIT