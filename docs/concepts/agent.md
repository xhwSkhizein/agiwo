# Agent

`Agent` is the canonical runtime entry point in Agiwo. It owns one model, one config object, and one assembled tool surface, then exposes three public execution primitives:

- `start(...)` for handle-based execution
- `run(...)` for one-shot request/response
- `run_stream(...)` for streaming consumption

If you only need one agent turn, `run()` is enough. If you need real-time output or custom orchestration, use `run_stream()` or `start()`.

## Construction

```python
from agiwo.agent import Agent, AgentConfig
from agiwo.llm import OpenAIModel

agent = Agent(
    AgentConfig(
        name="assistant",
        description="Helpful SDK assistant",
        system_prompt="Answer concisely and use tools when needed.",
    ),
    model=OpenAIModel(name="gpt-5.4"),
)
```

Public constructor shape:

```python
Agent(
    config: AgentConfig,
    *,
    model: Model,
    tools: list[BaseTool] | None = None,
    hooks: AgentHooks | None = None,
    id: str | None = None,
)
```

Key points:

- `config` is pure configuration only. Live objects stay outside `AgentConfig`.
- `tools` are functional tools and are still filtered by `allowed_tools`.
- system tools are injected by the runtime and are not passed through `tools`.
- `id` should be stable whenever the same logical agent is recreated across requests, otherwise persisted history and scheduler state cannot line up.

## AgentConfig

```python
from agiwo.agent import AgentConfig, AgentOptions

config = AgentConfig(
    name="assistant",
    description="Helpful SDK assistant",
    system_prompt="Use tools only when they materially help.",
    allowed_tools=None,
    allowed_skills=None,
    options=AgentOptions(max_steps=50, run_timeout=600),
)
```

Stable fields:

- `name`, `description`, `system_prompt`
- `allowed_tools`
- `allowed_skills`
- `options`

Permission semantics:

- `allowed_tools=None`: default builtin tools plus all extra functional tools
- `allowed_tools=[]`: no functional tools
- `allowed_skills=None`: all discovered skills remain eligible
- `allowed_skills=[]`: skills are disabled

`allowed_skills` must already be expanded to explicit skill names before entering the runtime.

## Execution Primitives

### `run()`

```python
result = await agent.run("Summarize the main tradeoffs in one paragraph.")
print(result.response)
```

This is the convenience API for one-shot execution. It waits until the run settles and returns `RunOutput`.

### `run_stream()`

```python
async for event in agent.run_stream("Explain recursion in one sentence."):
    if event.type == "step_delta" and event.delta.content:
        print(event.delta.content, end="", flush=True)
```

This yields `AgentStreamItem` values from the same runtime pipeline used by `run()`.

### `start()`

```python
handle = agent.start("Write a three-line release note.")

async for event in handle.stream():
    if event.type == "step_delta" and event.delta.content:
        print(event.delta.content, end="", flush=True)

result = await handle.wait()
```

Use `start()` when you want explicit control over streaming, waiting, or cancellation.

## Builtin And Extra Tools

Agiwo assembles tools through `ToolManager`:

- builtin functional tools such as `bash`, `web_search`, `web_reader`, and `memory_retrieval`
- extra user-supplied tools passed via `tools=[...]`
- `SkillTool` when skills are enabled
- scheduler runtime tools when the agent is executed under `Scheduler`

Scheduler tools such as `spawn_agent` and `sleep_and_wait` are runtime-owned system tools. They are not registered manually on the agent.

## Agent-As-Tool

`Agent.as_tool()` exposes another agent as a functional tool:

```python
researcher_tool = researcher.as_tool()

orchestrator = Agent(
    AgentConfig(
        name="orchestrator",
        system_prompt="Delegate focused research to the researcher tool.",
    ),
    model=OpenAIModel(name="gpt-5.4"),
    tools=[researcher_tool],
)
```

This is the public composition path for nested agents outside scheduler-controlled orchestration.

## Storage And Observability

Persistence is configured under `AgentOptions.storage`, not as top-level `AgentConfig` fields:

```python
from agiwo.agent import (
    AgentConfig,
    AgentOptions,
    AgentStorageOptions,
    RunStepStorageConfig,
    TraceStorageConfig,
)

config = AgentConfig(
    name="assistant",
    options=AgentOptions(
        storage=AgentStorageOptions(
            run_step_storage=RunStepStorageConfig(
                storage_type="sqlite",
                config={"db_path": "runs.db"},
            ),
            trace_storage=TraceStorageConfig(
                storage_type="sqlite",
                config={"db_path": "traces.db"},
            ),
        )
    ),
)
```

## Lifecycle

Agents hold live model clients and storage handles. Call `await agent.close()` when you create agents directly and no longer need them.

When a scheduler owns runtime agents, the scheduler closes its managed instances during cleanup.
