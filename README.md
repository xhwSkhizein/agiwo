# Agiwo

Agiwo is a Python Agent SDK for building LLM-powered agents with tool calling, streaming, observability, and lifecycle hooks.

## Quick Start

```bash
uv sync
cp .env.example .env  # fill in your API keys
```

```python
from agiwo import Agent, AgentOptions, AgentHooks
from agiwo.llm import DeepseekModel
from agiwo.tool import BaseTool, ToolResult

agent = Agent(
    id="my-agent",
    description="A helpful assistant",
    model=DeepseekModel(id="deepseek-chat", name="deepseek-chat"),
    tools=[MyTool()],
    system_prompt="You are a helpful assistant.",
    options=AgentOptions(max_steps=10),
)

# Synchronous execution
result = await agent.run("What is 2+2?")
print(result.response)

# Streaming execution
async for event in agent.run_stream("What is 2+2?"):
    print(event)
```

## Requirements

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv) for dependency management

## Core Concepts

- **Agent** - The primary entry point. Executes LLM loops with tool calling, streaming, and observability.
- **Model** - Abstract LLM provider. Built-in: `OpenAIModel`, `AnthropicModel`, `DeepseekModel`, `NvidiaModel`.
- **BaseTool** - Abstract tool interface. Implement `execute()` to create custom tools.
- **AgentTool** - Wraps an Agent as a Tool for nested agent execution (Agent as Tool pattern).
- **AgentHooks** - Lifecycle hooks for extensibility (see below).
- **AgentOptions** - Execution configuration (max steps, timeout, etc.).
- **RunStepStorage** - Persistence for runs and steps. Built-in: `InMemoryRunStepStorage`, `SQLiteRunStepStorage`, `MongoRunStepStorage`.
- **BaseTraceStorage** - Observability storage with OTLP export support.
- **SkillManager** - Discovers and loads SKILL.md-based agent skills.

## Lifecycle Hooks

All hooks are optional and async. Provide only what you need via `AgentHooks`:

```python
from agiwo import Agent, AgentHooks

async def my_before_tool(tool_name, args):
    print(f"Calling {tool_name} with {args}")
    return None  # return modified args or None to keep original

async def my_memory_write(session_id, content, metadata):
    await my_vector_db.insert(session_id, content, metadata)

async def my_memory_retrieve(session_id, query):
    return await my_vector_db.search(session_id, query)

agent = Agent(
    id="my-agent",
    model=my_model,
    hooks=AgentHooks(
        on_before_tool_call=my_before_tool,
        on_after_tool_call=...,
        on_before_llm_call=...,
        on_after_llm_call=...,
        on_before_run=...,
        on_after_run=...,
        on_memory_write=my_memory_write,
        on_memory_retrieve=my_memory_retrieve,
        on_step=...,
        on_event=...,
    ),
)
```

## Dependency Injection

Stores are injected via constructor, not global config:

```python
from agiwo import Agent
from agiwo.agent.storage import SQLiteRunStepStorage
from agiwo.observability import SQLiteTraceStorage

agent = Agent(
    id="my-agent",
    model=my_model,
    run_step_storage=SQLiteRunStepStorage(db_path="my.db"),
    trace_storage=SQLiteTraceStorage(db_path="my.db"),
)
```

## Project Structure

```
agiwo/
├── agent/           # Core agent: Agent, executor, hooks, schema, options
│   └── storage/     # Run Step stores (in-memory, SQLite, MongoDB)
├── llm/             # LLM providers (OpenAI, Anthropic, DeepSeek, NVIDIA)
├── tool/            # Tool system (BaseTool, AgentTool, ToolExecutor)
│   └── permission/  # Tool permission/consent system
├── skill/           # Skill discovery, loading, and management
├── observability/   # Tracing, OTLP export, trace stores
├── config/          # Global settings (env-based)
└── utils/           # Logging, retry, abort signal
```

## Testing

```bash
# Run all unit tests
uv run pytest tests/ -v

# Run real API integration tests (requires API keys in .env)
uv run python test_real_api.py
uv run python test_real_agent.py
```

## License

MIT
