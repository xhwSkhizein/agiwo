<h1 align="center">Agiwo</h1>

<p align="center">
  <em>Open-source streaming-first Python AI agent framework and control plane</em>
</p>

<p align="center">
  Build, orchestrate, trace, and operate tool-using LLM agents with streaming execution, scheduler-based orchestration, persistence, and observability.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <a href="https://github.com/xhwSkhizein/agiwo/actions/workflows/ci.yml">
    <img src="https://github.com/xhwSkhizein/agiwo/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
</p>

## Public Docs

- Website: `https://docs.agiwo.o-ai.tech`
- Getting started: `https://docs.agiwo.o-ai.tech/docs/getting-started/`
- Comparison: `https://docs.agiwo.o-ai.tech/docs/compare/agiwo-vs-langgraph-openai-agents-autogen/`
- Repository overview: `https://docs.agiwo.o-ai.tech/docs/repo-overview/`

## Repository Structure

Agiwo has three main areas:

- `agiwo/` — the SDK runtime, including agent execution, tools, scheduler orchestration, model abstraction, memory, workspace, and observability
- `console/` — the FastAPI control plane and internal web UI
- `docs/` — design notes, concepts, and repository-native documentation

## What Is Agiwo?

Agiwo has two parts:

- **SDK**: an async, streaming-first Python framework for building LLM agents with tools, hooks, storage, observability, skills, and scheduler-based orchestration.
- **Console**: an optional self-hosted FastAPI + Next.js control plane for managing agent configs, chatting over SSE, inspecting scheduler state, and integrating channels. It is currently best suited for internal deployments, supports Feishu as its only built-in channel integration, and is not yet production-ready.

The project favors explicit runtime wiring over hidden global state. Agent execution, tool execution, scheduler orchestration, and persistence are all separate layers.

## Current Capabilities

- Streaming-first agent execution through one runtime pipeline surfaced as `start()`, `run()`, and `run_stream()`
- Tool calling with builtin tools, custom `BaseTool` implementations, and agent-as-tool composition via `Agent.as_tool()`
- Scheduler orchestration for roots and child agents, including `submit`, `route_root_input`, `stream`, `wait_for`, `steer`, and cancellation flows
- Run and step persistence plus trace collection with memory, SQLite, and MongoDB-backed storage options
- Global skill discovery with per-agent allowlisting through explicit `allowed_skills`
- Optional Console package for control-plane operations, trace inspection, session chat, and Feishu channel integration

## Quick Start

### Install

```bash
# SDK
pip install agiwo
```

For development from source:

```bash
git clone https://github.com/xhwSkhizein/agiwo.git
cd agiwo
uv sync
```

For SDK usage, export provider credentials in your shell or place them in a local `.env`. Set only the credentials for the providers you actually use.

Example:

```bash
export OPENAI_API_KEY=...
```

### Minimal SDK Example

```python
import asyncio

from agiwo.agent import Agent, AgentConfig
from agiwo.llm import OpenAIModel


async def main() -> None:
    agent = Agent(
        AgentConfig(
            name="assistant",
            description="A helpful assistant",
            system_prompt="You are a concise assistant.",
        ),
        model=OpenAIModel(name="gpt-5.4"),
    )

    result = await agent.run("What is 2 + 2?")
    print(result.response)

    async for event in agent.run_stream("Give me a one-line summary of recursion."):
        if event.type == "step_delta" and event.delta.content:
            print(event.delta.content, end="", flush=True)

    await agent.close()


asyncio.run(main())
```

### Custom Tool Example

```python
from agiwo.tool import BaseTool, ToolResult, ToolContext


class WeatherTool(BaseTool):
    name = "get_weather"
    description = "Get the current weather for a city"

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
        }

    async def execute(
        self,
        parameters: dict,
        context: ToolContext,
        abort_signal=None,
    ) -> ToolResult:
        city = parameters["city"]
        return ToolResult.success(
            tool_name=self.name,
            content=f"Weather in {city}: sunny, 25C",
            content_for_user=f"{city}: sunny, 25C",
            output={"city": city, "condition": "sunny", "temp_c": 25},
        )
```

### Agent As Tool

```python
from agiwo.agent import Agent, AgentConfig
from agiwo.llm import DeepseekModel

researcher = Agent(
    AgentConfig(
        name="researcher",
        description="Research specialist",
        system_prompt="You are strong at collecting and summarizing evidence.",
    ),
    model=DeepseekModel(id="deepseek-chat"),
)

orchestrator = Agent(
    AgentConfig(
        name="orchestrator",
        description="Delegates focused research tasks",
        system_prompt="Delegate independent research tasks when useful.",
    ),
    model=DeepseekModel(id="deepseek-chat"),
    tools=[researcher.as_tool()],
)
```

### Scheduler Example

```python
import asyncio

from agiwo.agent import Agent, AgentConfig
from agiwo.scheduler import Scheduler
from agiwo.llm import DeepseekModel


async def main() -> None:
    agent = Agent(
        AgentConfig(
            name="orchestrator",
            description="Can delegate and wait",
            system_prompt="Use spawned agents only for truly independent sub-tasks.",
        ),
        model=DeepseekModel(id="deepseek-chat"),
    )

    async with Scheduler() as scheduler:
        result = await scheduler.run(agent, "Research two competing approaches and summarize them.")
        print(result.response)


asyncio.run(main())
```

For long-running roots, the scheduler API also supports `submit`, `enqueue_input`, `route_root_input`, `stream`, `wait_for`, `steer`, `cancel`, and `shutdown`.

## Console

The Console is a separately published control-plane package and is intentionally positioned below the SDK in scope and readiness.

- Package: `pip install agiwo-console`
- Recommended deployment model: internal/self-hosted use
- Built-in channel integrations today: Feishu only
- Readiness: useful for operators, not yet production-ready

### Start The API Server

The Console environment template lives at [console/.env.example.full](console/.env.example.full).

```bash
pip install agiwo-console
cat > .env <<'EOF'
OPENAI_API_KEY=...
EOF
agiwo-console serve --env-file .env
```

If you are running from source instead of the published package, the full template lives at `console/.env.example.full`.

The API server defaults to `http://localhost:8422`.

Useful routes:

- `GET /api/health`
- `GET /api/agents`
- `POST /api/chat/{agent_id}`
- `GET /api/scheduler/states`
- `GET /api/traces`

### Start The Web UI

```bash
cd console/web
npm install
npm run dev
```

The frontend reads `NEXT_PUBLIC_API_URL` from `console/web/.env.local`.

Example:

```bash
echo 'NEXT_PUBLIC_API_URL=http://localhost:8422' > console/web/.env.local
```

## Configuration Model

There are two configuration layers:

- **SDK config** in `agiwo/config/settings.py`
- **Console config** in `console/server/config.py`

Environment variable namespaces:

- SDK-owned keys: `AGIWO_*`
- Console-owned keys: `AGIWO_CONSOLE_*`
- Provider credentials: canonical external names such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `AWS_REGION`

Important current rules:

- Compatible providers (`openai-compatible`, `anthropic-compatible`) must be configured with explicit `base_url` and `api_key_env_name`
- Builtin tools that create their own models read shared defaults from `AGIWO_TOOL_DEFAULT_MODEL_*`

## Repository Discoverability Notes

Recommended GitHub repository description:

`Open-source Python AI agent framework and control plane for streaming, tool use, orchestration, tracing, and persistence.`

Recommended GitHub topics:

`ai-agents`, `python`, `llm`, `agent-framework`, `multi-agent`, `tool-calling`, `observability`, `agent-orchestration`, `fastapi`
- Agent config includes `allowed_skills: list[str] | None` to filter available skills (global skill discovery is configured via `AGIWO_SKILL_DIRS`)
- Console agent config writes are full replace, not patch merge
- Scheduler state storage is owned by the `Scheduler`; Console `StorageManager` manages run-step, trace, and citation storage only

## Architecture At A Glance

### SDK

- `agiwo/agent/`: canonical agent runtime
  - `models/`: data models (config, run, stream, input, step)
  - `runtime/`: session runtime, run context, state helpers
  - `nested/`: child-agent adapter (`AgentTool`, `as_tool()`)
  - `retrospect/`: tool result retrospect optimization
  - `storage/`: run/step persistence (memory, SQLite, MongoDB)
- `agiwo/llm/`: model abstractions, providers, config policy, factory (`create_model`), token usage estimation
- `agiwo/tool/`: tool abstractions, `ToolContext`, builtin tools, authz domain types, process registry, citation storage
- `agiwo/scheduler/`: orchestration facade, engine, runner, commands, runtime state, tool control, store, runtime tools
- `agiwo/workspace/`: workspace layout, bootstrap, and workspace document loading
- `agiwo/memory/`: shared MEMORY indexing/search plus `WorkspaceMemoryService`
- `agiwo/observability/`: trace/span models and storage backends; `agiwo.agent.trace_writer.AgentTraceCollector` bridges agent runs into traces
- `agiwo/skill/`: skill discovery, loading, registry, `SkillTool`, allowlist handling
- `agiwo/embedding/`: embedding abstractions and factory (local/OpenAI-style)
- `agiwo/config/`: SDK global settings, provider enums, termination config
- `agiwo/utils/`: cross-module runtime tools, abort signals, logging, storage support

### Console

- `console/server/routers/`: HTTP and SSE API boundary
- `console/server/services/`: agent lifecycle (`agent_lifecycle.py`), registry (`agent_registry/`), storage wiring (`storage_wiring.py`), tool catalog, metrics, SSE services
- `console/server/models/`: shared Console data models (views, session, config)
- `console/server/channels/`: channel runtime, session binding, Feishu integration
- `console/server/config.py`: ConsoleConfig (pydantic-settings, env prefix: AGIWO_CONSOLE_)
- `console/web/`: Next.js frontend

## Development Workflow

Install dependencies once:

```bash
uv sync
```

Low-noise lint is the default workflow after code changes:

```bash
uv run python scripts/lint.py changed
```

If the worktree is already dirty, lint only the files you touched:

```bash
uv run python scripts/lint.py files path/to/file.py path/to/other.py
```

Run tests:

```bash
uv run pytest tests/ -v
(cd console && uv run pytest tests/ -v)
```

## Current Status

The project is usable for experimentation and development. Core SDK APIs (Agent, Tool, Scheduler) are stabilizing. The Console remains an internal-use control plane that is still evolving and should not yet be treated as production-ready.

Areas that still change:

- Console channel/session wiring and Feishu integration details
- Scheduler orchestration edge cases and operator-facing controls
- Trace query APIs and visualization

If you are changing the architecture, update both [AGENTS.md](AGENTS.md) and this README so the repo-level guidance stays aligned with the code.
