<h1 align="center">Agiwo</h1>

<p align="center">
  <em>Streaming-first AI Agent SDK and Console for Python</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <a href="https://github.com/xhwSkhizein/agiwo/actions/workflows/ci.yml">
    <img src="https://github.com/xhwSkhizein/agiwo/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
</p>

## What Is Agiwo?

Agiwo has two parts:

- **SDK**: an async, streaming-first Python framework for building LLM agents with tools, hooks, storage, observability, skills, and scheduler-based orchestration.
- **Console**: a FastAPI + Next.js control plane for managing agent configs, chatting over SSE, inspecting scheduler state, viewing traces, and integrating channels such as Feishu.

The project favors explicit runtime wiring over hidden global state. Agent execution, tool execution, scheduler orchestration, and persistence are all separate layers.

### Why Agiwo?

| | Agiwo | LangChain | OpenAI Agents SDK |
|---|---|---|---|
| Streaming-first | ✅ All paths share pipeline | ❌ Wraps sync API | ✅ |
| Multi-agent scheduler | ✅ Spawn/steer/cancel/sleep | ❌ (needs LangGraph) | ❌ |
| Agent-as-tool composition | ✅ `as_tool(agent)` | ✅ (but heavy) | ✅ |
| Hook system | ✅ 10 lifecycle hooks | ❌ | ❌ |
| Tool caching | ✅ Session-scoped | ❌ | ❌ |
| Memory retrieval | ✅ BM25 + vector hybrid | ✅ | ❌ |
| Console web UI | ✅ Next.js | ❌ | ❌ |
| Feishu integration | ✅ | ❌ | ❌ |
| Zero global state | ✅ | ❌ | ✅ |
| Token cost tracking | ✅ Per-step | ❌ | ✅ |

## Current Capabilities

- Streaming-first agent execution: `run()` and `run_stream()` share the same execution pipeline.
- Tool calling with session-scoped caching and builtin tools such as `bash`, `bash_process`, `web_search`, `web_reader`, and memory retrieval.
- Agent-as-tool composition through `AgentTool` / `as_tool()`.
- Scheduler orchestration with `submit`, `enqueue_input`, `stream`, `wait_for`, sleep/wake, pending events, and steering.
- Run/step persistence plus trace collection with memory, SQLite, and Mongo-backed implementations where supported.
- Optional file-based skills discovered from skill directories.
- Console APIs for agent config CRUD, chat SSE, trace inspection, scheduler state inspection, and channel runtime integration.

## Quick Start

### Install

```bash
# Core SDK (OpenAI provider only)
pip install agiwo

# With specific providers/features
pip install "agiwo[anthropic]"          # Anthropic provider
pip install "agiwo[web]"               # Web tools (web_reader, Playwright)
pip install "agiwo[feishu]"            # Feishu channel integration
pip install "agiwo[mongo]"             # MongoDB storage backends
pip install "agiwo[all]"               # Everything

# From source (development)
git clone https://github.com/xhwSkhizein/agiwo.git
cd agiwo
uv sync
```

For SDK-only usage, export provider credentials in your shell or place them in a local `.env`.

Example:

```bash
export OPENAI_API_KEY=...
```

### Minimal SDK Example

```python
import asyncio

from agiwo import Agent, AgentConfig
from agiwo.llm import OpenAIModel


async def main() -> None:
    agent = Agent(
        AgentConfig(
            name="assistant",
            description="A helpful assistant",
            system_prompt="You are a concise assistant.",
        ),
        model=OpenAIModel(id="gpt-4o-mini", name="gpt-4o-mini"),
    )

    result = await agent.run("What is 2 + 2?")
    print(result.response)

    async for event in agent.run_stream("Give me a one-line summary of recursion."):
        if event.delta and event.delta.content:
            print(event.delta.content, end="", flush=True)

    await agent.close()


asyncio.run(main())
```

### Custom Tool Example

```python
from agiwo import BaseTool, ToolResult
from agiwo.tool import ToolContext


class WeatherTool(BaseTool):
    def get_name(self) -> str:
        return "get_weather"

    def get_description(self) -> str:
        return "Get the current weather for a city"

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
        }

    def is_concurrency_safe(self) -> bool:
        return True

    async def execute(
        self,
        parameters: dict,
        context: ToolContext,
        abort_signal=None,
    ) -> ToolResult:
        city = parameters["city"]
        del context, abort_signal
        return ToolResult.success(
            tool_name=self.get_name(),
            tool_call_id=parameters.get("tool_call_id", ""),
            input_args=parameters,
            content=f"Weather in {city}: sunny, 25C",
            content_for_user=f"{city}: sunny, 25C",
            output={"city": city, "condition": "sunny", "temp_c": 25},
        )
```

### Agent As Tool

```python
from agiwo import Agent, AgentConfig, as_tool
from agiwo.llm import DeepseekModel

researcher = Agent(
    AgentConfig(
        name="researcher",
        description="Research specialist",
        system_prompt="You are strong at collecting and summarizing evidence.",
    ),
    model=DeepseekModel(id="deepseek-chat", name="deepseek-chat"),
)

orchestrator = Agent(
    AgentConfig(
        name="orchestrator",
        description="Delegates focused research tasks",
        system_prompt="Delegate independent research tasks when useful.",
    ),
    model=DeepseekModel(id="deepseek-chat", name="deepseek-chat"),
    tools=[as_tool(researcher)],
)
```

### Scheduler Example

```python
import asyncio

from agiwo import Agent, AgentConfig, Scheduler
from agiwo.llm import DeepseekModel


async def main() -> None:
    agent = Agent(
        AgentConfig(
            name="orchestrator",
            description="Can delegate and wait",
            system_prompt="Use spawned agents only for truly independent sub-tasks.",
        ),
        model=DeepseekModel(id="deepseek-chat", name="deepseek-chat"),
    )

    async with Scheduler() as scheduler:
        result = await scheduler.run(agent, "Research two competing approaches and summarize them.")
        print(result.response)


asyncio.run(main())
```

For long-running roots, the scheduler API also supports `submit`, `enqueue_input`, `stream`, `wait_for`, `steer`, `cancel`, and `shutdown`.

## Console

### Start The API Server

The Console environment template lives at [console/.env.example](console/.env.example).

```bash
cd console
cp .env.example .env
uv run uvicorn server.app:app --reload --env-file .env
```

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

- compatible providers (`openai-compatible`, `anthropic-compatible`) must be configured with explicit `base_url` and `api_key_env_name`
- builtin tools that create their own models read shared defaults from `AGIWO_TOOL_DEFAULT_MODEL_*`
- Console agent configs store provider/model selection plus model params; agent config writes are full replace, not patch merge
- scheduler state storage is owned by the `Scheduler`, while Console `StorageManager` only manages run-step, trace, and citation storage

## Architecture At A Glance

### SDK

- `agiwo/agent/`: public `Agent` API, runtime types, agent runtime tools, agent trace adapter, prompt runtime, tool auth runtime, run/session storage, and internal executor pipeline in `inner/`
- `agiwo/llm/`: model abstractions, providers, config policy, factory, token usage estimation
- `agiwo/tool/`: tool abstractions, `ToolContext`, builtin tools, authz domain types, process registry, citation storage
- `agiwo/scheduler/`: orchestration layer, runtime/executor, store, services, runtime tools
- `agiwo/workspace/`: workspace layout, bootstrap, and workspace document loading
- `agiwo/memory/`: shared MEMORY indexing/search plus `WorkspaceMemoryService`
- `agiwo/observability/`: trace models and storage backends; agent trace collection lives under `agiwo/agent/trace/`
- `agiwo/skill/`: skill discovery, loading, registry, `SkillTool`

### Console

- `console/server/routers/`: HTTP and SSE API boundary
- `console/server/services/`: agent lifecycle, registry, storage wiring, SSE services
- `console/server/domain/`: shared Console domain models
- `console/server/channels/`: channel runtime, session binding, Feishu integration
- `console/server/tools.py`: canonical tool catalog and tool reference resolution
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

The project is usable for experimentation, but still moving quickly. Areas that still change often:

- module boundaries and internal decomposition during ongoing refactors
- Console domain/channel/session wiring
- scheduler orchestration details and operator-facing controls
- provider/config normalization and builtin tool dependency construction

If you are changing the architecture, update both [AGENTS.md](AGENTS.md) and this README so the repo-level guidance stays aligned with the code.
