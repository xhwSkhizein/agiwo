# Getting Started

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

## Installation

### From Source

```bash
git clone https://github.com/xhwSkhizein/agiwo.git
cd agiwo
uv sync
```

### With pip (coming soon)

```bash
pip install agiwo
```

## Configuration

Agiwo reads provider credentials from environment variables. Create a `.env` file in your project root:

```bash
# At least one provider is required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...
```

Or export them in your shell:

```bash
export OPENAI_API_KEY=sk-...
```

### Configuration Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `DEEPSEEK_API_KEY` | DeepSeek API key | — |
| `AGIWO_*` | SDK-level settings (see `agiwo/config/settings.py`) | varies |
| `AGIWO_CONSOLE_*` | Console deployment settings | varies |

## Your First Agent

Create a file `hello.py`:

```python
import asyncio

from agiwo import Agent, AgentConfig
from agiwo.llm import OpenAIModel


async def main() -> None:
    agent = Agent(
        AgentConfig(
            name="assistant",
            description="A helpful assistant",
            system_prompt="You are a concise assistant. Answer in one sentence.",
        ),
        model=OpenAIModel(id="gpt-4o-mini", name="gpt-4o-mini"),
    )

    result = await agent.run("What is the capital of France?")
    print(result.response)

    await agent.close()


asyncio.run(main())
```

Run it:

```bash
uv run python hello.py
# The capital of France is Paris.
```

## Streaming Responses

For real-time output, use `run_stream()`:

```python
async for event in agent.run_stream("Explain recursion in one sentence."):
    if event.delta and event.delta.content:
        print(event.delta.content, end="", flush=True)
```

## Adding Tools

```python
from agiwo import BaseTool, ToolResult
from agiwo.tool import ToolContext


class CalculatorTool(BaseTool):
    def get_name(self) -> str:
        return "calculator"

    def get_description(self) -> str:
        return "Perform basic arithmetic"

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression like '2 + 3'",
                },
            },
            "required": ["expression"],
        }

    def is_concurrency_safe(self) -> bool:
        return True

    async def execute(
        self,
        parameters: dict,
        context: ToolContext,
        abort_signal=None,
    ) -> ToolResult:
        try:
            result = eval(parameters["expression"], {"__builtins__": {}}, {})
            return ToolResult.success(
                tool_name=self.get_name(),
                content=str(result),
            )
        except Exception as e:
            return ToolResult.failed(
                tool_name=self.get_name(),
                error=str(e),
            )


agent = Agent(
    AgentConfig(
        name="math_assistant",
        description="Can do math",
        system_prompt="Use the calculator tool for arithmetic.",
    ),
    model=OpenAIModel(id="gpt-4o-mini", name="gpt-4o-mini"),
    tools=[CalculatorTool()],
)
```

## Builtin Tools

Agiwo ships with several builtin tools that are automatically available:

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands |
| `bash_process` | Manage long-running background processes |
| `web_search` | Search the web |
| `web_reader` | Fetch and extract web page content |
| `memory_retrieval` | Search MEMORY/ files with hybrid retrieval |

You don't need to register these manually — they're included by default.

## Switching Providers

### Anthropic

```python
from agiwo.llm import AnthropicModel

model = AnthropicModel(id="claude-sonnet-4-20250514", name="claude-sonnet-4")
```

### DeepSeek

```python
from agiwo.llm import DeepseekModel

model = DeepseekModel(id="deepseek-chat", name="deepseek-chat")
```

### OpenAI-Compatible (custom endpoint)

```python
from agiwo.llm import OpenAICompatibleModel

model = OpenAICompatibleModel(
    id="my-model",
    name="my-model",
    base_url="https://api.example.com/v1",
    api_key_env_name="MY_API_KEY",
)
```

## Next Steps

- [Core Concepts: Agent](../concepts/agent.md) — Understand the Agent execution model
- [Custom Tools](../guides/custom-tools.md) — Build tools with caching, auth, and more
- [Multi-Agent](../guides/multi-agent.md) — Compose agents with the scheduler
- [Console](../console/overview.md) — Set up the web control plane
