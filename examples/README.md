# Examples

Runnable examples demonstrating Agiwo's core features.

## Prerequisites

```bash
export OPENAI_API_KEY=sk-...
# Or for other providers:
# export ANTHROPIC_API_KEY=sk-ant-...
# export DEEPSEEK_API_KEY=sk-...
```

## Examples

| Example | Description |
|---------|-------------|
| [01_hello_agent.py](./01_hello_agent.py) | Minimal agent — one question, one answer |
| [02_streaming.py](./02_streaming.py) | Real-time streaming responses |
| [03_custom_tool.py](./03_custom_tool.py) | Build and use a custom tool |
| [04_builtin_tools.py](./04_builtin_tools.py) | Using builtin web and bash tools |
| [05_hooks.py](./05_hooks.py) | Observe agent lifecycle with hooks |
| [06_agent_as_tool.py](./06_agent_as_tool.py) | Compose agents — researcher + writer |
| [07_scheduler.py](./07_scheduler.py) | Long-running agent with scheduler |
| [08_multi_agent.py](./08_multi_agent.py) | Multi-agent fan-out/fan-in pattern |

## Running

```bash
# From the repository root
uv run python examples/01_hello_agent.py
```
