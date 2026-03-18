"""
Example 05: Hooks

Observe and intercept agent lifecycle events.
"""

import asyncio
import time

from agiwo import Agent, AgentConfig, AgentHooks
from agiwo.llm import OpenAIModel


async def main() -> None:
    # Define hooks
    hooks = AgentHooks(
        on_run_start=_on_run_start,
        on_run_end=_on_run_end,
        on_tool_start=_on_tool_start,
        on_tool_end=_on_tool_end,
    )

    agent = Agent(
        AgentConfig(
            name="monitored_agent",
            description="An agent with hooks",
            system_prompt="You are a helpful assistant.",
        ),
        model=OpenAIModel(id="gpt-4o-mini", name="gpt-4o-mini"),
        hooks=hooks,
    )

    result = await agent.run("What is 2 + 2? Use the bash tool to verify.")
    print(f"\nFinal answer: {result.response}")

    await agent.close()


async def _on_run_start(context) -> None:
    print(f"[HOOK] Run started: {context.run_id}")


async def _on_run_end(context, result) -> None:
    print(f"[HOOK] Run ended: {context.run_id}")
    print(f"[HOOK] Response: {result.response[:100]}...")


async def _on_tool_start(context, tool_name, parameters) -> None:
    print(f"[HOOK] Tool starting: {tool_name}")
    print(f"[HOOK] Parameters: {parameters}")


async def _on_tool_end(context, result) -> None:
    print(f"[HOOK] Tool finished: {result.tool_name} ({result.duration:.2f}s)")
    print(f"[HOOK] Success: {result.is_success}")


if __name__ == "__main__":
    asyncio.run(main())
