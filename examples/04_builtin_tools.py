"""
Example 04: Builtin Tools

Using Agiwo's built-in tools — web_search, bash, etc.
"""

import asyncio

from agiwo.agent import Agent
from agiwo.agent import AgentConfig
from agiwo.llm import OpenAIModel


async def main() -> None:
    # Builtin tools (bash, web_search, web_reader, memory_retrieval)
    # are automatically included — no need to register them.

    agent = Agent(
        AgentConfig(
            name="researcher",
            description="Can search the web and run commands",
            system_prompt=(
                "You are a researcher. Use web_search to find information, "
                "and bash when you need to check system state."
            ),
        ),
        model=OpenAIModel(name="gpt-5.4"),
        # No tools=[] needed — builtins are auto-included
    )

    result = await agent.run(
        "What is the latest stable version of Python? Just tell me the version number."
    )
    print(f"Answer: {result.response}")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
