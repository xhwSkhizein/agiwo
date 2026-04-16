"""
Example 01: Hello Agent

The simplest possible agent — ask a question, get an answer.
"""

import asyncio

from agiwo.agent import Agent
from agiwo.agent import AgentConfig
from agiwo.llm import OpenAIModel


async def main() -> None:
    agent = Agent(
        AgentConfig(
            name="greeter",
            description="A friendly assistant",
            system_prompt="You are concise. Answer in one sentence.",
        ),
        model=OpenAIModel(name="gpt-5.4"),
    )

    result = await agent.run("What is the capital of France?")
    print(f"Answer: {result.response}")
    print(f"Tokens: {result.usage}")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
