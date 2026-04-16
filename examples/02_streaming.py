"""
Example 02: Streaming

Real-time streaming output as the LLM generates tokens.
"""

import asyncio

from agiwo.agent import Agent
from agiwo.agent import AgentConfig
from agiwo.llm import OpenAIModel


async def main() -> None:
    agent = Agent(
        AgentConfig(
            name="storyteller",
            description="A creative storyteller",
            system_prompt="You are a creative writer. Be vivid and engaging.",
        ),
        model=OpenAIModel(name="gpt-5.4"),
    )

    print("Streaming response:\n")

    async for event in agent.run_stream(
        "Write a two-sentence story about a robot learning to paint."
    ):
        if event.delta and event.delta.content:
            print(event.delta.content, end="", flush=True)

    print("\n\nDone!")
    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
