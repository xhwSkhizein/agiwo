"""
Example 07: Scheduler

Long-running agent with scheduler orchestration — submit, steer, wait.
"""

import asyncio

from agiwo.agent import Agent
from agiwo.agent import AgentConfig
from agiwo.scheduler import Scheduler
from agiwo.llm import OpenAIModel


async def main() -> None:
    agent = Agent(
        AgentConfig(
            name="analyst",
            description="A thorough analyst",
            system_prompt="You analyze topics in depth. Be systematic.",
        ),
        model=OpenAIModel(name="gpt-5.4"),
    )

    async with Scheduler() as scheduler:
        # Simple run — submit and wait
        print("=== Simple Run ===")
        result = await scheduler.run(
            agent,
            "List 3 pros and 3 cons of microservices architecture.",
        )
        print(result.response)

        # Fire-and-forget with later steering
        print("\n=== Submit + Steer ===")
        state_id = await scheduler.submit(
            agent,
            "Analyze the trade-offs of using PostgreSQL vs MongoDB.",
        )
        print(f"Submitted: {state_id}")

        # Steer the running agent
        await asyncio.sleep(1)
        await scheduler.steer(
            state_id,
            "Focus specifically on query performance and scalability.",
        )
        print("Steered!")

        # Wait for completion
        result = await scheduler.wait_for(state_id)
        print(f"\nResult: {result.response}")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
