"""
Example 07: Scheduler waitset and control

Session chat belongs to MainAgent.accept. This example shows slim Scheduler
orchestration: enqueue persistent root turns and wait for settlement.
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
        id="analyst-root",
    )

    async with Scheduler() as scheduler:
        session_id = "example-07"
        await scheduler.register_worker_parent(
            state_id=agent.id,
            session_id=session_id,
            agent=agent,
        )

        print("=== Persistent root via enqueue_input ===")
        await scheduler.enqueue_input(
            agent.id,
            "List 3 pros and 3 cons of microservices architecture.",
            agent=agent,
        )
        first = await scheduler.wait_for(agent.id)
        print(first.response)

        print("\n=== Second turn on the same persistent root ===")
        await scheduler.enqueue_input(
            agent.id,
            "Now focus on operational complexity only.",
            agent=agent,
        )
        second = await scheduler.wait_for(agent.id)
        print(second.response)

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
