"""
Seed script — creates test Agent configs for nested agent-as-tool testing.

Creates:
1. "Researcher" — a child agent with calculator + current_time tools
2. "Coordinator" — a main agent that uses Researcher as a tool

Usage:
    cd console
    uv run python seed_agents.py
"""

import asyncio

from server.config import ConsoleConfig
from server.services.agent_registry import AgentConfigRecord, AgentRegistry

RESEARCHER_ID = "researcher"
COORDINATOR_ID = "coordinator"


async def main() -> None:
    config = ConsoleConfig()
    registry = AgentRegistry(config)
    await registry.initialize()

    # 1. Child Agent: Researcher
    researcher = AgentConfigRecord(
        id=RESEARCHER_ID,
        name="Researcher",
        description="A research assistant that can perform calculations and check current time. Delegate research and analysis tasks to this agent.",
        model_provider="deepseek",
        model_name="deepseek-chat",
        system_prompt=(
            "You are a research assistant. You have access to a calculator and current time tool. "
            "When given a task, break it down, use your tools as needed, and provide a thorough answer. "
            "Always show your reasoning process."
        ),
        tools=["calculator", "current_time"],
        options={"max_steps": 5},
        model_params={"max_tokens": 1024},
    )
    await registry.create_agent(researcher)
    print(f"Created child agent: {researcher.name} (id={researcher.id})")

    # 2. Main Agent: Coordinator (uses Researcher as tool)
    coordinator = AgentConfigRecord(
        id=COORDINATOR_ID,
        name="Coordinator",
        description="A coordinator agent that delegates tasks to the Researcher agent.",
        model_provider="deepseek",
        model_name="deepseek-chat",
        system_prompt=(
            "You are a coordinator agent. You have a Researcher tool available. "
            "When the user asks a question that requires research or calculation, "
            "delegate the task to the Researcher tool by calling it with a clear task description. "
            "Summarize the Researcher's findings and provide a final answer to the user."
        ),
        tools=[f"agent:{RESEARCHER_ID}"],
        options={"max_steps": 8},
        model_params={"max_tokens": 1024},
    )
    await registry.create_agent(coordinator)
    print(f"Created main agent: {coordinator.name} (id={coordinator.id})")

    await registry.close()
    print("\nDone! You can now test nested agent execution:")
    print(f"  POST /api/chat/{COORDINATOR_ID}")
    print(f'  {{"message": "What is 42 * 37 and what time is it now?"}}')


if __name__ == "__main__":
    asyncio.run(main())
