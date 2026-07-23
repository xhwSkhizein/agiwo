"""
Example 08: Multi-Agent Fan-Out/Fan-In

Spawn child agents through scheduler runtime tools, collect results, synthesize.
Session user entry is MainAgent.accept; this sample uses Agent.run for one-shots
and Scheduler only for child delegation.
"""

import asyncio

from agiwo.agent import Agent
from agiwo.agent import AgentConfig
from agiwo.scheduler import Scheduler
from agiwo.llm import OpenAIModel


async def main() -> None:
    researcher = Agent(
        AgentConfig(
            name="researcher",
            description="Researches a specific topic",
            system_prompt="Research the given topic and provide 3 key findings.",
        ),
        model=OpenAIModel(name="gpt-5.4"),
        id="researcher-root",
    )

    synthesizer = Agent(
        AgentConfig(
            name="synthesizer",
            description="Combines research into a summary",
            system_prompt="Given multiple research findings, write a cohesive summary.",
        ),
        model=OpenAIModel(name="gpt-5.4"),
    )

    topics = [
        "Python performance optimization",
        "Rust memory safety",
        "Go concurrency model",
    ]

    print("=== Fan Out: parallel Agent.run one-shots ===")
    findings: list[str] = []
    for topic in topics:
        result = await researcher.run(f"Research: {topic}")
        findings.append(f"## {topic}\n{result.response}")
        print(f"  Collected: {topic}")

    print("\n=== Fan In: synthesize with a fresh Agent.run ===")
    combined = "\n\n".join(findings)
    summary = await synthesizer.run(
        "Compare and contrast these programming language approaches:\n\n" + combined
    )
    print(f"\n{summary.response}")

    print("\n=== Scheduler child delegation (optional pattern) ===")
    async with Scheduler() as scheduler:
        await scheduler.register_worker_parent(
            state_id=researcher.id,
            session_id="example-08",
            agent=researcher,
        )
        await scheduler.enqueue_input(
            researcher.id,
            "Briefly note when spawn_child_agent is appropriate.",
            agent=researcher,
        )
        delegated = await scheduler.wait_for(researcher.id)
        print(delegated.response)

    await researcher.close()
    await synthesizer.close()


if __name__ == "__main__":
    asyncio.run(main())
