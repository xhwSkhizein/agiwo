"""
Example 08: Multi-Agent Fan-Out/Fan-In

Spawn multiple agents in parallel, collect results, synthesize.
"""

import asyncio

from agiwo.agent import Agent
from agiwo.agent import AgentConfig
from agiwo.scheduler import Scheduler
from agiwo.llm import OpenAIModel


async def main() -> None:
    # Researcher agent — investigates a single topic
    researcher = Agent(
        AgentConfig(
            name="researcher",
            description="Researches a specific topic",
            system_prompt="Research the given topic and provide 3 key findings.",
        ),
        model=OpenAIModel(name="gpt-5.4"),
    )

    # Synthesizer agent — combines multiple findings
    synthesizer = Agent(
        AgentConfig(
            name="synthesizer",
            description="Combines research into a summary",
            system_prompt="Given multiple research findings, write a cohesive summary.",
        ),
        model=OpenAIModel(name="gpt-5.4"),
    )

    async with Scheduler() as scheduler:
        # Fan out: research multiple topics in parallel
        topics = [
            "Python performance optimization",
            "Rust memory safety",
            "Go concurrency model",
        ]

        print("=== Fan Out: Researching topics ===")
        state_ids = []
        for topic in topics:
            state_id = await scheduler.submit(researcher, f"Research: {topic}")
            state_ids.append(state_id)
            print(f"  Submitted: {topic} → {state_id}")

        # Fan in: collect all results
        print("\n=== Fan In: Collecting results ===")
        findings = []
        for i, state_id in enumerate(state_ids):
            result = await scheduler.wait_for(state_id)
            findings.append(f"## {topics[i]}\n{result.response}")
            print(f"  Collected: {topics[i]}")

        # Synthesize
        print("\n=== Synthesizing ===")
        combined = "\n\n".join(findings)
        final = await scheduler.run(
            synthesizer,
            f"Compare and contrast these programming language approaches:\n\n{combined}",
        )
        print(f"\n{final.response}")

    await researcher.close()
    await synthesizer.close()


if __name__ == "__main__":
    asyncio.run(main())
