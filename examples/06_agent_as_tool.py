"""
Example 06: Agent as Tool

Compose agents — one specialist wrapped as a tool for another.
"""

import asyncio

from agiwo import Agent, AgentConfig
from agiwo.agent.runtime_tools import as_tool
from agiwo.llm import OpenAIModel


async def main() -> None:
    # Specialist: does detailed research
    researcher = Agent(
        AgentConfig(
            name="researcher",
            description="Researches a topic and returns key facts",
            system_prompt=(
                "You are a research specialist. When given a topic, "
                "provide 3-5 key facts with brief explanations. Be factual and concise."
            ),
        ),
        model=OpenAIModel(id="gpt-4o-mini", name="gpt-4o-mini"),
    )

    # Orchestrator: delegates research, then synthesizes
    writer = Agent(
        AgentConfig(
            name="writer",
            description="Writes summaries based on research",
            system_prompt=(
                "You are a writer. When given research material, "
                "synthesize it into a clear, engaging paragraph. "
                "Use the researcher tool to gather facts before writing."
            ),
        ),
        model=OpenAIModel(id="gpt-4o-mini", name="gpt-4o-mini"),
        tools=[as_tool(researcher)],
    )

    result = await writer.run(
        "Write a brief intro about the history of the Python programming language."
    )
    print(result.response)

    # Cleanup both agents
    await writer.close()
    await researcher.close()


if __name__ == "__main__":
    asyncio.run(main())
