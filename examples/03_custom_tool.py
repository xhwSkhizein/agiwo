"""
Example 03: Custom Tool

Build a tool that the agent can call during reasoning.
"""

import asyncio

from agiwo.agent import Agent
from agiwo.agent import AgentConfig
from agiwo.llm import OpenAIModel
from agiwo.tool import BaseTool, ToolContext, ToolResult


class WeatherTool(BaseTool):
    """Simulated weather lookup tool."""

    name = "get_weather"
    description = "Get the current weather for a city"

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name (e.g., Beijing, London)",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                    "default": "celsius",
                },
            },
            "required": ["city"],
        }

    async def execute(
        self,
        parameters: dict,
        context: ToolContext,
        abort_signal=None,
    ) -> ToolResult:
        city = parameters["city"]
        unit = parameters.get("unit", "celsius")

        # Simulated data — replace with real API call
        temp = 22 if unit == "celsius" else 72
        unit_symbol = "°C" if unit == "celsius" else "°F"

        return ToolResult.success(
            tool_name=self.name,
            content=f"Weather in {city}: sunny, {temp}{unit_symbol}",
            content_for_user=f"{city}: sunny, {temp}{unit_symbol}",
            output={"city": city, "condition": "sunny", "temp": temp, "unit": unit},
        )


async def main() -> None:
    agent = Agent(
        AgentConfig(
            name="weather_assistant",
            description="Can check weather",
            system_prompt="Use the get_weather tool when asked about weather.",
        ),
        model=OpenAIModel(name="gpt-5.4"),
        tools=[WeatherTool()],
    )

    result = await agent.run("What's the weather like in Beijing?")
    print(f"Answer: {result.response}")

    # Inspect tool results
    for tool_result in result.tool_results:
        print(f"Tool '{tool_result.tool_name}' returned: {tool_result.content}")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
