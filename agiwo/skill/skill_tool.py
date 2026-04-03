"""
Skill Tool - Tool for activating Agent Skills.

This module provides a tool implementation that allows agents to activate
skills by loading their complete SKILL.md content into the context.
"""

import time
from typing import Any

from agiwo.skill.exceptions import SkillNotFoundError
from agiwo.skill.loader import SkillLoader
from agiwo.skill.registry import SkillRegistry
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.tool.context import ToolContext
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger


logger = get_logger(__name__)


class SkillTool(BaseTool):
    """
    Tool for activating Agent Skills.

    When called, loads the complete SKILL.md content for the specified skill
    and returns it as tool result content. The content is separated into
    `content` (for LLM) and `content_for_user` (for display).
    """

    def __init__(
        self,
        registry: SkillRegistry,
        loader: SkillLoader,
        allowed_skills: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.registry = registry
        self.loader = loader
        self._allowed_skills = (
            frozenset(allowed_skills) if allowed_skills is not None else None
        )

    name = "skill"
    description = (
        "Activate an Agent Skill by loading its complete instructions. "
        "Use this tool when a user task matches a skill's description or user explicitly requests a skill."
        "After activation, follow the instructions in the skill's SKILL.md file."
    )

    def get_parameters(self) -> dict[str, Any]:
        """Return the JSON schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to activate",
                },
            },
            "required": ["skill_name"],
        }

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        """
        Activate a skill by loading its complete SKILL.md content.

        Args:
            parameters: Tool parameters containing 'skill_name'
            context: Execution context
            abort_signal: Optional abort signal

        Returns:
            ToolResult with skill content in 'content' field and user message
            in 'content_for_user' field
        """
        start_time = time.time()
        skill_name = parameters.get("skill_name")
        error: str | None = None

        if not skill_name:
            error = "Missing required parameter: skill_name"
        elif not isinstance(skill_name, str):
            error = f"Invalid skill_name type: expected string, got {type(skill_name)}"
        elif self._allowed_skills is not None and skill_name not in self._allowed_skills:
            error = f"Skill '{skill_name}' is not allowed by the configured allowlist."

        metadata = self.registry.get_metadata(skill_name) if error is None else None
        if error is None and not metadata:
            error = (
                f"Skill '{skill_name}' not found. Available skills: "
                f"{', '.join(self.registry.list_available())}"
            )

        if error is not None:
            return ToolResult.failed(
                tool_name=self.name,
                error=error,
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                start_time=start_time,
            )

        try:
            skill_content = await self.loader.load_skill(skill_name)

            logger.info(
                "skill_activated",
                skill_name=skill_name,
                duration_ms=(time.time() - start_time) * 1000,
            )

            return ToolResult.success(
                tool_name=self.name,
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                content=skill_content.body,
                content_for_user=f'The skill "{skill_name}" has been activated.',
                output={
                    "skill_name": skill_name,
                    "metadata": metadata.model_dump(),
                },
                start_time=start_time,
            )

        except SkillNotFoundError as e:
            return ToolResult.failed(
                tool_name=self.name,
                error=str(e),
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                start_time=start_time,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(
                "skill_activation_failed",
                skill_name=skill_name,
                error=str(e),
            )
            return ToolResult.failed(
                tool_name=self.name,
                error=f"Failed to activate skill '{skill_name}': {e}",
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                start_time=start_time,
            )
