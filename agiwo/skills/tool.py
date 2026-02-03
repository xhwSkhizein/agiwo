"""
Skill Tool - Tool for activating Agent Skills.

This module provides a tool implementation that allows agents to activate
skills by loading their complete SKILL.md content into the context.
"""

import time
from typing import Any

from agiwo.tool.base import BaseTool, ToolResult
from agiwo.skills.exceptions import SkillNotFoundError
from agiwo.skills.loader import SkillLoader
from agiwo.skills.registry import SkillRegistry
from agiwo.utils.abort_signal import AbortSignal
from agiwo.agent.execution_context import ExecutionContext
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
    ) -> None:
        """
        Initialize Skill tool.

        Args:
            registry: SkillRegistry instance for metadata lookup
            loader: SkillLoader instance for loading skill content
        """
        super().__init__()
        self.registry = registry
        self.loader = loader

    def get_name(self) -> str:
        """Return the tool name."""
        return "Skill"

    def get_description(self) -> str:
        """Return the tool description."""
        return (
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

    def is_concurrency_safe(self) -> bool:
        """Skill tool is concurrency safe."""
        return True

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
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

        if not skill_name:
            return self._create_error_result(
                parameters,
                "Missing required parameter: skill_name",
                start_time,
            )

        if not isinstance(skill_name, str):
            return self._create_error_result(
                parameters,
                f"Invalid skill_name type: expected string, got {type(skill_name)}",
                start_time,
            )

        # Check if skill exists
        metadata = self.registry.get_metadata(skill_name)
        if not metadata:
            return self._create_error_result(
                parameters,
                f"Skill '{skill_name}' not found. Available skills: {', '.join(self.registry.list_available())}",
                start_time,
            )

        try:
            # Load complete skill content
            skill_content = await self.loader.load_skill(skill_name)

            # Resolve {baseDir} variable (already done in load_skill, but ensure)
            resolved_body = self.loader.resolve_base_dir(skill_name, skill_content.body)

            end_time = time.time()

            logger.info(
                "skill_activated",
                skill_name=skill_name,
                duration_ms=(end_time - start_time) * 1000,
            )

            return ToolResult(
                tool_name=self.name,
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                content=resolved_body,  # Full skill content for LLM
                content_for_user=f'The skill "{skill_name}" has been activated.',
                output={
                    "skill_name": skill_name,
                    "metadata": metadata.model_dump(),
                },
                is_success=True,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
            )

        except SkillNotFoundError as e:
            return self._create_error_result(parameters, str(e), start_time)
        except Exception as e:
            logger.error(
                "skill_activation_failed",
                skill_name=skill_name,
                error=str(e),
            )
            return self._create_error_result(
                parameters,
                f"Failed to activate skill '{skill_name}': {e}",
                start_time,
            )

    def _create_error_result(
        self,
        parameters: dict[str, Any],
        error: str,
        start_time: float,
    ) -> ToolResult:
        """
        Create an error ToolResult.

        Args:
            parameters: Tool parameters
            error: Error message
            start_time: Start time for duration calculation

        Returns:
            ToolResult with error information
        """
        end_time = time.time()
        return ToolResult(
            tool_name=self.name,
            tool_call_id=parameters.get("tool_call_id", ""),
            input_args=parameters,
            content=f"Error: {error}",
            content_for_user=f"Failed to activate skill: {error}",
            output=None,
            error=error,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            is_success=False,
        )
