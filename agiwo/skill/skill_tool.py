"""
Skill Tool - Tool for activating Agent Skills.

This module provides a tool implementation that allows agents to activate
skills by loading their complete SKILL.md content into the context.
"""

import json
import time
from typing import Any

from agiwo.config.settings import get_settings
from agiwo.skill.exceptions import SkillNotFoundError
from agiwo.skill.loader import SkillLoader
from agiwo.skill.registry import SkillMetadata, SkillRegistry
from agiwo.skill.search import SkillSearchRecommendation, SkillSearchService
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
        search_service: SkillSearchService | None = None,
    ) -> None:
        super().__init__()
        self.registry = registry
        self.loader = loader
        self._search_service = search_service
        self._allowed_skills = (
            frozenset(allowed_skills) if allowed_skills is not None else None
        )

    name = "skill"
    description = (
        "Search for a matching skill or activate a specific skill. "
        "Use mode=search with the user's original request when you are unsure whether a skill is needed. "
        "Use mode=activate only after you have chosen a specific skill."
    )

    def get_parameters(self) -> dict[str, Any]:
        """Return the JSON schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["search", "activate"],
                    "description": "Search for a recommended skill or activate one by name",
                },
                "query": {
                    "type": "string",
                    "description": "Original user request when mode=search",
                },
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to activate when mode=activate",
                },
            },
            "required": ["mode"],
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
        mode = parameters.get("mode")
        if mode == "search":
            return await self._execute_search(parameters, context, start_time)
        if mode == "activate":
            return await self._execute_activate(parameters, context, start_time)
        return ToolResult.failed(
            tool_name=self.name,
            error=f"Invalid mode: {mode}",
            tool_call_id=context.tool_call_id,
            input_args=parameters,
            start_time=start_time,
        )

    async def _execute_search(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        start_time: float,
    ) -> ToolResult:
        query = parameters.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult.failed(
                tool_name=self.name,
                error="Missing required parameter: query",
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                start_time=start_time,
            )
        if not get_settings().skill_search_enabled:
            recommendation = SkillSearchRecommendation(decision="no_recommendation")
        else:
            recommendation = (
                await self._search_service.search(
                    query=query,
                    metadata_items=self._searchable_metadata(),
                )
                if self._search_service is not None
                else SkillSearchRecommendation(decision="no_recommendation")
            )

        payload = {
            "decision": recommendation.decision,
            "skill_name": recommendation.skill_name,
            "reason": recommendation.reason,
        }

        return ToolResult.success(
            tool_name=self.name,
            tool_call_id=context.tool_call_id,
            input_args=parameters,
            content=json.dumps(payload),
            content_for_user="Skill search completed.",
            output=payload,
            start_time=start_time,
        )

    async def _execute_activate(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        start_time: float,
    ) -> ToolResult:
        skill_name = parameters.get("skill_name")
        error: str | None = None

        if not skill_name:
            error = "Missing required parameter: skill_name"
        elif not isinstance(skill_name, str):
            error = f"Invalid skill_name type: expected string, got {type(skill_name)}"
        elif (
            self._allowed_skills is not None and skill_name not in self._allowed_skills
        ):
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

    def _searchable_metadata(self) -> list[SkillMetadata]:
        metadata_items = [
            self.registry.get_metadata(name)
            for name in self.registry.list_available()
        ]
        filtered = [item for item in metadata_items if item is not None]
        if self._allowed_skills is None:
            return filtered
        return [item for item in filtered if item.name in self._allowed_skills]
