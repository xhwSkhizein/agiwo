"""
Skill Manager - Unified entry point for Agent Skills system.

This module provides the main SkillManager class that coordinates skill
discovery, loading, and tool creation. It also generates system prompts
with available skills for agents.
"""

from pathlib import Path

from agiwo.skill.config import SkillDiscoveryConfig, resolve_skill_dirs
from agiwo.skill.loader import SkillLoader
from agiwo.skill.prompt_catalog import (
    SkillPromptCatalog,
    SkillPromptSnapshot,
)
from agiwo.skill.registry import SkillMetadata, SkillRegistry
from agiwo.skill.skill_tool import SkillTool
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class SkillManager:
    """
    Unified manager for Agent Skills system.

    Responsibilities:
    - Coordinate skill discovery and metadata caching
    - Generate system prompts with available skills
    - Provide SkillTool instance for agents
    - Support hot reload for skill updates
    """

    def __init__(
        self,
        config: SkillDiscoveryConfig,
    ) -> None:
        """
        Initialize skill manager.

        Args:
            config: Skill discovery configuration
        """
        self._config = config
        self.registry = SkillRegistry()
        self.loader = SkillLoader(self.registry)
        self._prompt_catalog = SkillPromptCatalog()
        self._skill_tool: SkillTool | None = None
        self._metadata_cache: list[SkillMetadata] = []
        self._change_token = ""
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize skill manager by discovering all available skills.

        This should be called during agent startup to populate the
        metadata cache with all available skills.
        """
        skills_dirs = self.get_resolved_skills_dirs()
        self._metadata_cache = await self.registry.discover_skills(skills_dirs)
        self._change_token = self._prompt_catalog.compute_change_token(skills_dirs)
        self._initialized = True
        logger.info("skill_manager_initialized", skill_count=len(self._metadata_cache))

    def get_skill_tool(self) -> SkillTool:
        """
        Get the SkillTool instance for agent integration.

        Returns:
            SkillTool instance
        """
        if self._skill_tool is None:
            self._skill_tool = SkillTool(
                registry=self.registry,
                loader=self.loader,
            )
        return self._skill_tool

    def render_skills_section(self) -> str:
        """
        Generate system prompt section listing available skills.

        Returns a Markdown-formatted section that can be appended to
        agent system prompts. Only includes metadata (name + description)
        to keep context size small.

        Returns:
            Markdown string with skills list, empty string if no skills
        """
        return self._prompt_catalog.render_section(self._metadata_cache)

    async def reload(self) -> None:
        """
        Reload skills from configured directories.

        Re-scans skill directories and updates the metadata cache.
        """
        skills_dirs = self.get_resolved_skills_dirs()
        self._metadata_cache = await self.registry.discover_skills(skills_dirs)
        self._change_token = self._prompt_catalog.compute_change_token(skills_dirs)
        self._initialized = True
        logger.info("skills_reloaded", skill_count=len(self._metadata_cache))

    async def refresh_if_changed(self) -> None:
        if not self._initialized:
            await self.initialize()
            return
        skills_dirs = self.get_resolved_skills_dirs()
        current_token = self._prompt_catalog.compute_change_token(skills_dirs)
        if current_token != self._change_token:
            await self.reload()

    def get_resolved_skills_dirs(self) -> list[Path]:
        """Return the current effective skill directories."""
        dirs: list[Path] = []
        for resolved in resolve_skill_dirs(self._config):
            if resolved.exists():
                dirs.append(resolved)
            else:
                logger.debug("skill_dir_not_found", path=str(resolved))
        return dirs

    def get_change_token(self) -> str:
        return self._change_token

    def get_prompt_snapshot(self) -> SkillPromptSnapshot:
        return SkillPromptSnapshot(
            rendered_section=self.render_skills_section(),
            change_token=self._change_token,
        )
