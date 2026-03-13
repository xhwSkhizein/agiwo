"""
Skill Manager - Unified entry point for Agent Skills system.

This module provides the main SkillManager class that coordinates skill
discovery, loading, and tool creation. It also generates system prompts
with available skills for agents.
"""

from pathlib import Path

from agiwo.config.settings import settings
from agiwo.agent.options import resolve_skills_dirs
from agiwo.skill.loader import SkillLoader
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
        skills_dirs: list[Path],
    ) -> None:
        """
        Initialize skill manager.

        Args:
            skills_dirs: List of directories to scan for skills
        """
        self._skills_dirs = skills_dirs
        self.registry = SkillRegistry()
        self.loader = SkillLoader(self.registry)
        self._skill_tool: SkillTool | None = None
        self._metadata_cache: list[SkillMetadata] = []

    async def initialize(self) -> None:
        """
        Initialize skill manager by discovering all available skills.

        This should be called during agent startup to populate the
        metadata cache with all available skills.
        """
        skills_dirs = self._resolve_skills_dirs()
        self._metadata_cache = await self.registry.discover_skills(skills_dirs)
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
        if not self._metadata_cache:
            return ""

        lines = ["## Available Skills"]
        lines.append("\n")
        lines.append("Skills are tools. Use them quietly. The user doesn't need to see the machinery.")
        lines.append(
            "These skills are discovered at startup. Each entry includes a name and description. "
            "Use the Skill tool to activate a skill when needed."
        )
        lines.append("")
        
        lines.append("<avaliable_skills>")
        for metadata in self._metadata_cache:
            lines.append("  <skill>")
            lines.append(f"    <name>{metadata.name}</name>")
            lines.append(f"    <description>{metadata.description}</description>")
            lines.append(f"    <location>{metadata.path}</location>")
            lines.append("  </skill>")
        lines.append("</avaliable_skills>")
        
        lines.append("")
        lines.append("### How to use skills:")
        lines.append(
            "1. When a user task matches a skill's description, use the Skill tool "
            "to activate it."
        )
        lines.append(
            "2. After activation, follow the instructions in the skill's SKILL.md file."
        )
        lines.append(
            "3. Load reference files (references/) only when needed for specific steps."
        )
        lines.append(
            "4. Execute scripts (scripts/) only when the skill instructions require it."
        )
        lines.append(
            "5. Use assets (assets/) as templates or resources, don't load their content."
        )

        return "\n".join(lines)

    async def reload(self) -> None:
        """
        Reload skills from configured directories.

        Re-scans skill directories and updates the metadata cache.
        """
        skills_dirs = self._resolve_skills_dirs()
        self._metadata_cache = await self.registry.discover_skills(skills_dirs)
        logger.info("skills_reloaded", skill_count=len(self._metadata_cache))

    def get_resolved_skills_dirs(self) -> list[Path]:
        """Return the current effective skill directories."""
        return self._resolve_skills_dirs()

    def _resolve_skills_dirs(self) -> list[Path]:
        """
        Resolve skill directories from configuration and environment.

        Returns:
            List of resolved Path objects (existing only)
        """
        raw_dirs = [str(d) for d in self._skills_dirs] + settings.get_env_skills_dirs()
        all_resolved = resolve_skills_dirs(raw_dirs, str(settings.get_root_path()))

        dirs: list[Path] = []
        for resolved in all_resolved:
            if resolved.exists():
                dirs.append(resolved)
            else:
                logger.debug("skill_dir_not_found", path=str(resolved))
        return dirs
