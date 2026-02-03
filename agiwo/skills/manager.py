"""
Skill Manager - Unified entry point for Agent Skills system.

This module provides the main SkillManager class that coordinates skill
discovery, loading, and tool creation. It also generates system prompts
with available skills for agents.
"""

import json
import os
from pathlib import Path

from agiwo.skills.loader import SkillLoader
from agiwo.skills.registry import SkillMetadata, SkillRegistry
from agiwo.skills.tool import SkillTool
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
        skill_dirs: list[Path],
    ) -> None:
        """
        Initialize skill manager.

        Args:
            skill_dirs: List of directories to scan for skills
        """
        self._skill_dirs = skill_dirs
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
        skill_dirs = self._resolve_skill_dirs()
        self._metadata_cache = await self.registry.discover_skills(skill_dirs)
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
        lines.append(
            "These skills are discovered at startup. Each entry includes a name "
            "and description. Use the Skill tool to activate a skill when needed."
        )
        lines.append("")

        for metadata in self._metadata_cache:
            lines.append(f"- **{metadata.name}**: {metadata.description}")

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
        skill_dirs = self._resolve_skill_dirs()
        self._metadata_cache = await self.registry.discover_skills(skill_dirs)
        logger.info("skills_reloaded", skill_count=len(self._metadata_cache))

    def _resolve_skill_dirs(self) -> list[Path]:
        """
        Resolve skill directories from configuration and environment.

        Returns:
            List of resolved Path objects
        """
        dirs: list[Path] = []

        # Add configured directories
        for skill_dir in self._skill_dirs:
            resolved = Path(skill_dir).expanduser().resolve()
            if resolved.exists():
                dirs.append(resolved)
            else:
                logger.debug("skill_dir_not_found", path=str(resolved))

        for env_dir in _iter_skill_dirs_from_env():
            resolved = Path(env_dir).expanduser().resolve()
            if resolved.exists():
                dirs.append(resolved)
            else:
                logger.debug("skill_dir_not_found", path=str(resolved))

        return dirs


def _iter_skill_dirs_from_env() -> list[str]:
    raw = os.getenv("AGIO_SKILLS_DIRS") or os.getenv("AGIO_SKILLS_DIR")
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, str):
        items = [parsed]
    elif parsed is None:
        items = raw.split(",")
    else:
        items = [raw]

    return [item.strip() for item in items if str(item).strip()]
