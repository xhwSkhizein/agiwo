"""
Skill Manager - Unified entry point for Agent Skills system.

This module provides the main SkillManager class that coordinates skill
discovery, loading, and tool creation. It also generates system prompts
with available skills for agents.
"""

from pathlib import Path

from agiwo.config.settings import get_settings
from agiwo.skill.allowlist import (
    expand_allowed_skills,
    normalize_allowed_skills,
    validate_expanded_allowed_skills,
    validate_known_allowed_skills,
)
from agiwo.skill.config import SkillDiscoveryConfig, resolve_skill_dirs
from agiwo.skill.loader import SkillLoader
from agiwo.skill.prompt_catalog import SkillPromptCatalog
from agiwo.skill.registry import SkillMetadata, SkillRegistry
from agiwo.skill.skill_tool import SkillTool
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


_GLOBAL_SKILL_MANAGER: "SkillManager | None" = None
_GLOBAL_SKILL_MANAGER_KEY: tuple[str, tuple[str, ...]] | None = None


def _global_skill_manager_key() -> tuple[str, tuple[str, ...]]:
    runtime_settings = get_settings()
    return (
        runtime_settings.root_path,
        tuple(runtime_settings.skills_dirs),
    )


def build_global_skill_manager() -> "SkillManager":
    root_path, skills_dirs = _global_skill_manager_key()
    return SkillManager(
        SkillDiscoveryConfig(
            skills_dirs=list(skills_dirs),
            root_path=root_path,
        )
    )


def get_global_skill_manager() -> "SkillManager":
    global _GLOBAL_SKILL_MANAGER
    global _GLOBAL_SKILL_MANAGER_KEY

    key = _global_skill_manager_key()
    if _GLOBAL_SKILL_MANAGER is None or _GLOBAL_SKILL_MANAGER_KEY != key:
        _GLOBAL_SKILL_MANAGER = build_global_skill_manager()
        _GLOBAL_SKILL_MANAGER_KEY = key
    return _GLOBAL_SKILL_MANAGER


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
        self.initialize_sync()

    def initialize_sync(self) -> None:
        if self._initialized:
            return
        skills_dirs = self.get_resolved_skills_dirs()
        self._metadata_cache = self.registry.discover_skills_sync(skills_dirs)
        self._change_token = self._prompt_catalog.compute_change_token(skills_dirs)
        self._initialized = True
        logger.info("skill_manager_initialized", skill_count=len(self._metadata_cache))

    def create_skill_tool(
        self,
        allowed_skills: list[str] | None = None,
    ) -> SkillTool:
        return SkillTool(
            registry=self.registry,
            loader=self.loader,
            allowed_skills=allowed_skills,
        )

    def render_skills_section(
        self,
        allowed_skills: list[str] | None = None,
    ) -> str:
        items = self._metadata_cache
        if allowed_skills is not None:
            allowed_set = set(allowed_skills)
            items = [item for item in items if item.name in allowed_set]
        return self._prompt_catalog.render_section(items)

    def list_available_skills(self) -> list[SkillMetadata]:
        return list(self._metadata_cache)

    def list_available_skill_names(self) -> list[str]:
        return [item.name for item in self._metadata_cache]

    def expand_allowed_skills(
        self,
        allowed_skills: list[str] | None,
        *,
        available_skill_names: list[str] | None = None,
    ) -> list[str] | None:
        self.initialize_sync()
        universe = available_skill_names or self.list_available_skill_names()
        return expand_allowed_skills(allowed_skills, universe)

    def validate_explicit_allowed_skills(
        self,
        allowed_skills: list[str] | None,
        *,
        available_skill_names: list[str] | None = None,
    ) -> list[str] | None:
        self.initialize_sync()
        normalized = normalize_allowed_skills(allowed_skills)
        validate_expanded_allowed_skills(normalized)
        universe = available_skill_names or self.list_available_skill_names()
        validate_known_allowed_skills(normalized, universe)
        return list(normalized) if normalized is not None else None

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
