"""
Skill Registry - Metadata discovery and caching.

This module provides skill discovery and metadata management for the
Agent Skills system. It scans skill directories, parses SKILL.md files,
and caches metadata for efficient access.
"""

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from agiwo.skill.exceptions import SkillParseError
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class SkillMetadata(BaseModel):
    """
    Skill metadata extracted from SKILL.md frontmatter.

    Contains only essential information for skill discovery and selection.
    Full skill content is loaded on-demand when activated.
    """

    name: str  # Skill name (1-64 chars, lowercase alphanumeric and hyphens)
    description: str  # Skill description (1-1024 chars)
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    path: Path  # Path to SKILL.md file
    base_dir: Path  # Skill root directory (for resolving {baseDir} variable)


class SkillRegistry:
    """
    Registry for discovering and managing skill metadata.

    Responsibilities:
    - Scan skill directories for available skills
    - Parse SKILL.md frontmatter to extract metadata
    - Cache metadata for efficient access
    - Support hot reload for skill updates
    """

    def __init__(self) -> None:
        """Initialize skill registry with empty cache."""
        self._metadata_cache: dict[str, SkillMetadata] = {}
        self._skills_dirs: list[Path] = []

    async def discover_skills(self, skills_dirs: list[Path]) -> list[SkillMetadata]:
        """
        Discover all available skills in the given directories.

        Scans each directory for subdirectories containing SKILL.md files,
        parses frontmatter, and caches metadata. Continues processing even
        if some skills fail to parse.

        Args:
            skills_dirs: List of directories to scan for skills

        Returns:
            List of discovered skill metadata
        """
        self._skills_dirs = skills_dirs
        cache: dict[str, SkillMetadata] = {}

        for skill_dir in skills_dirs:
            skill_dir = Path(skill_dir).expanduser().resolve()
            if not skill_dir.exists():
                logger.warning("skill_dir_not_found", path=str(skill_dir))
                continue

            if not skill_dir.is_dir():
                logger.warning("skill_dir_not_directory", path=str(skill_dir))
                continue

            for skill_path in skill_dir.iterdir():
                if not skill_path.is_dir():
                    continue

                skill_md = skill_path / "SKILL.md"
                if not skill_md.exists():
                    continue

                try:
                    metadata = self._parse_skill_frontmatter(skill_md)
                    if metadata.name in cache:
                        logger.warning(
                            "duplicate_skill_name",
                            name=metadata.name,
                            existing_path=str(cache[metadata.name].path),
                            new_path=str(metadata.path),
                        )
                    cache[metadata.name] = metadata
                    logger.debug(
                        "skill_discovered",
                        name=metadata.name,
                        path=str(skill_path),
                    )
                except Exception as e:
                    logger.warning(
                        "skill_parse_failed",
                        path=str(skill_path),
                        error=str(e),
                    )

        self._metadata_cache = cache
        logger.info("skills_discovery_complete", count=len(cache))
        return list(cache.values())

    def get_metadata(self, skill_name: str) -> SkillMetadata | None:
        """
        Get metadata for a skill by name.

        Args:
            skill_name: Name of the skill

        Returns:
            SkillMetadata if found, None otherwise
        """
        return self._metadata_cache.get(skill_name)

    def list_available(self) -> list[str]:
        """
        List all available skill names.

        Returns:
            List of skill names
        """
        return list(self._metadata_cache.keys())

    async def reload(self) -> None:
        """
        Reload skills from configured directories.

        Re-scans all skill directories and updates the metadata cache.
        """
        if not self._skills_dirs:
            logger.warning("no_skills_dirs_configured")
            return

        await self.discover_skills(self._skills_dirs)

    def _parse_skill_frontmatter(self, skill_path: Path) -> SkillMetadata:
        """
        Parse YAML frontmatter from SKILL.md file.

        Validates required fields and skill name format according to
        Agent Skills specification.

        Args:
            skill_path: Path to SKILL.md file

        Returns:
            SkillMetadata with parsed information

        Raises:
            SkillParseError: If parsing fails or validation fails
        """
        try:
            content = skill_path.read_text(encoding="utf-8")
        except Exception as e:
            raise SkillParseError(f"Failed to read {skill_path}: {e}") from e

        # Extract YAML frontmatter
        match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
        if not match:
            raise SkillParseError(f"Missing YAML frontmatter in {skill_path}")

        frontmatter_str = match.group(1)

        try:
            frontmatter = yaml.safe_load(frontmatter_str)
        except Exception as e:
            raise SkillParseError(f"Invalid YAML in {skill_path}: {e}") from e

        if not isinstance(frontmatter, dict):
            raise SkillParseError(f"Frontmatter must be a dictionary in {skill_path}")

        # Validate required fields
        if "name" not in frontmatter:
            raise SkillParseError(f"Missing 'name' field in {skill_path}")

        if "description" not in frontmatter:
            raise SkillParseError(f"Missing 'description' field in {skill_path}")

        name = frontmatter["name"]
        description = frontmatter["description"]

        # Validate name format (1-64 chars, lowercase alphanumeric and hyphens)
        if not isinstance(name, str):
            raise SkillParseError(f"Skill name must be a string in {skill_path}")

        if not re.match(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$", name):
            raise SkillParseError(
                f"Invalid skill name '{name}' in {skill_path}. "
                "Name must be 1-64 characters, lowercase alphanumeric "
                "and hyphens only, cannot start or end with hyphen."
            )

        if len(name) > 64:
            raise SkillParseError(
                f"Skill name '{name}' exceeds 64 characters in {skill_path}"
            )

        # Validate description
        if not isinstance(description, str):
            raise SkillParseError(f"Skill description must be a string in {skill_path}")

        if len(description) == 0:
            raise SkillParseError(f"Skill description cannot be empty in {skill_path}")

        if len(description) > 1024:
            raise SkillParseError(
                f"Skill description exceeds 1024 characters in {skill_path}"
            )

        # Extract optional fields
        license_str = frontmatter.get("license")
        compatibility_str = frontmatter.get("compatibility")
        metadata_dict = frontmatter.get("metadata", {})

        if not isinstance(metadata_dict, dict):
            raise SkillParseError(f"Metadata must be a dictionary in {skill_path}")

        # Convert metadata values to strings
        metadata_str_dict: dict[str, str] = {}
        for key, value in metadata_dict.items():
            if not isinstance(key, str):
                continue
            metadata_str_dict[key] = str(value)

        return SkillMetadata(
            name=name,
            description=description,
            license=license_str if isinstance(license_str, str) else None,
            compatibility=compatibility_str
            if isinstance(compatibility_str, str)
            else None,
            metadata=metadata_str_dict,
            path=skill_path,
            base_dir=skill_path.parent,
        )
