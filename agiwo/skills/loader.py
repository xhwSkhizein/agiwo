"""
Skill Loader - Load skill content and resolve resources.

This module provides functionality to load full skill content from SKILL.md
files and resolve resource paths (scripts, references, assets) with
{baseDir} variable substitution.
"""

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from agiwo.skills.exceptions import SkillNotFoundError, SkillParseError
from agiwo.skills.registry import SkillMetadata, SkillRegistry
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class SkillContent(BaseModel):
    """
    Complete skill content including metadata and body.

    Contains the full SKILL.md content parsed into frontmatter and body.
    """

    metadata: SkillMetadata
    body: str  # Markdown body content after frontmatter
    frontmatter: dict[str, Any]  # Complete YAML frontmatter


class SkillLoader:
    """
    Loader for skill content and resources.

    Responsibilities:
    - Load complete SKILL.md files
    - Resolve {baseDir} variable in content
    - Provide paths to scripts, references, and assets
    """

    def __init__(self, registry: SkillRegistry) -> None:
        """
        Initialize skill loader with registry.

        Args:
            registry: SkillRegistry instance for metadata lookup
        """
        self.registry = registry

    async def load_skill(self, skill_name: str) -> SkillContent:
        """
        Load complete SKILL.md content for a skill.

        Parses the file to extract frontmatter and body, then resolves
        {baseDir} variables in the body content.

        Args:
            skill_name: Name of the skill to load

        Returns:
            SkillContent with metadata, body, and frontmatter

        Raises:
            SkillNotFoundError: If skill is not found
            SkillParseError: If parsing fails
        """
        metadata = self.registry.get_metadata(skill_name)
        if not metadata:
            raise SkillNotFoundError(f"Skill '{skill_name}' not found")

        try:
            content = metadata.path.read_text(encoding="utf-8")
        except Exception as e:
            raise SkillParseError(
                f"Failed to read SKILL.md for '{skill_name}': {e}"
            ) from e

        # Extract frontmatter and body
        match = re.match(r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL)
        if not match:
            raise SkillParseError(
                f"Invalid SKILL.md format for '{skill_name}': missing frontmatter"
            )

        frontmatter_str = match.group(1)
        body = match.group(2)

        try:
            frontmatter = yaml.safe_load(frontmatter_str)
        except Exception as e:
            raise SkillParseError(
                f"Invalid YAML frontmatter for '{skill_name}': {e}"
            ) from e

        if not isinstance(frontmatter, dict):
            raise SkillParseError(
                f"Frontmatter must be a dictionary for '{skill_name}'"
            )

        # Resolve {baseDir} variable
        resolved_body = self.resolve_base_dir(skill_name, body)

        return SkillContent(
            metadata=metadata,
            body=resolved_body,
            frontmatter=frontmatter,
        )

    def resolve_base_dir(self, skill_name: str, content: str) -> str:
        """
        Replace {baseDir} variable with skill root directory absolute path.

        Args:
            skill_name: Name of the skill
            content: Content string that may contain {baseDir}

        Returns:
            Content with {baseDir} replaced by absolute path
        """
        metadata = self.registry.get_metadata(skill_name)
        if not metadata:
            return content

        # Use resolve() to normalize paths (handles Windows short names like RUNNER~1)
        base_dir = str(metadata.base_dir.resolve())
        return content.replace("{baseDir}", base_dir)

    async def load_reference(self, skill_name: str, rel_path: str) -> str:
        """
        Load a reference file from the skill's references/ directory.

        Args:
            skill_name: Name of the skill
            rel_path: Relative path from references/ directory

        Returns:
            File content as string

        Raises:
            SkillNotFoundError: If skill is not found
            FileNotFoundError: If reference file is not found
        """
        metadata = self.registry.get_metadata(skill_name)
        if not metadata:
            raise SkillNotFoundError(f"Skill '{skill_name}' not found")

        ref_path = metadata.base_dir / "references" / rel_path
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Reference file not found: {ref_path} for skill '{skill_name}'"
            )

        return ref_path.read_text(encoding="utf-8")

    async def get_script_path(self, skill_name: str, script_name: str) -> Path:
        """
        Get absolute path to a script in the skill's scripts/ directory.

        Args:
            skill_name: Name of the skill
            script_name: Name of the script file

        Returns:
            Absolute path to the script

        Raises:
            SkillNotFoundError: If skill is not found
        """
        metadata = self.registry.get_metadata(skill_name)
        if not metadata:
            raise SkillNotFoundError(f"Skill '{skill_name}' not found")

        return (metadata.base_dir / "scripts" / script_name).absolute()

    async def get_asset_path(self, skill_name: str, asset_name: str) -> Path:
        """
        Get absolute path to an asset in the skill's assets/ directory.

        Args:
            skill_name: Name of the skill
            asset_name: Name of the asset file

        Returns:
            Absolute path to the asset

        Raises:
            SkillNotFoundError: If skill is not found
        """
        metadata = self.registry.get_metadata(skill_name)
        if not metadata:
            raise SkillNotFoundError(f"Skill '{skill_name}' not found")

        return (metadata.base_dir / "assets" / asset_name).absolute()
