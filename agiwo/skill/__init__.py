"""
Agent Skills - Progressive disclosure skill system.

This module provides support for Agent Skills specification, enabling
agents to discover, activate, and execute skills with progressive disclosure
to optimize context usage.
"""

from agiwo.skill.exceptions import SkillError, SkillNotFoundError, SkillParseError
from agiwo.skill.loader import SkillContent, SkillLoader
from agiwo.skill.manager import SkillManager
from agiwo.skill.prompt_catalog import (
    SkillPromptCatalog,
    SkillPromptProvider,
)
from agiwo.skill.skill_tool import SkillTool
from agiwo.skill.registry import SkillMetadata, SkillRegistry


__all__ = [
    "SkillError",
    "SkillNotFoundError",
    "SkillParseError",
    "SkillMetadata",
    "SkillRegistry",
    "SkillContent",
    "SkillLoader",
    "SkillTool",
    "SkillManager",
    "SkillPromptCatalog",
    "SkillPromptProvider",
]
