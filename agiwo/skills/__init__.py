"""
Agent Skills - Progressive disclosure skill system.

This module provides support for Agent Skills specification, enabling
agents to discover, activate, and execute skills with progressive disclosure
to optimize context usage.
"""

from agiwo.skills.exceptions import SkillError, SkillNotFoundError, SkillParseError
from agiwo.skills.loader import SkillContent, SkillLoader
from agiwo.skills.manager import SkillManager
from agiwo.skills.registry import SkillMetadata, SkillRegistry
from agiwo.skills.tool import SkillTool

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
]
