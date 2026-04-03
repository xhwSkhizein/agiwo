"""
Agent Skills - Progressive disclosure skill system.

This module provides support for Agent Skills specification, enabling
agents to discover, activate, and execute skills with progressive disclosure
to optimize context usage.
"""

from agiwo.skill.exceptions import SkillError, SkillNotFoundError, SkillParseError
from agiwo.skill.allowlist import (
    contains_allowed_skill_patterns,
    expand_allowed_skills,
    matches_allowed_skill,
    normalize_allowed_skills,
    validate_expanded_allowed_skills,
)
from agiwo.skill.loader import SkillContent, SkillLoader
from agiwo.skill.manager import (
    SkillManager,
    build_global_skill_manager,
    get_global_skill_manager,
)
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
    "contains_allowed_skill_patterns",
    "expand_allowed_skills",
    "SkillMetadata",
    "SkillRegistry",
    "SkillContent",
    "SkillLoader",
    "SkillTool",
    "SkillManager",
    "build_global_skill_manager",
    "get_global_skill_manager",
    "matches_allowed_skill",
    "normalize_allowed_skills",
    "validate_expanded_allowed_skills",
    "SkillPromptCatalog",
    "SkillPromptProvider",
]
