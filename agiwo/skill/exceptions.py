"""
Skill-related exceptions.

This module defines custom exceptions for the Agent Skills system.
"""


class SkillError(Exception):
    """Base exception for all skill-related errors."""

    pass


class SkillNotFoundError(SkillError):
    """Raised when a skill is not found."""

    pass


class SkillParseError(SkillError):
    """Raised when skill parsing fails."""

    pass
