"""SystemPromptBuilder - System prompt construction with modular sections.

This module provides a clean, extensible way to build system prompts
with composable sections (base, environment, skills, etc.)
"""

import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from agiwo.skill.manager import SkillManager
from agiwo.utils.logging import get_logger

if TYPE_CHECKING:
    from agiwo.agent.options import AgentOptions

logger = get_logger(__name__)


class SystemPromptBuilder(Protocol):
    """Protocol for building system prompts."""

    def build(self) -> str:
        """Build the complete system prompt."""
        ...


class DefaultSystemPromptBuilder(SystemPromptBuilder):
    """Default implementation with modular sections.

    Sections are assembled in order:
    1. Base system prompt
    2. Workspace & Environment
    3. Skills (if enabled)
    """

    def __init__(self, base_prompt: str, agent_id: str, options: "AgentOptions"):
        self.base_prompt = base_prompt
        self.agent_id = agent_id
        self.options = options
        self._skill_manager: SkillManager | None = None

    def build(self) -> str:
        """Assemble system prompt from modular sections."""
        if self.options.enable_skill:
            self._skill_manager = self._create_skill_manager()

        sections = [
            self._build_base_section(),
            self._build_skills_section(),
            self._build_environment_section(),
        ]
        return "\n\n".join(filter(None, sections))

    @property
    def skill_manager(self) -> SkillManager | None:
        return self._skill_manager

    def _create_skill_manager(self) -> SkillManager:
        """Create SkillManager from options configuration."""
        if self.options.skills_dir:
            skills_dirs = [Path(self.options.skills_dir).expanduser().resolve()]
        else:
            default_dir = Path(f"~/.agent/skills").expanduser().resolve()
            default_dir.mkdir(parents=True, exist_ok=True)
            skills_dirs = [default_dir]

        manager = SkillManager(skills_dirs=skills_dirs)
        logger.info("skill_manager_created", skills_dirs=[str(d) for d in skills_dirs])
        return manager

    def _build_base_section(self) -> str:
        """Build base system prompt section."""
        return self.base_prompt.strip() if self.base_prompt else ""

    def _build_environment_section(self) -> str:
        """Build workspace and environment section."""
        work_dir = f"{os.path.expanduser('~')}/.agiwo/{self.agent_id}"
        work_dir = os.path.abspath(work_dir)
        os.makedirs(work_dir + "/NOTES", exist_ok=True)
        os.makedirs(work_dir + "/MEMORY", exist_ok=True)
        os.makedirs(work_dir + "/USER", exist_ok=True)
        os.makedirs(work_dir + "/WORK", exist_ok=True)
        current_date = datetime.now().strftime("%Y%m%d")
        current_time = datetime.now().strftime("%H:%M")
        timezone = datetime.now().astimezone().tzinfo
        return f"""## Workspace & Environment

Your workspace is: **{work_dir}**
Treat this directory as your home. Every file in this directory is accessible to you.
- **NOTES/** — Record daily summaries and lessons learned (format: yyyy-mm-dd.md)
- **MEMORY/** — Persistent memory logs for important & long-term information
- **USER/** — User preferences, choices, and decisions
- **WORK/** — Working directory for your tasks, **IMPORTANT**: After creating subdirectories under this directory, you must use Git for file state management and version control in the corresponding subdirectories if any modifications or changes are made to the files within them.

Current date: {current_date}
Current time: {current_time}
Time-zone: {timezone}
"""

    def _build_skills_section(self) -> str:
        """Build skills section if skill manager is available."""
        if self._skill_manager is None:
            return ""
        rendered = self._skill_manager.render_skills_section()
        return rendered.strip() if rendered else ""


__all__ = [
    "SystemPromptBuilder",
    "DefaultSystemPromptBuilder",
]
