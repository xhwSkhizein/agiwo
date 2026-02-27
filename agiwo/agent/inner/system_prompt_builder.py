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

    Supports automatic refresh based on file changes (SOUL.md, skills).
    """

    def __init__(
        self, base_prompt: str, agent_name: str, agent_id: str, options: "AgentOptions"
    ):
        self.base_prompt = base_prompt
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.options = options
        self._skill_manager: SkillManager | None = None
        self._workspace_initialized = False
        self._system_prompt: str | None = None
        # Change tracking for auto-refresh
        self._soul_mtime: float | None = None
        self._skills_fingerprint: str | None = None

    def build(self) -> str:
        """Assemble system prompt from modular sections."""
        sections = [
            self._build_soul_section(),
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
            default_dir = (
                Path(f"{os.path.expanduser(self.options.config_root)}/skills")
                .expanduser()
                .resolve()
            )
            default_dir.mkdir(parents=True, exist_ok=True)
            skills_dirs = [default_dir]

        manager = SkillManager(skills_dirs=skills_dirs)
        logger.info("skill_manager_created", skills_dirs=[str(d) for d in skills_dirs])
        return manager

    async def initialize(self) -> str:
        """Initialize workspace and build system prompt.
        
        This method is called lazily at execution time (not during construction)
        to avoid side effects for agents that are created but never run.
        
        Returns:
            The built system prompt string.
        """
        if not self._workspace_initialized:
            workspace = f"{os.path.expanduser(self.options.config_root)}/{self.agent_name}"
            workspace = os.path.abspath(workspace)
            instance_work_dir = f"{workspace}/WORK/{self.agent_id}"
            os.makedirs(workspace + "/NOTES", exist_ok=True)
            os.makedirs(workspace + "/MEMORY", exist_ok=True)
            os.makedirs(instance_work_dir, exist_ok=True)
            self._workspace_initialized = True

            if self.options.enable_skill:
                self._skill_manager = self._create_skill_manager()
                await self._skill_manager.initialize()

        if self._system_prompt is None:
            # Initialize change tracking state
            soul_path = Path(
                f"{os.path.expanduser(self.options.config_root)}/bodhi/SOUL.md"
            )
            if soul_path.exists():
                self._soul_mtime = soul_path.stat().st_mtime
            
            if self._skill_manager is not None:
                self._skills_fingerprint = self._compute_skills_fingerprint()
            
            self._system_prompt = self.build()
        
        return self._system_prompt

    async def get_system_prompt(self) -> str:
        """Get system prompt, auto-refresh if source files changed.
        
        Checks SOUL.md mtime and skills directory state. If changed,
        automatically refreshes the prompt.
        
        Returns:
            The current system prompt string.
        """
        if self._system_prompt is None:
            return await self.initialize()
        
        # Check if SOUL.md changed
        soul_changed = self._check_soul_changed()
        # Check if skills changed
        skills_changed = await self._check_skills_changed()
        
        if soul_changed or skills_changed:
            logger.info("prompt_refresh_triggered", 
                       soul_changed=soul_changed, 
                       skills_changed=skills_changed)
            return await self._refresh_prompt()
        
        return self._system_prompt

    def _check_soul_changed(self) -> bool:
        """Check if SOUL.md has been modified since last build."""
        soul_path = Path(
            f"{os.path.expanduser(self.options.config_root)}/bodhi/SOUL.md"
        )
        if not soul_path.exists():
            return self._soul_mtime is not None
        
        current_mtime = soul_path.stat().st_mtime
        if self._soul_mtime is None:
            return True
        return current_mtime != self._soul_mtime

    def _compute_skills_fingerprint(self) -> str:
        """Compute a fingerprint of skills directory state.
        
        Uses file paths and mtimes to detect changes.
        """
        if self._skill_manager is None:
            return ""
        
        skills_dirs = self._resolve_skills_dirs()
        fingerprints: list[str] = []
        
        for skills_dir in skills_dirs:
            if not skills_dir.exists():
                continue
            for skill_path in skills_dir.iterdir():
                if not skill_path.is_dir():
                    continue
                skill_md = skill_path / "SKILL.md"
                if skill_md.exists():
                    mtime = skill_md.stat().st_mtime
                    fingerprints.append(f"{skill_path.name}:{mtime}")
        
        return "|".join(sorted(fingerprints))

    async def _check_skills_changed(self) -> bool:
        """Check if skills directory state has changed."""
        if self._skill_manager is None:
            return False
        
        current_fingerprint = self._compute_skills_fingerprint()
        if self._skills_fingerprint is None:
            return True
        return current_fingerprint != self._skills_fingerprint

    async def _refresh_prompt(self) -> str:
        """Refresh system prompt and update change tracking state."""
        # Update SOUL.md tracking
        soul_path = Path(
            f"{os.path.expanduser(self.options.config_root)}/bodhi/SOUL.md"
        )
        if soul_path.exists():
            self._soul_mtime = soul_path.stat().st_mtime
        else:
            self._soul_mtime = None
        
        # Update skills tracking
        if self._skill_manager is not None:
            await self._skill_manager.reload()
            self._skills_fingerprint = self._compute_skills_fingerprint()
        
        # Rebuild prompt
        self._system_prompt = self.build()
        logger.info("prompt_refreshed", 
                   soul_mtime=self._soul_mtime,
                   skills_count=len(self._skills_fingerprint.split("|")) if self._skills_fingerprint else 0)
        return self._system_prompt

    def _resolve_skills_dirs(self) -> list[Path]:
        """Resolve configured skills directories.
        
        Returns:
            List of resolved Path objects.
        """
        if self.options.skills_dir:
            return [Path(self.options.skills_dir).expanduser().resolve()]
        
        default_dir = Path(
            f"{os.path.expanduser(self.options.config_root)}/skills"
        ).expanduser().resolve()
        
        return [default_dir] if default_dir.exists() else []

    def _build_base_section(self) -> str:
        """Build base system prompt section."""
        return self.base_prompt.strip() if self.base_prompt else ""

    def _build_environment_section(self) -> str:
        """Build workspace and environment section."""
        workspace = f"{os.path.expanduser(self.options.config_root)}/{self.agent_name}"
        workspace = os.path.abspath(workspace)
        instance_work_dir = f"{workspace}/WORK/{self.agent_id}"
        current_date = datetime.now().strftime("%Y%m%d")
        current_time = datetime.now().strftime("%H:%M")
        timezone = datetime.now().astimezone().tzinfo
        return f"""## Workspace & Environment

Your workspace is: **{workspace}**
Your instance ID is: **{self.agent_id}**
Your instance work directory is: **{instance_work_dir}**

- **NOTES/** — Shared knowledge base. Use format: `{self.agent_id}_<category>_<yyyy-mm-dd>.md`
- **MEMORY/** — Shared persistent memory. Use format: `{self.agent_id}_<yyyy-mm-dd>.md`
- **WORK/{self.agent_id}/** — Your private working directory. Store task plans, findings, progress notes and working files here. **IMPORTANT**: Use Git for version control in subdirectories where you modify files.

Time-zone: {timezone}
Current date: {current_date}
Current time: {current_time}
"""

    def _build_skills_section(self) -> str:
        """Build skills section if skill manager is available."""
        if self._skill_manager is None:
            return ""
        rendered = self._skill_manager.render_skills_section()
        return rendered.strip() if rendered else ""

    def _build_soul_section(self) -> str:
        """load bodhi/SOUL.md like openclaw, it's the core of 觉"""
        soul_path = Path(
            f"{os.path.expanduser(self.options.config_root)}/bodhi/SOUL.md"
        )
        if not soul_path.exists():
            return ""
        soul = soul_path.read_text()
        logger.info("soul_path", path=str(soul_path))
        soul: str = f"{soul}\nSOUL.md path: {soul_path}\n_This file is yours to evolve. As you learn who you are, update it._"
        return soul.strip() if soul else ""


__all__ = [
    "SystemPromptBuilder",
    "DefaultSystemPromptBuilder",
]
