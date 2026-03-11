"""SystemPromptBuilder - System prompt construction with modular sections.

This module provides a clean, extensible way to build system prompts
with composable sections (base, environment, skills, etc.)
"""

import os
import platform
import locale

try:
    import distro  # Linux发行版补充，非必需，没有则自动忽略
except ImportError:
    distro = None
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol


from agiwo.tool.base import BaseTool
from agiwo.utils.logging import get_logger

if TYPE_CHECKING:
    from agiwo.agent.options import AgentOptions
    from agiwo.skill.manager import SkillManager

logger = get_logger(__name__)


def get_os_info():
    """
    获取操作系统及版本信息，返回一行整合后的字符串
    :return: 操作系统信息字符串
    """
    # 基础系统信息
    os_name = platform.system()
    os_release = platform.release()
    os_version = platform.version()
    machine_type = platform.machine()

    # 不同系统补充信息
    extra_info = ""
    if os_name == "Darwin":
        mac_version = platform.mac_ver()[0]
        extra_info = f", macOS verion:{mac_version}"
    elif os_name == "Windows":
        win_ver = platform.win32_ver()[1]  # 提取Windows具体版本号
        extra_info = f", Windows version detail:{win_ver}"
    elif os_name == "Linux" and distro:
        distro_name = distro.name()
        distro_ver = distro.version()
        extra_info = f", Linux distro:{distro_name} {distro_ver}"

    # 整合为一行返回
    return f"OS:{os_name}, os_release:{os_release}, os_version:{os_version}, arch:{machine_type}{extra_info}"


def get_language_info():
    """
    获取系统默认语言信息，返回一行整合后的字符串
    :return: 语言信息字符串（失败则返回提示）
    """
    try:
        default_lang, _ = locale.getlocale()
        # 整合为一行返回
        return f"{default_lang}"
    except Exception as e:
        logger.warning(f"获取语言信息失败: {str(e)}")
        return None


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
        self,
        base_prompt: str,
        agent_name: str,
        agent_id: str,
        options: "AgentOptions",
        tools: list[BaseTool] | None = None,
        skill_manager: "SkillManager | None" = None,
    ):
        self.base_prompt = base_prompt
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.options = options
        self._skill_manager = skill_manager
        self._workspace_initialized = False
        self._system_prompt: str | None = None
        self.tools = tools
        # Computed workspace path (root_path / agent_name)
        root = Path(options.get_effective_root_path()).expanduser().resolve()
        self._workspace_path: Path = root / agent_name
        # Change tracking for auto-refresh
        self._soul_mtime: float | None = None
        self._skills_fingerprint: str | None = None

    def build(self) -> str:
        """Assemble system prompt from modular sections."""
        sections = [
            self._build_identity_section(),
            self._build_soul_section(),
            self._build_base_section(),
            self._build_environment_section(),
            self._build_tools_section(),
            self._build_skills_section(),
            self._build_user_section(),
        ]
        return "\n\n".join(filter(None, sections))

    @property
    def _soul_path(self) -> Path:
        return self._workspace_path / "SOUL.md"

    @property
    def _identity_path(self) -> Path:
        return self._workspace_path / "IDENTITY.md"

    @property
    def _user_path(self) -> Path:
        return self._workspace_path / "USER.md"

    def _ensure_template_files(self, workspace: str) -> None:
        """Ensure IDENTITY.md, SOUL.md, USER.md exist in workspace.

        If files don't exist, copy them from templates directory.

        Args:
            workspace: Absolute path to the agent workspace directory.
        """
        template_files = ["IDENTITY.md", "SOUL.md", "USER.md"]

        for filename in template_files:
            target_path = Path(workspace) / filename
            if target_path.exists():
                continue

            source_path = (
                Path(__file__).parent.parent.parent.parent / "templates" / filename
            )
            if source_path.exists():
                try:
                    shutil.copy2(source_path, target_path)
                    logger.info(
                        "copied_template_file",
                        source=str(source_path),
                        target=str(target_path),
                    )
                except Exception as e:
                    logger.warning(
                        "failed_to_copy_template", filename=filename, error=str(e)
                    )
            else:
                logger.warning("template_file_not_found", filename=str(source_path))

    async def initialize(self) -> str:
        """Initialize workspace and build system prompt.

        This method is called lazily at execution time (not during construction)
        to avoid side effects for agents that are created but never run.

        Returns:
            The built system prompt string.
        """
        if not self._workspace_initialized:
            workspace = str(self._workspace_path)
            instance_work_dir = str(self._workspace_path / "WORK" / self.agent_id)
            os.makedirs(str(self._workspace_path / "MEMORY"), exist_ok=True)
            os.makedirs(instance_work_dir, exist_ok=True)

            self._ensure_template_files(workspace)

            self._workspace_initialized = True

            if self._skill_manager is not None:
                await self._skill_manager.initialize()

        if self._system_prompt is None:
            if self._soul_path.exists():
                self._soul_mtime = self._soul_path.stat().st_mtime

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
            logger.info(
                "prompt_refresh_triggered",
                soul_changed=soul_changed,
                skills_changed=skills_changed,
            )
            return await self._refresh_prompt()

        return self._system_prompt

    def _check_soul_changed(self) -> bool:
        """Check if SOUL.md has been modified since last build."""
        if not self._soul_path.exists():
            return self._soul_mtime is not None

        current_mtime = self._soul_path.stat().st_mtime
        if self._soul_mtime is None:
            return True
        return current_mtime != self._soul_mtime

    def _compute_skills_fingerprint(self) -> str:
        """Compute a fingerprint of skills directory state.

        Uses file paths and mtimes to detect changes.
        """
        if self._skill_manager is None:
            return ""

        skills_dirs = self._get_skills_dirs_for_fingerprint()
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
        if self._soul_path.exists():
            self._soul_mtime = self._soul_path.stat().st_mtime
        else:
            self._soul_mtime = None

        # Update skills tracking
        if self._skill_manager is not None:
            await self._skill_manager.reload()
            self._skills_fingerprint = self._compute_skills_fingerprint()

        # Rebuild prompt
        self._system_prompt = self.build()
        logger.info(
            "prompt_refreshed",
            soul_mtime=self._soul_mtime,
            skills_count=len(self._skills_fingerprint.split("|"))
            if self._skills_fingerprint
            else 0,
        )
        return self._system_prompt

    def _get_skills_dirs_for_fingerprint(self) -> list[Path]:
        """Get skill directories used for change detection."""
        if self._skill_manager is not None:
            return self._skill_manager.get_resolved_skills_dirs()
        return self.options.get_configured_skills_dirs()

    def _build_identity_section(self) -> str:
        """load ${WORKSPACE}/IDENTITY.md as agent's identity."""
        if not self._identity_path.exists():
            return ""
        content = self._identity_path.read_text()
        logger.info("loaded IDENTITY.md file", path=str(self._identity_path))
        result = f"{content}\n_This IDENTITY.md is yours to evolve. As you learn who you are, update it._"
        return result.strip() if result else ""

    def _build_soul_section(self) -> str:
        """load {self.agent_name}/SOUL.md like openclaw, it's the core of 觉"""
        if not self._soul_path.exists():
            return ""
        content = self._soul_path.read_text()
        logger.info("loaded SOUL.md file", path=str(self._soul_path))
        result = f"---\n\n{content}\nSOUL.md path: {self._soul_path}\n"
        return result.strip() if result else ""

    def _build_base_section(self) -> str:
        """Build base system prompt section."""
        return self.base_prompt.strip() if self.base_prompt else ""

    def _build_environment_section(self) -> str:
        """Build workspace and environment section."""
        workspace = str(self._workspace_path)
        current_date = datetime.now().strftime("%Y-%m-%d")
        timezone = datetime.now().astimezone().tzinfo

        return f"""---\n\n## Workspace & Environment

The workspace is your domain, while other users' systems are places you visit.

Your workspace: **{workspace}**:

- **MEMORY/** — Contains persistent memory you think is important. Use format: `<yyyy-mm-dd>.md` for daily notes, use `<category>_<yyyy-mm-dd>.md` when it's a specific category note.

> Information in **MEMORY/** will auto inject into your conversation context use `<inject-memories>` tag. You also can use `memory_retrieval` to search relevant memories when you need.

- **WORK/** — Your working directory. Store task plans, findings, progress notes and working files here. **IMPORTANT**: Use Git for version control in subdirectories where you modify files.

- **Environment**
OS: {get_os_info()}
Language: {get_language_info()}
Time-zone: {timezone}
Date: {current_date}, use the `bash` tool get current time if there is a need
"""

    def _build_tools_section(self) -> str:
        if not self.tools or len(self.tools) == 0:
            return ""
        tool_section = "# Tools you can use\n\n"
        tool_section += "<tools>\n"
        for tool in self.tools:
            tool_section += f"<tool>\n  <name>{tool.name}</name>\n  <description>{tool.get_short_description()}</description>\n</tool>\n"
        tool_section += "</tools>\n"
        return f"---\n\n{tool_section.strip()}" if tool_section else ""

    def _build_skills_section(self) -> str:
        """Build skills section if skill manager is available."""
        if self._skill_manager is None:
            return ""
        rendered = f"---\n\n{self._skill_manager.render_skills_section()}"
        return rendered.strip() if rendered else ""

    def _build_user_section(self) -> str:
        """load ${WORKSPACE}/USER.md, it's user's information you talk to."""
        if not self._user_path.exists():
            return ""
        content = self._user_path.read_text()
        logger.info("loaded USER.md file", path=str(self._user_path))
        result = f"---\n\n{content}"
        return result.strip() if result else ""


__all__ = [
    "SystemPromptBuilder",
    "DefaultSystemPromptBuilder",
]
