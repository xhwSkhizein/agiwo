import locale
import platform

try:
    import distro
except ImportError:  # pragma: no cover - optional dependency
    distro = None

from agiwo.agent.prompt.snapshot import PromptSnapshot
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


def get_os_info() -> str:
    os_name = platform.system()
    os_release = platform.release()
    os_version = platform.version()
    machine_type = platform.machine()

    extra_info = ""
    if os_name == "Darwin":
        mac_version = platform.mac_ver()[0]
        extra_info = f", macOS verion:{mac_version}"
    elif os_name == "Windows":
        win_ver = platform.win32_ver()[1]
        extra_info = f", Windows version detail:{win_ver}"
    elif os_name == "Linux" and distro:
        distro_name = distro.name()
        distro_ver = distro.version()
        extra_info = f", Linux distro:{distro_name} {distro_ver}"

    return (
        f"OS:{os_name}, os_release:{os_release}, os_version:{os_version}, "
        f"arch:{machine_type}{extra_info}"
    )


def get_language_info() -> str | None:
    try:
        default_lang, _ = locale.getlocale()
        return default_lang
    except Exception as error:  # noqa: BLE001 - environment boundary
        logger.warning("get_language_info_failed", error=str(error))
        return None


def render_identity(snapshot: PromptSnapshot) -> str:
    content = snapshot.documents.identity_text
    if not content:
        return ""
    return (
        f"{content}\n"
        "_This IDENTITY.md is yours to evolve. As you learn who you are, update it._"
    ).strip()


def render_soul(snapshot: PromptSnapshot) -> str:
    content = snapshot.documents.soul_text
    if not content:
        return ""
    return (
        f"---\n\n{content}\nSOUL.md path: {snapshot.workspace.soul_path}\n"
    ).strip()


def render_base(snapshot: PromptSnapshot) -> str:
    return snapshot.base_prompt.strip() if snapshot.base_prompt else ""


def render_environment(snapshot: PromptSnapshot) -> str:
    workspace = snapshot.workspace.workspace
    environment = snapshot.environment
    return f"""---\n\n## Workspace & Environment

The workspace is your domain, while other users' systems are places you visit.

Your workspace: **{workspace}**:

- **MEMORY/** — Contains persistent memory you think is important. Use format: `<yyyy-mm-dd>.md` for daily notes, use `<category>_<yyyy-mm-dd>.md` when it's a specific category note.

> Information in **MEMORY/** will auto inject into your conversation context use `<inject-memories>` tag. You also can use `memory_retrieval` to search relevant memories when you need.

- **WORK/** — Your working directory. Store task plans, findings, progress notes and working files here. **IMPORTANT**: Use Git for version control in subdirectories where you modify files.

- **Environment**
OS: {environment.os_info}
Language: {environment.language_info}
Time-zone: {environment.timezone}
Date: {environment.current_date}, use the `bash` tool get current time if there is a need
"""


def render_tools(snapshot: PromptSnapshot) -> str:
    if not snapshot.tools:
        return ""
    lines = ["# Tools you can use", "", "<tools>"]
    for tool in snapshot.tools:
        lines.append("<tool>")
        lines.append(f"  <name>{tool.name}</name>")
        lines.append(f"  <description>{tool.description}</description>")
        lines.append("</tool>")
    lines.append("</tools>")
    rendered = "\n".join(lines).strip()
    return f"---\n\n{rendered}"


def render_skills(snapshot: PromptSnapshot) -> str:
    if not snapshot.skills_section:
        return ""
    return f"---\n\n{snapshot.skills_section}".strip()


def render_user(snapshot: PromptSnapshot) -> str:
    content = snapshot.documents.user_text
    if not content:
        return ""
    return f"---\n\n{content}".strip()


def render_prompt(snapshot: PromptSnapshot) -> str:
    sections = [
        render_identity(snapshot),
        render_soul(snapshot),
        render_base(snapshot),
        render_environment(snapshot),
        render_tools(snapshot),
        render_skills(snapshot),
        render_user(snapshot),
    ]
    return "\n\n".join(filter(None, sections))
