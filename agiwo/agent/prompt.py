"""System prompt construction."""

import locale
import platform
from datetime import datetime
from typing import Any

try:
    import distro
except ImportError:  # pragma: no cover - optional dependency
    distro = None

from agiwo.agent.hooks import filter_relevant_memories
from agiwo.agent.models.input import ChannelContext, UserMessage
from agiwo.agent.models.run import MemoryRecord
from agiwo.agent.models.step import StepView
from agiwo.skill.allowlist import skills_enabled
from agiwo.skill.manager import get_global_skill_manager
from agiwo.tool.base import BaseTool
from agiwo.utils.logging import get_logger
from agiwo.workspace import WorkspaceBootstrapper, WorkspaceDocumentStore
from agiwo.workspace.documents import WorkspaceDocuments
from agiwo.workspace.layout import AgentWorkspace

logger = get_logger(__name__)


def _get_os_info() -> str:
    os_name = platform.system()
    os_release = platform.release()
    os_version = platform.version()
    machine_type = platform.machine()

    extra_info = ""
    if os_name == "Darwin":
        mac_version = platform.mac_ver()[0]
        extra_info = f", macOS version:{mac_version}"
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


def _get_language_info() -> str | None:
    try:
        default_lang, _ = locale.getlocale()
        return default_lang
    except Exception as error:  # noqa: BLE001 - environment boundary
        logger.warning("get_language_info_failed", error=str(error))
        return None


def _render_identity(documents: WorkspaceDocuments) -> str:
    content = documents.identity_text
    if not content:
        return ""
    return (
        f"{content}\n"
        "_This IDENTITY.md is yours to evolve. As you learn who you are, update it._"
    ).strip()


def _render_soul(workspace: AgentWorkspace, documents: WorkspaceDocuments) -> str:
    content = documents.soul_text
    if not content:
        return ""
    return (f"---\n\n{content}\nSOUL.md path: {workspace.soul_path}\n").strip()


def _render_environment(
    workspace: AgentWorkspace,
    *,
    os_info: str,
    language_info: str | None,
    timezone: str,
    current_date: str,
) -> str:
    return f"""---\n\n## Workspace & Environment

The workspace is your domain, while other users' systems are places you visit.

Your workspace: **{workspace.workspace}**:

- **MEMORY/** — Contains persistent memory you think is important. Use format: `<yyyy-mm-dd>.md` for daily notes, use `<category>_<yyyy-mm-dd>.md` when it's a specific category note.

> Information in **MEMORY/** will auto inject into your conversation context use `<inject-memories>` tag. You also can use `memory_retrieval` to search relevant memories when you need.

- **WORK/** — Your working directory. Store task plans, findings, progress notes and working files here. **IMPORTANT**: Use Git for version control in subdirectories where you modify files.

- **Environment**
OS: {os_info}
Language: {language_info}
Time-zone: {timezone}
Date: {current_date}, use the `bash` tool get current time if there is a need
"""


def _render_goal_directed_review() -> str:
    return """---
## Goal-Directed Review

You are expected to work in a goal-directed manner. The system helps you
stay on track through a review mechanism.

### Milestones
When you receive a task, break it into concrete milestones using the
`declare_milestones` tool. Each milestone should be a verifiable
sub-goal. Keep milestones focused and specific -- "understand the code"
is too vague; "identify how auth tokens are validated" is concrete.

### System Reviews
The system will periodically ask you to review your trajectory against
the active milestone. When you see a `<system-review>` tag in a tool
result, you MUST respond with the `review_trajectory` tool:

- If your recent steps advance the milestone: set `aligned=true` and
  briefly note what was accomplished.
- If your recent steps drifted from the milestone: set `aligned=false`
  and provide a concise experience summary of what was learned.
  The system will condense the off-track results so they don't clutter
  the context.

The system review is not optional -- it enforces that you periodically
check your own direction. Treat it as a mandatory checkpoint.

### Step-Back
When you indicate misalignment, the system automatically condenses the
off-target tool results into your experience summary. The tool call
history is preserved so future decisions can reference what was tried,
but the verbose outputs are replaced with the lesson learned."""


def _render_tools(tools: tuple[tuple[str, str], ...]) -> str:
    if not tools:
        return ""
    lines = ["# Tools you can use", "", "<tools>"]
    for name, description in tools:
        lines.append("<tool>")
        lines.append(f"  <name>{name}</name>")
        lines.append(f"  <description>{description}</description>")
        lines.append("</tool>")
    lines.append("</tools>")
    rendered = "\n".join(lines).strip()
    return f"---\n\n{rendered}"


def _render_tools_document(documents: WorkspaceDocuments) -> str:
    content = documents.tools_text
    if not content:
        return ""
    return f"---\n\n{content}".strip()


def _render_user(documents: WorkspaceDocuments) -> str:
    content = documents.user_text
    if not content:
        return ""
    return f"---\n\n{content}".strip()


def _render_channel_context(ctx: ChannelContext) -> str:
    lines = [f"source: {ctx.source}"]
    for key, value in ctx.metadata.items():
        if key in ("recent_dm_messages", "recent_group_messages") and isinstance(
            value, list
        ):
            if value:
                lines.append(f"{key}:")
                for msg in value:
                    lines.append(f"  - {msg}")
        elif isinstance(value, (str, int, float, bool)):
            lines.append(f"{key}: {value}")
    return "<channel-context>\n" + "\n".join(lines) + "\n</channel-context>"


def _render_memories(memories: list[MemoryRecord]) -> str:
    content = "\n\n".join(memory.content for memory in memories)
    return f"<relevant-memories>\n{content}\n</relevant-memories>"


def _render_hook_result(result: str) -> str:
    return f"<before_run_hook_result>\n{result}\n</before_run_hook_result>"


def system_notice(content: str) -> str:
    """Create a standardized inline system-notice tag.

    The format is inline: <system-notice>{content}</system-notice>
    Use this for all system notices to ensure consistent formatting.
    """
    return f"<system-notice>{content}</system-notice>"


def _prepend_to_user_message(msg: dict[str, Any], preamble: str) -> None:
    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = preamble + "\n\n" + content
    elif isinstance(content, list):
        msg["content"] = [{"type": "text", "text": preamble}] + content
    else:
        msg["content"] = preamble


def assemble_run_messages(
    system_prompt: str,
    existing_steps: list[StepView] | None = None,
    memories: list[MemoryRecord] | None = None,
    before_run_hook_result: str | None = None,
    *,
    channel_context: ChannelContext | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        step.to_message() for step in (existing_steps or [])
    ]
    filtered_memories = filter_relevant_memories(messages, memories or [])

    preamble_parts: list[str] = []
    if channel_context:
        preamble_parts.append(_render_channel_context(channel_context))
    if filtered_memories:
        preamble_parts.append(_render_memories(filtered_memories))
    if before_run_hook_result:
        preamble_parts.append(_render_hook_result(before_run_hook_result))

    if preamble_parts:
        preamble_text = "\n\n".join(preamble_parts)
        if messages and messages[-1].get("role") == "user":
            _prepend_to_user_message(messages[-1], preamble_text)
        elif preamble_text.strip():
            messages.append({"role": "user", "content": preamble_text})

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    return messages


def apply_steering_messages(
    messages: list[dict[str, Any]],
    steer_inputs: list[UserMessage] | None,
) -> list[dict[str, Any]]:
    """Append staged steer inputs to ``messages`` and return it.

    The caller owns ``messages`` (typically a fresh ``state.snapshot_messages()``
    deep copy).  This function mutates and returns the same list so we avoid a
    per-turn deep copy of the entire conversation history.
    """
    if not steer_inputs:
        return messages

    for normalized in steer_inputs:
        if not normalized.has_content():
            continue
        messages.append(
            {
                "role": "user",
                "content": normalized.to_message_content(),
            }
        )

    return messages


async def build_system_prompt(
    *,
    base_prompt: str,
    workspace: AgentWorkspace,
    tools: list[BaseTool] | None = None,
    allowed_skills: list[str] | None = None,
    bootstrapper: WorkspaceBootstrapper,
    document_store: WorkspaceDocumentStore,
) -> str:
    await bootstrapper.ensure_prompt_ready(workspace)
    skill_manager = (
        get_global_skill_manager() if skills_enabled(allowed_skills) else None
    )
    if skill_manager is not None:
        await skill_manager.initialize()
        await skill_manager.refresh_if_changed()

    current_dt = datetime.now().astimezone()
    skills_section = ""
    if skill_manager is not None:
        skills_section = skill_manager.render_skills_section(allowed_skills)

    documents = document_store.read(workspace)
    sections = [
        _render_identity(documents),
        _render_soul(workspace, documents),
        base_prompt.strip() if base_prompt else "",
        _render_environment(
            workspace,
            os_info=_get_os_info(),
            language_info=_get_language_info(),
            timezone=str(current_dt.tzinfo),
            current_date=current_dt.strftime("%Y-%m-%d"),
        ),
        _render_goal_directed_review(),
        _render_tools_document(documents),
        _render_tools(
            tuple((tool.name, tool.get_short_description()) for tool in (tools or []))
        ),
        f"---\n\n{skills_section}".strip() if skills_section else "",
        _render_user(documents),
    ]
    return "\n\n".join(filter(None, sections))


def compose_child_system_prompt(
    *,
    base_prompt: str,
    system_prompt_override: str | None,
    instruction: str | None,
) -> str:
    """Build the system prompt for a child / nested agent run."""
    if system_prompt_override is not None:
        return system_prompt_override
    if instruction is None or not instruction.strip():
        return base_prompt
    instruction_block = (
        f"<system-instruction>\n{instruction.strip()}\n</system-instruction>"
    )
    return f"{base_prompt}\n\n{instruction_block}".strip()


__all__ = [
    "apply_steering_messages",
    "assemble_run_messages",
    "build_system_prompt",
    "compose_child_system_prompt",
    "system_notice",
]
