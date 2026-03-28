"""Definition-scoped assembly helpers for Agent."""

from dataclasses import dataclass, replace as dataclass_replace
from typing import TYPE_CHECKING

from agiwo.agent.models.config import AgentConfig
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.hooks.memory import DefaultMemoryHook
from agiwo.config.settings import settings
from agiwo.skill.config import SkillDiscoveryConfig, normalize_skill_dirs
from agiwo.skill.manager import SkillManager
from agiwo.tool.base import BaseTool
from agiwo.tool.builtin import ensure_builtin_tools_loaded
from agiwo.tool.builtin.bash_tool import ensure_bash_tool_pair
from agiwo.tool.builtin.registry import DEFAULT_TOOLS
from agiwo.workspace import AgentWorkspace, build_agent_workspace

if TYPE_CHECKING:
    from agiwo.agent.agent import Agent


@dataclass(frozen=True)
class ResolvedAgentDefinition:
    hooks: AgentHooks
    tools: tuple[BaseTool, ...]
    skill_manager: SkillManager | None
    workspace: AgentWorkspace


@dataclass(frozen=True)
class ResolvedChildDefinition:
    config: AgentConfig
    tools: tuple[BaseTool, ...]


def build_agent_hooks(
    config: AgentConfig,
    hooks: AgentHooks | None,
) -> AgentHooks:
    resolved = dataclass_replace(hooks or AgentHooks())
    if resolved.on_memory_retrieve is None:
        memory_hook = DefaultMemoryHook(
            root_path=config.options.get_effective_root_path()
        )
        resolved.on_memory_retrieve = memory_hook.retrieve_memories
    return resolved


def build_skill_manager(config: AgentConfig) -> SkillManager | None:
    if not config.options.enable_skill:
        return None
    return SkillManager(
        SkillDiscoveryConfig(
            configured_dirs=normalize_skill_dirs(config.options.skills_dirs),
            env_dirs=list(settings.skills_dirs or []),
            root_path=config.options.get_effective_root_path(),
        )
    )


def normalize_disabled_sdk_tool_names(names: set[str] | None) -> set[str]:
    normalized = set(names or set())
    if "bash" in normalized:
        normalized.add("bash_process")
    if "bash_process" in normalized:
        normalized.add("bash")
    return normalized


def exact_tool_disable_set() -> set[str]:
    return {"skill", *DEFAULT_TOOLS.keys()}


def build_agent_tools(
    *,
    tools: list[BaseTool] | None,
    disabled_sdk_tool_names: set[str] | None,
    skill_manager: SkillManager | None,
) -> tuple[BaseTool, ...]:
    ensure_builtin_tools_loaded()
    disabled_names = normalize_disabled_sdk_tool_names(disabled_sdk_tool_names)
    provided_tools = list(tools or [])
    base_tool_names = {tool.get_name() for tool in provided_tools}
    default_tools: list[BaseTool] = []
    for name, tool_cls in DEFAULT_TOOLS.items():
        if name in disabled_names or name in base_tool_names:
            continue
        default_tools.append(tool_cls())

    resolved_base_tools = ensure_bash_tool_pair([*provided_tools, *default_tools])
    if skill_manager is not None and "skill" not in disabled_names:
        if all(tool.get_name() != "skill" for tool in resolved_base_tools):
            resolved_base_tools.append(skill_manager.get_skill_tool())
    return tuple(resolved_base_tools)


def compose_child_system_prompt(
    *,
    base_prompt: str,
    system_prompt_override: str | None,
    instruction: str | None,
) -> str:
    if system_prompt_override is not None:
        return system_prompt_override
    if not instruction:
        return base_prompt
    instruction_block = (
        f"<system-instruction>\n{instruction.strip()}\n</system-instruction>"
    )
    return f"{base_prompt}\n\n{instruction_block}".strip()


def resolve_agent_definition(
    *,
    config: AgentConfig,
    agent_id: str,
    tools: list[BaseTool] | None,
    hooks: AgentHooks | None,
) -> ResolvedAgentDefinition:
    resolved_hooks = build_agent_hooks(config, hooks)
    skill_manager = build_skill_manager(config)
    resolved_tools = build_agent_tools(
        tools=tools,
        disabled_sdk_tool_names=config.disabled_sdk_tool_names,
        skill_manager=skill_manager,
    )
    workspace = build_agent_workspace(
        root_path=config.options.get_effective_root_path(),
        agent_name=config.name,
        agent_id=agent_id,
    )
    return ResolvedAgentDefinition(
        hooks=resolved_hooks,
        tools=resolved_tools,
        skill_manager=skill_manager,
        workspace=workspace,
    )


def resolve_child_definition(
    agent: "Agent",
    *,
    instruction: str | None,
    system_prompt_override: str | None,
    exclude_tool_names: set[str] | None,
    extra_tools: list[BaseTool] | None = None,
) -> ResolvedChildDefinition:
    tool_names_to_skip = normalize_disabled_sdk_tool_names(exclude_tool_names)
    child_tools = [
        tool for tool in agent.tools if tool.get_name() not in tool_names_to_skip
    ]
    if extra_tools:
        child_tools.extend(extra_tools)
    child_disabled_tool_names = normalize_disabled_sdk_tool_names(
        {
            name
            for name in set(exclude_tool_names or set())
            if name in exact_tool_disable_set()
        }
    )

    child_options = agent.config.options.model_copy(deep=True)
    child_options.enable_termination_summary = True
    child_config = AgentConfig(
        name=agent.name,
        description=agent.description,
        system_prompt=compose_child_system_prompt(
            base_prompt=agent.config.system_prompt,
            system_prompt_override=system_prompt_override,
            instruction=instruction,
        ),
        options=child_options,
        disabled_sdk_tool_names=child_disabled_tool_names,
    )
    return ResolvedChildDefinition(
        config=child_config,
        tools=tuple(child_tools),
    )


__all__ = [
    "ResolvedAgentDefinition",
    "ResolvedChildDefinition",
    "build_agent_hooks",
    "build_agent_tools",
    "build_skill_manager",
    "compose_child_system_prompt",
    "exact_tool_disable_set",
    "normalize_disabled_sdk_tool_names",
    "resolve_agent_definition",
    "resolve_child_definition",
]
