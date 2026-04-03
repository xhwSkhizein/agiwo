"""Definition-scoped assembly helpers for Agent."""

from dataclasses import dataclass, replace as dataclass_replace
from typing import TYPE_CHECKING

from agiwo.agent.models.config import AgentConfig
from agiwo.agent.hooks import AgentHooks, DefaultMemoryHook
from agiwo.agent.prompt import compose_child_system_prompt
from agiwo.skill.allowlist import validate_expanded_allowed_skills
from agiwo.skill.manager import get_global_skill_manager
from agiwo.skill.skill_tool import SkillTool
from agiwo.tool.base import BaseTool
from agiwo.tool.registry import (
    build_agent_tools,
    exact_tool_disable_set,
    normalize_disabled_sdk_tool_names,
)
from agiwo.workspace import AgentWorkspace, build_agent_workspace

if TYPE_CHECKING:
    from agiwo.agent.agent import Agent


@dataclass(frozen=True)
class ResolvedAgentDefinition:
    hooks: AgentHooks
    tools: tuple[BaseTool, ...]
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


def resolve_agent_definition(
    *,
    config: AgentConfig,
    agent_id: str,
    tools: list[BaseTool] | None,
    hooks: AgentHooks | None,
) -> ResolvedAgentDefinition:
    del agent_id
    resolved_hooks = build_agent_hooks(config, hooks)
    resolved_tools = build_agent_tools(
        tools=tools,
        disabled_sdk_tool_names=config.disabled_sdk_tool_names,
        allowed_skills=config.allowed_skills,
    )
    workspace = build_agent_workspace(
        root_path=config.options.get_effective_root_path(),
        agent_name=config.name,
    )
    return ResolvedAgentDefinition(
        hooks=resolved_hooks,
        tools=resolved_tools,
        workspace=workspace,
    )


def resolve_child_definition(
    agent: "Agent",
    *,
    instruction: str | None,
    system_prompt_override: str | None,
    exclude_tool_names: set[str] | None,
    extra_tools: list[BaseTool] | None = None,
    child_allowed_skills: list[str] | None = None,
) -> ResolvedChildDefinition:
    tool_names_to_skip = normalize_disabled_sdk_tool_names(exclude_tool_names)
    child_tools = [
        tool
        for tool in agent.tools
        if tool.name not in tool_names_to_skip and not isinstance(tool, SkillTool)
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
    validate_expanded_allowed_skills(child_allowed_skills)
    validated_child_allowed = get_global_skill_manager().validate_explicit_allowed_skills(
        child_allowed_skills
    )
    if child_allowed_skills is None:
        effective_allowed = (
            list(agent.config.allowed_skills)
            if agent.config.allowed_skills is not None
            else None
        )
    else:
        effective_allowed = validated_child_allowed
        if agent.config.allowed_skills is not None and effective_allowed is not None:
            parent_allowed = set(agent.config.allowed_skills)
            disallowed = [
                skill for skill in effective_allowed if skill not in parent_allowed
            ]
            if disallowed:
                skill_list = ", ".join(disallowed)
                raise ValueError(
                    "child_allowed_skills must be a subset of the parent's "
                    f"allowed_skills: {skill_list}"
                )
    resolved_tools = build_agent_tools(
        tools=child_tools,
        disabled_sdk_tool_names=child_disabled_tool_names,
        allowed_skills=effective_allowed,
    )
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
        allowed_skills=effective_allowed,
    )
    return ResolvedChildDefinition(
        config=child_config,
        tools=resolved_tools,
    )


__all__ = [
    "ResolvedAgentDefinition",
    "ResolvedChildDefinition",
    "build_agent_hooks",
    "resolve_agent_definition",
    "resolve_child_definition",
]
