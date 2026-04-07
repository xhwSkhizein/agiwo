"""Definition-scoped assembly helpers for Agent."""

from dataclasses import dataclass, replace as dataclass_replace
from typing import TYPE_CHECKING

from agiwo.agent.models.config import AgentConfig
from agiwo.agent.hooks import AgentHooks, DefaultMemoryHook
from agiwo.agent.prompt import compose_child_system_prompt
from agiwo.skill.allowlist import validate_expanded_allowed_skills
from agiwo.skill.manager import get_global_skill_manager
from agiwo.workspace import AgentWorkspace, build_agent_workspace

if TYPE_CHECKING:
    from agiwo.agent.agent import Agent


@dataclass(frozen=True)
class ResolvedAgentDefinition:
    hooks: AgentHooks
    workspace: AgentWorkspace


@dataclass(frozen=True)
class ResolvedChildDefinition:
    config: AgentConfig


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
    hooks: AgentHooks | None,
) -> ResolvedAgentDefinition:
    del agent_id
    resolved_hooks = build_agent_hooks(config, hooks)

    workspace = build_agent_workspace(
        root_path=config.options.get_effective_root_path(),
        agent_name=config.name,
    )
    return ResolvedAgentDefinition(
        hooks=resolved_hooks,
        workspace=workspace,
    )


def resolve_child_definition(
    agent: "Agent",
    *,
    instruction: str | None,
    system_prompt_override: str | None,
    child_allowed_skills: list[str] | None = None,
    child_allowed_tools: list[str] | None = None,
) -> ResolvedChildDefinition:
    """Resolve child agent definition, computing effective allowed_tools and allowed_skills.

    Note: Tool instances are no longer resolved here. Callers should use ToolManager.get_tools()
    to obtain tool instances based on the returned config.allowed_tools.
    """
    # Determine effective allowed_tools for child
    parent_allowed_tools = agent.config.allowed_tools
    if child_allowed_tools is None:
        # Inherit from parent if not explicitly specified
        effective_allowed_tools = parent_allowed_tools
    elif child_allowed_tools == []:
        # Empty list means no builtin tools (only extra_tools will be used)
        effective_allowed_tools = []
    else:
        # Validate child_allowed_tools is subset of parent's
        effective_allowed_tools = list(child_allowed_tools)
        if parent_allowed_tools is not None:
            parent_allowed = set(parent_allowed_tools)
            disallowed = [
                name for name in effective_allowed_tools if name not in parent_allowed
            ]
            if disallowed:
                tool_list = ", ".join(disallowed)
                raise ValueError(
                    "child_allowed_tools must be a subset of the parent's "
                    f"allowed_tools: {tool_list}"
                )

    child_options = agent.config.options.model_copy(deep=True)
    child_options.enable_termination_summary = True
    validate_expanded_allowed_skills(child_allowed_skills)
    validated_child_allowed = (
        get_global_skill_manager().validate_explicit_allowed_skills(
            child_allowed_skills
        )
    )
    if child_allowed_skills is None:
        effective_allowed_skills = (
            list(agent.config.allowed_skills)
            if agent.config.allowed_skills is not None
            else None
        )
    else:
        effective_allowed_skills = validated_child_allowed
        if (
            agent.config.allowed_skills is not None
            and effective_allowed_skills is not None
        ):
            parent_allowed = set(agent.config.allowed_skills)
            disallowed = [
                skill
                for skill in effective_allowed_skills
                if skill not in parent_allowed
            ]
            if disallowed:
                skill_list = ", ".join(disallowed)
                raise ValueError(
                    "child_allowed_skills must be a subset of the parent's "
                    f"allowed_skills: {skill_list}"
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
        allowed_skills=effective_allowed_skills,
        allowed_tools=effective_allowed_tools,
    )
    return ResolvedChildDefinition(
        config=child_config,
    )


__all__ = [
    "ResolvedAgentDefinition",
    "ResolvedChildDefinition",
    "build_agent_hooks",
    "resolve_agent_definition",
    "resolve_child_definition",
]
