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


def validate_child_subset(
    child: list[str] | tuple[str, ...] | None,
    parent: list[str] | tuple[str, ...] | None,
    label: str,
) -> None:
    """Validate that *child* allowlist is a subset of *parent*'s.

    Both ``None`` (unrestricted) and ``[]`` (empty) are accepted without
    error.  Only when both lists are non-None and child contains items
    absent from parent will a ``ValueError`` be raised.
    """
    if child is None or parent is None:
        return
    parent_set = set(parent)
    disallowed = [item for item in child if item not in parent_set]
    if disallowed:
        items = ", ".join(disallowed)
        raise ValueError(
            f"child {label} must be a subset of the parent's {label}: {items}"
        )


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
    parent_config = agent.config

    # --- allowed_tools ---
    if child_allowed_tools is None:
        effective_allowed_tools = parent_config.allowed_tools
    else:
        effective_allowed_tools = list(child_allowed_tools)
        validate_child_subset(
            effective_allowed_tools, parent_config.allowed_tools, "allowed_tools"
        )

    # --- allowed_skills ---
    validate_expanded_allowed_skills(child_allowed_skills)
    validated_child_allowed = (
        get_global_skill_manager().validate_explicit_allowed_skills(
            child_allowed_skills
        )
    )
    if child_allowed_skills is None:
        effective_allowed_skills = (
            list(parent_config.allowed_skills)
            if parent_config.allowed_skills is not None
            else None
        )
    else:
        effective_allowed_skills = validated_child_allowed
        validate_child_subset(
            effective_allowed_skills, parent_config.allowed_skills, "allowed_skills"
        )

    child_options = parent_config.options.model_copy(deep=True)
    child_options.enable_termination_summary = True

    child_config = AgentConfig(
        name=agent.name,
        description=agent.description,
        system_prompt=compose_child_system_prompt(
            base_prompt=parent_config.system_prompt,
            system_prompt_override=system_prompt_override,
            instruction=instruction,
        ),
        options=child_options,
        allowed_skills=effective_allowed_skills,
        allowed_tools=effective_allowed_tools,
    )
    return ResolvedChildDefinition(config=child_config)


__all__ = [
    "ResolvedAgentDefinition",
    "ResolvedChildDefinition",
    "build_agent_hooks",
    "resolve_agent_definition",
    "resolve_child_definition",
    "validate_child_subset",
]
