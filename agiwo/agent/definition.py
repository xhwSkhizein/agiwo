"""Definition-scoped assembly helpers for Agent."""

from dataclasses import dataclass

from agiwo.agent.models.config import AgentConfig
from agiwo.agent.hooks import (
    DefaultMemoryHook,
    HookGroup,
    HookPhase,
    HookRegistration,
    HookRegistry,
    transform,
)
from agiwo.agent.prompt import compose_child_system_prompt
from agiwo.skill.allowlist import validate_expanded_allowed_skills
from agiwo.skill.manager import get_global_skill_manager
from agiwo.tool.base import BaseTool
from agiwo.workspace import AgentWorkspace, build_agent_workspace


def validate_child_subset(
    child: list[str] | tuple[str, ...] | None,
    parent: list[str] | tuple[str, ...] | None,
    label: str,
) -> None:
    """Validate that *child* allowlist is a subset of *parent*'s.

    Both ``None`` (unrestricted) and ``[]`` (empty) are accepted without
    error.  Only when both lists are non-None and child contains items
    absent from parent will a ``ValueError`` be raised.

    For ``allowed_skills``, ``parent=None`` is NOT treated as unrestricted:
    if the parent has no restrictions, the child cannot opt into an explicit
    skills list (child must also be ``None``).
    """
    if label == "allowed_skills" and parent is None and child is not None:
        raise ValueError(
            "child allowed_skills must be None when parent has no restrictions (None)"
        )

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
    hooks: HookRegistry
    workspace: AgentWorkspace


@dataclass(frozen=True)
class ResolvedChildDefinition:
    config: AgentConfig
    extra_tools: list[BaseTool]  # Resolved extra tools for child agent


def build_agent_hooks(
    config: AgentConfig,
    hooks: HookRegistry | list[HookRegistration] | None,
) -> HookRegistry:
    if isinstance(hooks, HookRegistry):
        resolved = HookRegistry(list(hooks.registrations))
    else:
        resolved = HookRegistry(list(hooks or []))
    if not resolved.has_phase(HookPhase.ASSEMBLE_CONTEXT):
        memory_hook = DefaultMemoryHook(
            root_path=config.options.get_effective_root_path()
        )

        async def _memory_retrieve(payload: dict) -> dict:
            memories = await memory_hook.retrieve_memories(
                payload["user_input"], payload["context"]
            )
            updated = dict(payload)
            updated["memories"] = memories
            return updated

        resolved.add(
            transform(
                HookPhase.ASSEMBLE_CONTEXT,
                "default_memory_retrieve",
                _memory_retrieve,
                group=HookGroup.SYSTEM,
            )
        )
    return resolved


def resolve_agent_definition(
    *,
    config: AgentConfig,
    agent_id: str,
    hooks: HookRegistry | list[HookRegistration] | None,
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


def _resolve_child_extra_tools(
    parent_extra_tools: tuple[BaseTool, ...],
    parent_agent_id: str,
    *,
    extra_tools: list[BaseTool] | None,
    inherit_all_extra_tools: bool,
) -> list[BaseTool]:
    """Resolve which extra tools a child agent should inherit.

    Args:
        parent_extra_tools: Parent agent's extra tools
        parent_agent_id: Parent agent's ID (for filtering self-referencing AgentTool)
        extra_tools: Additional tools provided by caller
        inherit_all_extra_tools: If True, inherit all parent's extra_tools;
            otherwise filter out non-inheritable tools (e.g., self-referencing AgentTool)

    Returns:
        Merged list of extra tools (parent's inheritable tools + caller's tools),
        with caller's tools taking precedence on name conflicts.
    """
    if inherit_all_extra_tools:
        parent_extra = list(parent_extra_tools)
    else:
        parent_extra = [
            t
            for t in parent_extra_tools
            if not (
                hasattr(t, "is_inheritable_by")
                and not t.is_inheritable_by(parent_agent_id)
            )
        ]

    # Merge with deduplication while preserving order (parent first, then caller)
    seen: set[str] = set()
    merged: list[BaseTool] = []
    # First pass: parent tools (excluding those overridden by caller)
    caller_tool_names = {t.name for t in extra_tools or []}
    for t in parent_extra:
        if t.name not in caller_tool_names and t.name not in seen:
            seen.add(t.name)
            merged.append(t)
    # Second pass: caller tools (take precedence over parent)
    for t in extra_tools or []:
        if t.name not in seen:
            seen.add(t.name)
            merged.append(t)
    return merged


def resolve_child_definition(
    parent_config: AgentConfig,
    parent_extra_tools: tuple[BaseTool, ...],
    parent_agent_id: str,
    *,
    instruction: str | None,
    system_prompt_override: str | None,
    child_allowed_skills: list[str] | None = None,
    child_allowed_tools: list[str] | None = None,
    extra_tools: list[BaseTool] | None = None,
    inherit_all_extra_tools: bool = False,
) -> ResolvedChildDefinition:
    """Resolve child agent definition, computing config and extra tools.

    This function handles both configuration resolution and extra tools inheritance,
    providing a single source of truth for child agent assembly.

    Args:
        parent_config: Parent agent's configuration
        parent_extra_tools: Parent agent's extra tools
        parent_agent_id: Parent agent's ID (for filtering self-referencing AgentTool)
        instruction: Optional instruction for the child agent
        system_prompt_override: Optional system prompt override
        child_allowed_skills: Optional list of allowed skills for child
        child_allowed_tools: Optional list of allowed tools for child
        extra_tools: Additional tools provided by caller
        inherit_all_extra_tools: If True, inherit all parent's extra_tools

    Returns:
        ResolvedChildDefinition with child config and resolved extra tools
    """

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
        name=parent_config.name,
        description=parent_config.description,
        system_prompt=compose_child_system_prompt(
            base_prompt=parent_config.system_prompt,
            system_prompt_override=system_prompt_override,
            instruction=instruction,
        ),
        options=child_options,
        allowed_skills=effective_allowed_skills,
        allowed_tools=effective_allowed_tools,
    )

    # Resolve extra tools inheritance
    resolved_extra_tools = _resolve_child_extra_tools(
        parent_extra_tools,
        parent_agent_id,
        extra_tools=extra_tools,
        inherit_all_extra_tools=inherit_all_extra_tools,
    )

    return ResolvedChildDefinition(
        config=child_config, extra_tools=resolved_extra_tools
    )


__all__ = [
    "ResolvedAgentDefinition",
    "ResolvedChildDefinition",
    "build_agent_hooks",
    "resolve_agent_definition",
    "resolve_child_definition",
    "validate_child_subset",
]
