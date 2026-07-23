"""Child-agent execution mixin for Agent."""

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from agiwo.agent.definition import (
    ResolvedChildDefinition,
    build_agent_hooks,
    resolve_child_definition,
)
from agiwo.agent.models.execution import RunTreeRole
from agiwo.agent.models.input import UserInput
from agiwo.agent.models.run import RunIdentity, RunOutput
from agiwo.agent.prompt import build_system_prompt
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.tool.base import BaseTool
from agiwo.utils.abort_signal import AbortSignal
from agiwo.workspace import WorkspaceBootstrapper, WorkspaceDocumentStore

if TYPE_CHECKING:
    from agiwo.agent.agent import Agent


class AgentChildOps:
    """Mixin: spawn and run child agents."""

    async def run_child(
        self: "Agent",
        user_input: UserInput,
        *,
        session_runtime: SessionRuntime,
        parent_run_id: str,
        parent_depth: int,
        parent_user_id: str | None,
        parent_timeout_at: float | None,
        parent_metadata: dict[str, Any],
        instruction: str | None = None,
        system_prompt_override: str | None = None,
        child_allowed_tools: list[str] | None = None,
        child_allowed_skills: list[str] | None = None,
        metadata_overrides: dict[str, Any] | None = None,
        metadata_updates: dict | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> RunOutput:
        resolved_child: ResolvedChildDefinition = resolve_child_definition(
            parent_config=self._config,
            parent_extra_tools=self._extra_tools,
            parent_agent_id=self._id,
            instruction=instruction,
            system_prompt_override=system_prompt_override,
            child_allowed_tools=child_allowed_tools,
            child_allowed_skills=child_allowed_skills,
        )
        context = RunContext(
            identity=RunIdentity(
                run_id=str(uuid4()),
                agent_id=self._id,
                agent_name=self.name,
                user_id=parent_user_id,
                depth=parent_depth + 1,
                parent_run_id=parent_run_id,
                timeout_at=parent_timeout_at,
                run_tree_role=RunTreeRole.CHILD,
                metadata=dict(parent_metadata),
            ),
            session_runtime=session_runtime,
        )
        combined_metadata = dict(metadata_overrides or {})
        if metadata_updates:
            combined_metadata.update(metadata_updates)
        if combined_metadata:
            context.update_metadata(combined_metadata)
        child_abort_signal = abort_signal or session_runtime.abort_signal

        from agiwo.agent import agent as agent_module  # noqa: PLC0415

        return await agent_module.execute_run(
            user_input,
            context=context,
            model=self._model,
            system_prompt=await build_system_prompt(
                base_prompt=resolved_child.config.system_prompt,
                workspace=self._workspace,
                tools=resolved_child.extra_tools,
                allowed_skills=resolved_child.config.allowed_skills,
                bootstrapper=WorkspaceBootstrapper(),
                document_store=WorkspaceDocumentStore(),
            ),
            tools=resolved_child.extra_tools,
            hooks=build_agent_hooks(self._config, self._hooks),
            options=resolved_child.config.options.model_copy(deep=True),
            abort_signal=child_abort_signal,
            root_path=resolved_child.config.options.get_effective_root_path(),
        )

    async def create_child_agent(
        self: "Agent",
        *,
        child_id: str,
        instruction: str | None = None,
        system_prompt_override: str | None = None,
        child_allowed_tools: list[str] | None = None,
        child_allowed_skills: list[str] | None = None,
        extra_tools: list[BaseTool] | None = None,
        inherit_all_extra_tools: bool = False,
        system_tools: list[BaseTool] | None = None,
    ) -> "Agent":
        """Create a child Agent with inherited configuration.

        Parent's extra tools are inherited automatically (minus self-referencing
        AgentTool).  The *extra_tools* parameter adds caller-provided tools on
        top.

        When *inherit_all_extra_tools* is ``True`` (fork mode), the exclusion
        filter is skipped so that the child receives an identical tool set for
        LLM KV cache reuse.

        *system_tools* are injected unconditionally and not subject to
        ``allowed_tools`` filtering.
        """
        resolved_child: ResolvedChildDefinition = resolve_child_definition(
            parent_config=self._config,
            parent_extra_tools=self._extra_tools,
            parent_agent_id=self._id,
            instruction=instruction,
            system_prompt_override=system_prompt_override,
            child_allowed_tools=child_allowed_tools,
            child_allowed_skills=child_allowed_skills,
            extra_tools=extra_tools,
            inherit_all_extra_tools=inherit_all_extra_tools,
        )

        child = self.__class__(
            resolved_child.config,
            id=child_id,
            model=self.model,
            tools=resolved_child.extra_tools or None,
            hooks=build_agent_hooks(self._config, self._hooks),
        )
        if system_tools:
            child._inject_system_tools(system_tools)
        return child
