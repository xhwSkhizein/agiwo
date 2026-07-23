"""Resume and continuation mixin for Agent."""

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from agiwo.agent.execution_handle import AgentExecutionHandle
from agiwo.agent.models.input import UserInput
from agiwo.agent.models.log import RunResumePrepared, RunStarted
from agiwo.agent.models.run import RunOutput, RunStatus, RunView
from agiwo.agent.resume import build_resume_plan
from agiwo.agent.run_loop import execute_run
from agiwo.agent.storage.serialization import build_run_view_from_entries
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.runtime.state_ops import replace_messages
from agiwo.agent.models.execution import RunTreeRole
from agiwo.agent.models.run import RunIdentity
from agiwo.utils.abort_signal import AbortSignal

if TYPE_CHECKING:
    from agiwo.agent.agent import Agent


class AgentResumeOps:
    """Mixin: pause/resume and completed-run continuation."""

    async def prepare_resume(self: "Agent", *, run_id: str, session_id: str) -> str:
        """Validate a paused run and append RunResumePrepared. Returns checkpoint_id."""
        entries = await self._run_log_storage.list_entries(
            session_id=session_id, run_id=run_id, limit=100_000
        )
        view = build_run_view_from_entries(entries)
        if view is None or view.status is not RunStatus.PAUSED:
            raise ValueError(f"run {run_id!r} is not PAUSED")
        plan = build_resume_plan(entries)
        seq = await self._run_log_storage.allocate_sequence(session_id)
        prepared = RunResumePrepared(
            sequence=seq,
            session_id=session_id,
            run_id=run_id,
            agent_id=self._id,
            checkpoint_id=plan.checkpoint_id,
        )
        await self._run_log_storage.append_entries([prepared])
        return plan.checkpoint_id

    async def resume_paused_run(
        self: "Agent",
        *,
        run_id: str,
        session_id: str,
        abort_signal: AbortSignal | None = None,
    ) -> RunOutput:
        """Continue a PAUSED run with the same run_id (public API unchanged)."""
        self._ensure_open()
        entries = await self._run_log_storage.list_entries(
            session_id=session_id, run_id=run_id, limit=100_000
        )
        plan = build_resume_plan(entries)
        started_entry = next((e for e in entries if isinstance(e, RunStarted)), None)
        if started_entry is None:
            raise ValueError(f"run {run_id!r} missing RunStarted")

        resolved_abort = abort_signal or AbortSignal()
        session_runtime = SessionRuntime(
            session_id=session_id,
            run_log_storage=self._run_log_storage,
            abort_signal=resolved_abort,
        )
        context = RunContext(
            identity=RunIdentity(
                run_id=run_id,
                agent_id=self._id,
                agent_name=self.name,
                user_id=started_entry.user_id,
                run_tree_role=(
                    RunTreeRole(started_entry.run_tree_role)
                    if started_entry.run_tree_role
                    else RunTreeRole.NONE
                ),
            ),
            session_runtime=session_runtime,
        )
        replace_messages(context, plan.messages)
        if plan.continue_user_message is not None:
            msgs = context.snapshot_messages()
            msgs.append(
                {
                    "role": "user",
                    "content": plan.continue_user_message.extract_text(),
                    "is_user_provided": False,
                }
            )
            replace_messages(context, msgs)

        system_prompt = await self.get_effective_system_prompt()
        options = self._config.options.model_copy(deep=True)
        try:
            return await execute_run(
                plan.continue_user_message,
                context=context,
                model=self._model,
                system_prompt=system_prompt,
                tools=list(self._tools),
                hooks=self._hooks,
                options=options,
                abort_signal=resolved_abort,
                root_path=options.get_effective_root_path(),
                pending_tool_calls=plan.pending_tool_calls,
                resume=True,
                resume_checkpoint_id=plan.checkpoint_id,
            )
        finally:
            await session_runtime.close()

    async def _validate_completed_run(
        self: "Agent", *, run_id: str, session_id: str
    ) -> RunView:
        entries = await self._run_log_storage.list_entries(
            session_id=session_id,
            run_id=run_id,
            agent_id=self._id,
            limit=100_000,
        )
        view = build_run_view_from_entries(entries)
        if view is None:
            raise ValueError(f"unknown run_id={run_id!r}")
        if view.status is not RunStatus.COMPLETED:
            raise ValueError(
                f"run {run_id!r} status={view.status.value}; expected COMPLETED"
            )
        return view

    def continue_completed_run(
        self: "Agent",
        *,
        run_id: str,
        session_id: str,
        user_input: UserInput,
        abort_signal: AbortSignal | None = None,
        active_worker_ids: Callable[[], frozenset[str]] | None = None,
    ) -> AgentExecutionHandle:
        """Continue a completed Run with the same ``run_id`` (Wave D worker reports)."""
        self._ensure_open()
        resolved_abort_signal = abort_signal or AbortSignal()
        trace_runtime = self._start_trace_runtime(
            session_id=session_id,
            user_id=None,
            user_input=user_input,
        )
        session_runtime = SessionRuntime(
            session_id=session_id,
            run_log_storage=self._run_log_storage,
            trace_runtime=trace_runtime,
            abort_signal=resolved_abort_signal,
        )
        context = RunContext(
            identity=RunIdentity(
                run_id=run_id,
                agent_id=self._id,
                agent_name=self.name,
            ),
            session_runtime=session_runtime,
        )
        task = asyncio.create_task(
            self._execute_continuation(
                user_input,
                context=context,
                abort_signal=resolved_abort_signal,
                active_worker_ids=active_worker_ids,
            )
        )
        handle = AgentExecutionHandle(
            run_id=context.run_id,
            session_id=context.session_id,
            session_runtime=session_runtime,
            task=task,
            context=context,
        )
        self._register_execution(context.run_id, task, resolved_abort_signal)
        return handle

    async def _execute_continuation(
        self: "Agent",
        user_input: UserInput,
        *,
        context: RunContext,
        abort_signal: AbortSignal,
        active_worker_ids: Callable[[], frozenset[str]] | None = None,
    ) -> RunOutput:
        await self._validate_completed_run(
            run_id=context.run_id,
            session_id=context.session_id,
        )
        try:
            system_prompt = await self.get_effective_system_prompt()
            options = self._config.options.model_copy(deep=True)
        except Exception:
            await context.session_runtime.close()
            raise
        try:
            return await execute_run(
                user_input,
                context=context,
                model=self._model,
                system_prompt=system_prompt,
                tools=list(self._tools),
                hooks=self._hooks,
                options=options,
                abort_signal=abort_signal,
                root_path=options.get_effective_root_path(),
                continuation=True,
                active_worker_ids=active_worker_ids,
            )
        finally:
            await context.session_runtime.close()
