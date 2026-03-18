"""
Agent execution loop with a single step/sequence owner.
"""

import asyncio

from agiwo.agent.hooks import AgentHooks
from agiwo.agent.input import ChannelContext
from agiwo.agent.inner.compaction.runtime import CompactionRuntime
from agiwo.agent.inner.context import AgentRunContext
from agiwo.agent.inner.execution_bootstrap import prepare_execution
from agiwo.agent.inner.llm_handler import LLMStreamHandler
from agiwo.agent.inner.run_recorder import RunRecorder
from agiwo.agent.inner.run_state import RunState
from agiwo.agent.inner.steering import apply_steering_messages
from agiwo.agent.inner.termination_runtime import ExecutionTerminationRuntime
from agiwo.agent.inner.tool_runtime import ResolvedToolCall, ToolRuntime
from agiwo.agent.memory_types import MemoryRecord
from agiwo.agent.options import AgentOptions
from agiwo.agent.runtime import (
    LLMCallContext,
    RunOutput,
    StepRecord,
    TerminationReason,
)
from agiwo.agent.runtime_tools.contracts import RuntimeToolOutcome
from agiwo.tool.base import ToolResult
from agiwo.agent.runtime_tools import RuntimeToolLike
from agiwo.config.settings import settings
from agiwo.llm.base import Model
from agiwo.llm.limits import (
    resolve_max_context_window,
    resolve_max_input_tokens_per_call,
)
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class AgentExecutor:
    def __init__(
        self,
        *,
        model: Model,
        tools: list[RuntimeToolLike],
        options: AgentOptions | None = None,
        hooks: AgentHooks | None = None,
        run_recorder: RunRecorder,
        root_path: str | None = None,
    ) -> None:
        self.model = model
        self.options = options or AgentOptions()
        self.hooks = hooks or AgentHooks()
        self._base_run_recorder = run_recorder
        self.root_path = root_path or settings.root_path
        self.max_context_window = resolve_max_context_window(model)
        self.max_input_tokens_per_call = resolve_max_input_tokens_per_call(
            self.options.max_input_tokens_per_call,
            model,
        )

        self.llm_handler = LLMStreamHandler(model)
        self.tool_runtime = ToolRuntime(tools=tools)
        self.termination_runtime = ExecutionTerminationRuntime(
            options=self.options,
            max_input_tokens_per_call=self.max_input_tokens_per_call,
            llm_handler=self.llm_handler,
        )
        self._tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": tool.get_definition().name,
                    "description": tool.get_definition().description,
                    "parameters": tool.get_definition().parameters,
                },
            }
            for tool in tools
        ] or None

    async def execute(
        self,
        *,
        system_prompt: str,
        user_step: StepRecord,
        context: AgentRunContext,
        memories: list[MemoryRecord] | None = None,
        before_run_hook_result: str | None = None,
        channel_context: ChannelContext | None = None,
        pending_tool_calls: list[dict] | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> RunOutput:
        prepared = await prepare_execution(
            system_prompt=system_prompt,
            user_step=user_step,
            context=context,
            memories=memories,
            before_run_hook_result=before_run_hook_result,
            channel_context=channel_context,
            options=self.options,
            run_recorder=self._base_run_recorder,
            tool_schemas=self._tool_schemas,
            llm_handler=self.llm_handler,
            compact_prompt=self.options.compact_prompt,
            root_path=self.root_path,
        )
        state = prepared.state
        run_recorder = prepared.run_recorder
        compactor = prepared.compactor

        try:
            await self._run_loop(
                state,
                run_recorder,
                compactor,
                pending_tool_calls,
                abort_signal,
            )
        except asyncio.CancelledError:
            state.termination_reason = TerminationReason.CANCELLED
            logger.info("agent_execution_cancelled", run_id=state.context.run_id)
        except Exception as error:
            state.termination_reason = (
                TerminationReason.ERROR_WITH_CONTEXT
                if state.assistant_steps_count > 0
                else TerminationReason.ERROR
            )
            logger.error(
                "agent_execution_failed",
                run_id=state.context.run_id,
                error=str(error),
                error_type=type(error).__name__,
                steps_completed=state.steps_count,
                termination_reason=state.termination_reason,
                exc_info=True,
            )
        finally:
            await self.termination_runtime.maybe_generate_summary(
                state=state,
                run_recorder=run_recorder,
                abort_signal=abort_signal,
            )

        return state.build_output()

    async def _run_loop(
        self,
        state: RunState,
        run_recorder: RunRecorder,
        compactor: CompactionRuntime,
        pending_tool_calls: list[dict] | None,
        abort_signal: AbortSignal | None,
    ) -> None:
        if await self._resume_pending_tool_calls(
            state,
            run_recorder,
            pending_tool_calls,
            abort_signal,
        ):
            return

        while state.termination_reason is None:
            if not await self._run_cycle(
                state,
                run_recorder,
                compactor,
                abort_signal,
            ):
                return

    async def _resume_pending_tool_calls(
        self,
        state: RunState,
        run_recorder: RunRecorder,
        pending_tool_calls: list[dict] | None,
        abort_signal: AbortSignal | None,
    ) -> bool:
        if not pending_tool_calls:
            return False
        state.termination_reason = await self._execute_tool_calls(
            state=state,
            run_recorder=run_recorder,
            tool_calls=pending_tool_calls,
            abort_signal=abort_signal,
        )
        return state.termination_reason is not None

    async def _run_cycle(
        self,
        state: RunState,
        run_recorder: RunRecorder,
        compactor: CompactionRuntime,
        abort_signal: AbortSignal | None,
    ) -> bool:
        reason = self.termination_runtime.check_non_recoverable_limits(state)
        if reason is not None:
            state.termination_reason = reason
            return False

        if await self._handle_compaction(state, run_recorder, compactor, abort_signal):
            return state.termination_reason is None

        step, llm_context = await self._stream_next_step(
            state,
            run_recorder,
            abort_signal,
        )
        reason = self.termination_runtime.check_post_llm_limits(
            state,
            step,
            llm_context,
        )
        if reason is not None:
            state.termination_reason = reason
            return False
        return await self._handle_step_tools(
            state,
            run_recorder,
            step,
            abort_signal,
        )

    async def _stream_next_step(
        self,
        state: RunState,
        run_recorder: RunRecorder,
        abort_signal: AbortSignal | None,
    ) -> tuple[StepRecord, LLMCallContext]:
        apply_steering_messages(state.messages, state.context.steering_queue)

        state.current_step += 1
        if self.hooks.on_before_llm_call:
            modified = await self.hooks.on_before_llm_call(state.messages)
            if modified is not None:
                state.messages = modified

        step, llm_context = await self.llm_handler.stream_assistant_step(
            state,
            run_recorder,
            abort_signal,
        )
        await run_recorder.commit_step(step, llm=llm_context)

        if self.hooks.on_after_llm_call:
            await self.hooks.on_after_llm_call(step)
        return step, llm_context

    async def _handle_step_tools(
        self,
        state: RunState,
        run_recorder: RunRecorder,
        step: StepRecord,
        abort_signal: AbortSignal | None,
    ) -> bool:
        if step.tool_calls:
            state.termination_reason = await self._execute_tool_calls(
                state=state,
                run_recorder=run_recorder,
                tool_calls=step.tool_calls,
                abort_signal=abort_signal,
            )
            return state.termination_reason is None

        state.termination_reason = TerminationReason.COMPLETED
        return False

    async def _execute_tool_calls(
        self,
        *,
        state: RunState,
        run_recorder: RunRecorder,
        tool_calls: list[dict],
        abort_signal: AbortSignal | None,
    ) -> TerminationReason | None:
        prepared_calls = await self._prepare_tool_calls(tool_calls)
        outcomes: list[
            RuntimeToolOutcome
        ] = await self.tool_runtime.execute_resolved_batch(
            prepared_calls,
            context=state.context,
            abort_signal=abort_signal,
        )
        return await self._record_tool_outcomes(
            run_recorder=run_recorder,
            prepared_calls=prepared_calls,
            outcomes=outcomes,
        )

    async def _prepare_tool_calls(
        self,
        tool_calls: list[dict],
    ) -> list[ResolvedToolCall | ToolResult]:
        prepared_calls: list[ResolvedToolCall | ToolResult] = []
        for tool_call in tool_calls:
            resolved = self.tool_runtime.resolve_tool_call(tool_call)
            if isinstance(resolved, ToolResult):
                prepared_calls.append(resolved)
                continue
            if self.hooks.on_before_tool_call:
                modified = await self.hooks.on_before_tool_call(
                    resolved.call_id,
                    resolved.tool_name,
                    dict(resolved.args),
                )
                if modified is not None:
                    resolved = resolved.with_args(modified)
            prepared_calls.append(resolved)
        return prepared_calls

    async def _record_tool_outcomes(
        self,
        *,
        run_recorder: RunRecorder,
        prepared_calls: list[ResolvedToolCall | ToolResult],
        outcomes: list[RuntimeToolOutcome],
    ) -> TerminationReason | None:
        for prepared, outcome in zip(prepared_calls, outcomes):
            result = outcome.result
            call_id, tool_name, args = _resolve_tool_result_context(prepared, result)

            if self.hooks.on_after_tool_call:
                await self.hooks.on_after_tool_call(
                    call_id,
                    tool_name,
                    args,
                    result,
                )

            step = await run_recorder.create_tool_step(
                tool_call_id=call_id,
                name=result.tool_name,
                content=result.content or "",
                content_for_user=result.content_for_user,
            )
            await run_recorder.commit_step(step)

        for outcome in outcomes:
            if outcome.termination_reason is not None:
                return outcome.termination_reason
        return None

    async def _handle_compaction(
        self,
        state: RunState,
        run_recorder: RunRecorder,
        compactor: CompactionRuntime,
        abort_signal: AbortSignal | None,
    ) -> bool:
        result = await compactor.compact_if_needed(
            state=state,
            run_recorder=run_recorder,
            abort_signal=abort_signal,
            max_context_window=self.max_context_window,
        )
        if result is None:
            return False
        state.messages = result.compacted_messages

        logger.info(
            "compact_triggered",
            run_id=state.context.run_id,
            before_messages=len(state.messages),
        )
        if (
            self.options.max_run_cost is not None
            and state.token_cost >= self.options.max_run_cost
        ):
            state.termination_reason = TerminationReason.MAX_RUN_COST
        return True


def _resolve_tool_result_context(
    prepared: ResolvedToolCall | ToolResult,
    result: ToolResult,
) -> tuple[str, str, dict]:
    if isinstance(prepared, ResolvedToolCall):
        return prepared.call_id, prepared.tool_name, prepared.args
    return (
        result.tool_call_id or "",
        result.tool_name,
        result.input_args or {},
    )


__all__ = ["AgentExecutor"]
