"""Single-run execution engine for agent workflows."""

import asyncio
import time
from dataclasses import dataclass

from agiwo.agent.engine.compaction.runtime import CompactionRuntime
from agiwo.agent.engine.context import AgentRunContext
from agiwo.agent.engine.llm_handler import LLMStreamHandler
from agiwo.agent.engine.message_assembler import MessageAssembler
from agiwo.agent.engine.recorder import RunRecorder
from agiwo.agent.engine.state import RunState
from agiwo.agent.engine.steering import apply_steering_messages
from agiwo.agent.engine.termination import ExecutionTerminationRuntime
from agiwo.agent.engine.tool_runtime import ResolvedToolCall, ToolRuntime
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.input import UserInput
from agiwo.agent.input_codec import normalize_to_message
from agiwo.agent.lifecycle.definition import ResolvedExecutionDefinition
from agiwo.agent.memory_types import MemoryRecord
from agiwo.agent.options import AgentOptions
from agiwo.agent.runtime import (
    LLMCallContext,
    Run,
    RunOutput,
    RunStatus,
    StepRecord,
    TerminationReason,
)
from agiwo.agent.runtime_tools import RuntimeToolLike
from agiwo.agent.runtime_tools.contracts import RuntimeToolOutcome
from agiwo.agent.scheduler_port import StepObserver
from agiwo.config.settings import settings
from agiwo.llm.base import Model
from agiwo.llm.limits import (
    resolve_max_context_window,
    resolve_max_input_tokens_per_call,
)
from agiwo.tool.base import ToolResult
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PreparedExecution:
    state: RunState
    run_recorder: RunRecorder
    compactor: CompactionRuntime
    user_step: StepRecord


class ExecutionEngine:
    """Own the full execution pipeline for a single run."""

    def __init__(
        self,
        *,
        definition: ResolvedExecutionDefinition,
        step_observers: tuple[StepObserver, ...] = (),
        root_path: str | None = None,
    ) -> None:
        self._definition = definition
        self.model: Model = definition.model
        self.options: AgentOptions = definition.options or AgentOptions()
        self.hooks: AgentHooks = definition.hooks or AgentHooks()
        self._step_observers = step_observers
        self.root_path = root_path or settings.root_path
        self.max_context_window = resolve_max_context_window(self.model)
        self.max_input_tokens_per_call = resolve_max_input_tokens_per_call(
            self.options.max_input_tokens_per_call,
            self.model,
        )

        self.llm_handler = LLMStreamHandler(self.model)
        self.tool_runtime = ToolRuntime(tools=list(definition.tools))
        self.termination_runtime = ExecutionTerminationRuntime(
            options=self.options,
            max_input_tokens_per_call=self.max_input_tokens_per_call,
            llm_handler=self.llm_handler,
        )
        self._tool_schemas = self._build_tool_schemas(definition.tools)

    async def execute(
        self,
        user_input: UserInput,
        *,
        context: AgentRunContext,
        pending_tool_calls: list[dict] | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> RunOutput:
        before_run_hook_result, memories = await self._before_run(
            user_input=user_input,
            context=context,
        )
        prepared = await self._prepare_state(
            user_input=user_input,
            context=context,
            memories=memories,
            before_run_hook_result=before_run_hook_result,
        )
        state = prepared.state
        run_recorder = prepared.run_recorder
        run = self._create_run(user_input, context)

        await run_recorder.start_run(run)

        try:
            await run_recorder.commit_step(
                prepared.user_step,
                append_message=False,
                track_state=False,
            )
            await self._run_loop(
                state=state,
                run_recorder=run_recorder,
                compactor=prepared.compactor,
                pending_tool_calls=pending_tool_calls,
                abort_signal=abort_signal,
            )
            await self.termination_runtime.maybe_generate_summary(
                state=state,
                run_recorder=run_recorder,
                abort_signal=abort_signal,
            )
            result = state.build_output()
            await self._after_run(
                user_input=user_input,
                result=result,
                context=context,
            )
            await run_recorder.complete_run(result)
            return result
        except Exception as error:
            await run_recorder.fail_run(error)
            raise

    async def _prepare_state(
        self,
        *,
        user_input: UserInput,
        context: AgentRunContext,
        memories: list[MemoryRecord] | None,
        before_run_hook_result: str | None,
    ) -> PreparedExecution:
        session_storage = context.session_runtime.session_storage
        run_step_storage = context.session_runtime.run_step_storage
        user_step = await self._create_user_step(context, user_input)

        last_compact = await session_storage.get_latest_compact_metadata(
            context.session_id,
            context.agent_id,
        )
        compact_start_seq = last_compact.end_seq + 1 if last_compact is not None else 0

        existing_steps = await run_step_storage.get_steps(
            session_id=context.session_id,
            agent_id=context.agent_id,
            start_seq=compact_start_seq if compact_start_seq > 0 else None,
        )
        if all(step.id != user_step.id for step in existing_steps):
            existing_steps.append(user_step)
            existing_steps.sort(key=lambda step: step.sequence)

        user_message = normalize_to_message(user_input)
        messages = MessageAssembler.assemble(
            self._definition.system_prompt,
            existing_steps,
            memories,
            before_run_hook_result,
            channel_context=user_message.context,
        )
        state = RunState(
            context=context,
            config=self.options,
            messages=messages,
            tool_schemas=self._tool_schemas,
            last_compact_metadata=last_compact,
            compact_start_seq=compact_start_seq,
        )
        run_recorder = RunRecorder(
            context=context,
            hooks=self.hooks,
            step_observers=self._step_observers,
            state=state,
        )
        compactor = CompactionRuntime(
            llm_handler=self.llm_handler,
            session_storage=session_storage,
            compact_prompt=self.options.compact_prompt,
            root_path=self.root_path,
        )
        return PreparedExecution(
            state=state,
            run_recorder=run_recorder,
            compactor=compactor,
            user_step=user_step,
        )

    async def _before_run(
        self,
        *,
        user_input: UserInput,
        context: AgentRunContext,
    ) -> tuple[str | None, list[MemoryRecord]]:
        before_run_hook_result = None
        if self.hooks.on_before_run is not None:
            before_run_hook_result = await self.hooks.on_before_run(user_input, context)

        memories: list[MemoryRecord] = []
        if self.hooks.on_memory_retrieve is not None and user_input is not None:
            memories = await self.hooks.on_memory_retrieve(user_input, context)
        return before_run_hook_result, memories

    async def _run_loop(
        self,
        *,
        state: RunState,
        run_recorder: RunRecorder,
        compactor: CompactionRuntime,
        pending_tool_calls: list[dict] | None,
        abort_signal: AbortSignal | None,
    ) -> None:
        try:
            if await self._resume_pending_tools_if_any(
                state=state,
                run_recorder=run_recorder,
                pending_tool_calls=pending_tool_calls,
                abort_signal=abort_signal,
            ):
                return

            while not state.is_terminal:
                if self._handle_pre_llm_limits(state):
                    return
                if await self._compact_if_needed(
                    state=state,
                    run_recorder=run_recorder,
                    compactor=compactor,
                    abort_signal=abort_signal,
                ):
                    if state.is_terminal:
                        return
                    continue
                step, llm_context = await self._llm_step(
                    state=state,
                    run_recorder=run_recorder,
                    abort_signal=abort_signal,
                )
                if self._handle_post_llm_limits(state, step, llm_context):
                    return
                if not await self._run_tools_if_needed(
                    state=state,
                    run_recorder=run_recorder,
                    step=step,
                    abort_signal=abort_signal,
                ):
                    return
        except asyncio.CancelledError:
            state.terminate(TerminationReason.CANCELLED)
            logger.info("agent_execution_cancelled", run_id=state.context.run_id)
        except Exception:
            state.record_failure_context()
            logger.error(
                "agent_execution_failed",
                run_id=state.context.run_id,
                steps_completed=state.steps_count,
                termination_reason=state.termination_reason,
                exc_info=True,
            )

    async def _resume_pending_tools_if_any(
        self,
        *,
        state: RunState,
        run_recorder: RunRecorder,
        pending_tool_calls: list[dict] | None,
        abort_signal: AbortSignal | None,
    ) -> bool:
        if not pending_tool_calls:
            return False
        reason = await self._execute_tool_calls(
            state=state,
            run_recorder=run_recorder,
            tool_calls=pending_tool_calls,
            abort_signal=abort_signal,
        )
        if reason is not None:
            state.terminate(reason)
            return True
        return state.is_terminal

    def _handle_pre_llm_limits(self, state: RunState) -> bool:
        reason = self.termination_runtime.check_non_recoverable_limits(state)
        if reason is None:
            return False
        state.terminate(reason)
        return True

    async def _compact_if_needed(
        self,
        *,
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

        logger.info(
            "compact_triggered",
            run_id=state.context.run_id,
            before_messages=len(state.messages),
        )
        if (
            self.options.max_run_cost is not None
            and state.token_cost >= self.options.max_run_cost
        ):
            state.terminate(TerminationReason.MAX_RUN_COST)
        return True

    async def _llm_step(
        self,
        *,
        state: RunState,
        run_recorder: RunRecorder,
        abort_signal: AbortSignal | None,
    ) -> tuple[StepRecord, LLMCallContext]:
        apply_steering_messages(state.messages, state.context.steering_queue)

        state.advance_step()
        if self.hooks.on_before_llm_call:
            modified = await self.hooks.on_before_llm_call(state.messages)
            if modified is not None:
                state.set_messages(modified)

        step, llm_context = await self.llm_handler.stream_assistant_step(
            state,
            run_recorder,
            abort_signal,
        )
        await run_recorder.commit_step(step, llm=llm_context)

        if self.hooks.on_after_llm_call:
            await self.hooks.on_after_llm_call(step)
        return step, llm_context

    def _handle_post_llm_limits(
        self,
        state: RunState,
        step: StepRecord,
        llm_context: LLMCallContext,
    ) -> bool:
        reason = self.termination_runtime.check_post_llm_limits(
            state,
            step,
            llm_context,
        )
        if reason is None:
            return False
        state.terminate(reason)
        return True

    async def _run_tools_if_needed(
        self,
        *,
        state: RunState,
        run_recorder: RunRecorder,
        step: StepRecord,
        abort_signal: AbortSignal | None,
    ) -> bool:
        if step.tool_calls:
            reason = await self._execute_tool_calls(
                state=state,
                run_recorder=run_recorder,
                tool_calls=step.tool_calls,
                abort_signal=abort_signal,
            )
            if reason is not None:
                state.terminate(reason)
            return not state.is_terminal

        state.terminate(TerminationReason.COMPLETED)
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
        outcomes = await self.tool_runtime.execute_resolved_batch(
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

    async def _after_run(
        self,
        *,
        user_input: UserInput,
        result: RunOutput,
        context: AgentRunContext,
    ) -> None:
        if self.hooks.on_after_run:
            await self.hooks.on_after_run(result, context)
        if self.hooks.on_memory_write and result.response is not None:
            await self.hooks.on_memory_write(user_input, result, context)

    @staticmethod
    async def _create_user_step(
        context: AgentRunContext,
        user_input: UserInput,
    ) -> StepRecord:
        return StepRecord.user(
            context,
            sequence=await context.next_sequence(),
            user_input=user_input,
        )

    @staticmethod
    def _create_run(user_input: UserInput, context: AgentRunContext) -> Run:
        run = Run(
            id=context.run_id,
            agent_id=context.agent_id,
            session_id=context.session_id,
            user_input=user_input,
            status=RunStatus.RUNNING,
            parent_run_id=context.parent_run_id,
        )
        run.metrics.start_at = time.time()
        return run

    @staticmethod
    def _build_tool_schemas(
        tools: tuple[RuntimeToolLike, ...],
    ) -> list[dict] | None:
        return [
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


__all__ = ["ExecutionEngine"]
