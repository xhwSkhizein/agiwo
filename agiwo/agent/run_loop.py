"""Single-run execution engine — the core run loop."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from agiwo.agent.compaction import compact_if_needed
from agiwo.agent.models.config import AgentOptions
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.models.input import UserInput, UserMessage
from agiwo.agent.llm_caller import stream_assistant_step
from agiwo.agent.models.memory import MemoryRecord
from agiwo.agent.prompt import apply_steering_messages, assemble_run_messages
from agiwo.agent.models.run import (
    Run,
    RunMetrics,
    RunOutput,
    RunStatus,
    TerminationReason,
)
from agiwo.agent.models.step import (
    LLMCallContext,
    StepRecord,
)
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.state_ops import (
    record_compaction_metadata,
    replace_messages,
    set_tool_schemas,
    set_termination_reason,
)
from agiwo.agent.runtime.step_committer import commit_step
from agiwo.agent.models.stream import (
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
)
from agiwo.agent.termination.limits import (
    check_non_recoverable_limits,
    check_post_llm_limits,
)
from agiwo.agent.termination.summarizer import maybe_generate_termination_summary
from agiwo.agent.tool_executor import execute_tool_batch
from agiwo.tool.base import BaseTool
from agiwo.config.settings import settings
from agiwo.llm.base import Model
from agiwo.llm.limits import (
    resolve_max_context_window,
    resolve_max_input_tokens_per_call,
)
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


async def _start_run(state: RunContext, run: Run) -> None:
    run.trace_id = state.trace_id
    await state.session_runtime.run_step_storage.save_run(run)
    if state.session_runtime.trace_runtime is not None:
        state.session_runtime.trace_runtime.on_run_started(run)
    await state.session_runtime.publish(RunStartedEvent.from_context(state))


async def _complete_run(state: RunContext, run: Run, result: RunOutput) -> None:
    run.status = (
        RunStatus.CANCELLED
        if result.termination_reason == TerminationReason.CANCELLED
        else RunStatus.COMPLETED
    )
    run.response_content = result.response
    run.updated_at = datetime.now()
    run.metrics.end_at = datetime.now(timezone.utc).timestamp()
    if result.metrics is not None:
        for field_name in (
            "duration_ms",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_creation_tokens",
            "token_cost",
            "steps_count",
            "tool_calls_count",
            "tool_errors_count",
            "first_token_latency",
            "response_latency",
        ):
            setattr(run.metrics, field_name, getattr(result.metrics, field_name))
    await state.session_runtime.run_step_storage.save_run(run)
    if state.session_runtime.trace_runtime is not None:
        await state.session_runtime.trace_runtime.on_run_completed(
            result,
            run_id=run.id,
        )
    await state.session_runtime.publish(
        RunCompletedEvent.from_context(
            state,
            response=result.response,
            metrics=result.metrics,
            termination_reason=result.termination_reason,
        ),
    )


async def _fail_run(state: RunContext, run: Run, error: Exception) -> None:
    run.status = RunStatus.FAILED
    run.updated_at = datetime.now()
    run.metrics.end_at = datetime.now(timezone.utc).timestamp()
    await state.session_runtime.run_step_storage.save_run(run)
    if state.session_runtime.trace_runtime is not None:
        await state.session_runtime.trace_runtime.on_run_failed(
            error,
            run_id=run.id,
        )
    await state.session_runtime.publish(
        RunFailedEvent.from_context(state, error=str(error)),
    )


def _build_output(state: RunContext) -> RunOutput:
    return RunOutput(
        response=state.response_content,
        run_id=state.run_id,
        session_id=state.session_id,
        metrics=RunMetrics(
            duration_ms=state.elapsed * 1000,
            total_tokens=state.total_tokens,
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            cache_read_tokens=state.cache_read_tokens,
            cache_creation_tokens=state.cache_creation_tokens,
            token_cost=state.token_cost,
            steps_count=state.steps_count,
            tool_calls_count=state.tool_calls_count,
        ),
        termination_reason=state.termination_reason,
    )


async def execute_run(
    user_input: UserInput,
    *,
    context: RunContext,
    system_prompt: str,
    model: Model,
    tools: tuple[BaseTool, ...],
    options: AgentOptions | None = None,
    hooks: AgentHooks | None = None,
    pending_tool_calls: list[dict] | None = None,
    abort_signal: AbortSignal | None = None,
    root_path: str | None = None,
) -> RunOutput:
    """Execute a single agent run — the core entry point."""
    options = options or AgentOptions()
    hooks = hooks or AgentHooks()
    context.config = options
    context.hooks = hooks

    tools_map = {tool.get_name(): tool for tool in tools}
    max_input_tokens_per_call = resolve_max_input_tokens_per_call(
        options.max_input_tokens_per_call,
        model,
    )
    tool_schemas = [
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
    max_context_window = resolve_max_context_window(model)

    before_run_hook_result = None
    if hooks.on_before_run is not None:
        before_run_hook_result = await hooks.on_before_run(user_input, context)
    memories: list[MemoryRecord] = []
    if hooks.on_memory_retrieve is not None and user_input is not None:
        memories = await hooks.on_memory_retrieve(user_input, context)

    session_storage = context.session_runtime.session_storage
    run_step_storage = context.session_runtime.run_step_storage
    user_step = StepRecord.user(
        context,
        sequence=await context.session_runtime.allocate_sequence(),
        user_input=user_input,
    )
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
    user_message = UserMessage.from_value(user_input)
    replace_messages(
        context,
        assemble_run_messages(
            system_prompt,
            existing_steps,
            memories,
            before_run_hook_result,
            channel_context=user_message.context,
        ),
    )
    set_tool_schemas(context, tool_schemas)
    record_compaction_metadata(context, last_compact)
    run = Run(
        id=context.run_id,
        agent_id=context.agent_id,
        session_id=context.session_id,
        user_input=user_input,
        status=RunStatus.RUNNING,
        parent_run_id=context.parent_run_id,
    )
    run.metrics.start_at = time.time()
    await _start_run(context, run)
    try:
        await commit_step(context, user_step, append_message=False, track_state=False)
        await _run_loop(
            state=context,
            model=model,
            tools_map=tools_map,
            options=options,
            max_input_tokens_per_call=max_input_tokens_per_call,
            compact_prompt=options.compact_prompt,
            max_context_window=max_context_window,
            compact_start_seq=compact_start_seq,
            pending_tool_calls=pending_tool_calls,
            abort_signal=abort_signal,
            root_path=root_path or settings.root_path,
        )
        await maybe_generate_termination_summary(
            state=context,
            options=options,
            model=model,
            abort_signal=abort_signal,
        )
        result = _build_output(context)
        if hooks.on_after_run:
            await hooks.on_after_run(result, context)
        if hooks.on_memory_write and result.response is not None:
            await hooks.on_memory_write(user_input, result, context)
        await _complete_run(context, run, result)
        return result
    except Exception as error:
        await _fail_run(context, run, error)
        raise


async def _execute_tool_calls(
    *,
    state: RunContext,
    tool_calls: list[dict[str, Any]],
    tools_map: dict[str, BaseTool],
    abort_signal: AbortSignal | None,
) -> bool:
    tool_results = await execute_tool_batch(
        tool_calls,
        tools_map=tools_map,
        context=state,
        abort_signal=abort_signal,
    )
    for result in tool_results:
        call_id = result.tool_call_id or ""
        tool_name = result.tool_name
        args = result.input_args or {}

        if state.hooks.on_after_tool_call:
            await state.hooks.on_after_tool_call(call_id, tool_name, args, result)

        tool_step = StepRecord.tool(
            state,
            sequence=await state.session_runtime.allocate_sequence(),
            tool_call_id=call_id,
            name=result.tool_name,
            content=result.content or "",
            content_for_user=result.content_for_user,
        )
        await commit_step(state, tool_step)

        if result.termination_reason is not None:
            set_termination_reason(state, result.termination_reason)
            return True
    return state.is_terminal


async def _run_loop(
    *,
    state: RunContext,
    model: Model,
    tools_map: dict[str, BaseTool],
    options: AgentOptions,
    max_input_tokens_per_call: int,
    compact_prompt: str | None,
    max_context_window: int | None,
    compact_start_seq: int,
    pending_tool_calls: list[dict] | None,
    abort_signal: AbortSignal | None,
    root_path: str | None = None,
) -> None:
    try:
        current_step = 0
        if await _process_pending_tool_calls(
            state=state,
            pending_tool_calls=pending_tool_calls,
            tools_map=tools_map,
            abort_signal=abort_signal,
        ):
            return

        while not state.is_terminal:
            current_step, compact_start_seq, should_stop = await _run_loop_iteration(
                state=state,
                model=model,
                tools_map=tools_map,
                options=options,
                current_step=current_step,
                max_input_tokens_per_call=max_input_tokens_per_call,
                compact_prompt=compact_prompt,
                max_context_window=max_context_window,
                compact_start_seq=compact_start_seq,
                abort_signal=abort_signal,
                root_path=root_path,
            )
            if should_stop:
                return
    except asyncio.CancelledError:
        set_termination_reason(state, TerminationReason.CANCELLED)
        logger.info("agent_execution_cancelled", run_id=state.run_id)
    except Exception:
        set_termination_reason(
            state,
            TerminationReason.ERROR_WITH_CONTEXT
            if state.assistant_steps_count > 0
            else TerminationReason.ERROR,
        )
        logger.error(
            "agent_execution_failed",
            run_id=state.run_id,
            steps_completed=state.steps_count,
            termination_reason=state.termination_reason,
            exc_info=True,
        )


async def _process_pending_tool_calls(
    *,
    state: RunContext,
    pending_tool_calls: list[dict] | None,
    tools_map: dict[str, BaseTool],
    abort_signal: AbortSignal | None,
) -> bool:
    if not pending_tool_calls:
        return False
    return await _execute_tool_calls(
        state=state,
        tool_calls=pending_tool_calls,
        tools_map=tools_map,
        abort_signal=abort_signal,
    )


async def _run_loop_iteration(
    *,
    state: RunContext,
    model: Model,
    tools_map: dict[str, BaseTool],
    options: AgentOptions,
    current_step: int,
    max_input_tokens_per_call: int,
    compact_prompt: str | None,
    max_context_window: int | None,
    compact_start_seq: int,
    abort_signal: AbortSignal | None,
    root_path: str | None,
) -> tuple[int, int, bool]:
    if _apply_termination_reason(
        state,
        check_non_recoverable_limits(state, options, current_step),
    ):
        return current_step, compact_start_seq, True

    compact_start_seq, should_continue = await _run_compaction_cycle(
        state=state,
        model=model,
        abort_signal=abort_signal,
        max_context_window=max_context_window,
        compact_prompt=compact_prompt,
        compact_start_seq=compact_start_seq,
        root_path=root_path,
    )
    if should_continue or state.is_terminal:
        return current_step, compact_start_seq, state.is_terminal

    current_step += 1
    step, llm_context = await _run_assistant_turn(
        state=state,
        model=model,
        abort_signal=abort_signal,
    )
    should_stop = await _handle_assistant_turn_result(
        state=state,
        step=step,
        llm_context=llm_context,
        tools_map=tools_map,
        options=options,
        max_input_tokens_per_call=max_input_tokens_per_call,
        abort_signal=abort_signal,
    )
    return current_step, compact_start_seq, should_stop


def _apply_termination_reason(
    state: RunContext,
    reason: TerminationReason | None,
) -> bool:
    if reason is None:
        return False
    set_termination_reason(state, reason)
    return True


async def _run_compaction_cycle(
    *,
    state: RunContext,
    model: Model,
    abort_signal: AbortSignal | None,
    max_context_window: int | None,
    compact_prompt: str | None,
    compact_start_seq: int,
    root_path: str | None,
) -> tuple[int, bool]:
    compact_metadata = await compact_if_needed(
        state=state,
        model=model,
        abort_signal=abort_signal,
        max_context_window=max_context_window,
        compact_prompt=compact_prompt,
        compact_start_seq=compact_start_seq,
        root_path=root_path,
    )
    if compact_metadata is None:
        return compact_start_seq, False

    compact_start_seq = compact_metadata.end_seq + 1
    logger.info(
        "compact_triggered",
        run_id=state.run_id,
        before_messages=len(state.messages),
    )
    if (
        state.config.max_run_cost is not None
        and state.token_cost >= state.config.max_run_cost
    ):
        set_termination_reason(state, TerminationReason.MAX_RUN_COST)
    return compact_start_seq, True


async def _run_assistant_turn(
    *,
    state: RunContext,
    model: Model,
    abort_signal: AbortSignal | None,
) -> tuple[StepRecord, LLMCallContext]:
    replace_messages(
        state,
        apply_steering_messages(state.copy_messages(), state.steering_queue),
    )
    if state.hooks.on_before_llm_call:
        modified = await state.hooks.on_before_llm_call(state.copy_messages())
        if modified is not None:
            replace_messages(state, modified)

    step, llm_context = await stream_assistant_step(
        model,
        state,
        abort_signal,
    )
    await commit_step(state, step, llm=llm_context)

    if state.hooks.on_after_llm_call:
        await state.hooks.on_after_llm_call(step)
    return step, llm_context


async def _handle_assistant_turn_result(
    *,
    state: RunContext,
    step: StepRecord,
    llm_context: LLMCallContext,
    tools_map: dict[str, BaseTool],
    options: AgentOptions,
    max_input_tokens_per_call: int,
    abort_signal: AbortSignal | None,
) -> bool:
    if _apply_termination_reason(
        state,
        check_post_llm_limits(
            state,
            step,
            llm_context,
            options=options,
            max_input_tokens_per_call=max_input_tokens_per_call,
        ),
    ):
        return True

    if not step.tool_calls:
        set_termination_reason(state, TerminationReason.COMPLETED)
        return True

    return await _execute_tool_calls(
        state=state,
        tool_calls=step.tool_calls,
        tools_map=tools_map,
        abort_signal=abort_signal,
    )


__all__ = ["execute_run"]
