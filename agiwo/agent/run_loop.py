"""Single-run execution engine — the core run loop."""

import asyncio
import time
from dataclasses import dataclass
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


@dataclass
class PreparedRunContext:
    """Return type for `_prepare_run_context`."""

    tools_map: dict[str, BaseTool]
    max_input_tokens_per_call: int
    compact_start_seq: int
    user_step: StepRecord
    run: Run


@dataclass
class RunLoopState:
    """Aggregated mutable state threaded through the run loop."""

    model: Model
    tools_map: dict[str, BaseTool]
    options: AgentOptions
    max_input_tokens_per_call: int
    compact_prompt: str | None
    max_context_window: int | None
    compact_start_seq: int
    abort_signal: AbortSignal | None
    root_path: str | None
    current_step: int = 0


async def _start_run(state: RunContext, run: Run) -> None:
    run.trace_id = state.trace_id
    if state.session_runtime.trace_runtime is not None:
        run.trace_id = state.session_runtime.trace_runtime.on_run_started(run)
    await state.session_runtime.run_step_storage.save_run(run)
    await state.session_runtime.publish(RunStartedEvent.from_context(state))


async def _complete_run(state: RunContext, run: Run, result: RunOutput) -> None:
    run.status = (
        RunStatus.CANCELLED
        if result.termination_reason == TerminationReason.CANCELLED
        else RunStatus.COMPLETED
    )
    run.response_content = result.response
    now = datetime.now(timezone.utc)
    run.updated_at = now
    run.metrics.end_at = now.timestamp()
    if result.metrics is not None:
        preserved_start_at = run.metrics.start_at
        preserved_end_at = run.metrics.end_at
        run.metrics = result.metrics
        run.metrics.start_at = preserved_start_at
        run.metrics.end_at = preserved_end_at
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
    now = datetime.now(timezone.utc)
    run.updated_at = now
    run.metrics.end_at = now.timestamp()
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
        response=state.ledger.response_content,
        run_id=state.run_id,
        session_id=state.session_id,
        metrics=RunMetrics.from_ledger(state.ledger, elapsed_ms=state.elapsed * 1000),
        termination_reason=state.ledger.termination_reason,
    )


async def _prepare_run_context(
    user_input: UserInput,
    *,
    context: RunContext,
    system_prompt: str,
    tools: tuple[BaseTool, ...],
    model: Model,
    options: AgentOptions,
    hooks: AgentHooks,
) -> PreparedRunContext:
    """Build all state needed before the main loop starts."""
    tools_map = {tool.get_name(): tool for tool in tools}
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

    max_input_tokens_per_call = resolve_max_input_tokens_per_call(
        options.max_input_tokens_per_call,
        model,
    )
    return PreparedRunContext(
        tools_map=tools_map,
        max_input_tokens_per_call=max_input_tokens_per_call,
        compact_start_seq=compact_start_seq,
        user_step=user_step,
        run=run,
    )


async def _finalize_run(
    user_input: UserInput,
    *,
    context: RunContext,
    run: Run,
    options: AgentOptions,
    hooks: AgentHooks,
    model: Model,
    abort_signal: AbortSignal | None,
) -> RunOutput:
    """Generate summary, build output, and complete the run."""
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

    prepared = await _prepare_run_context(
        user_input,
        context=context,
        system_prompt=system_prompt,
        tools=tools,
        model=model,
        options=options,
        hooks=hooks,
    )
    max_context_window = resolve_max_context_window(model)

    await _start_run(context, prepared.run)
    try:
        await commit_step(
            context, prepared.user_step, append_message=False, track_state=False
        )
        loop_state = RunLoopState(
            model=model,
            tools_map=prepared.tools_map,
            options=options,
            max_input_tokens_per_call=prepared.max_input_tokens_per_call,
            compact_prompt=options.compact_prompt,
            max_context_window=max_context_window,
            compact_start_seq=prepared.compact_start_seq,
            abort_signal=abort_signal,
            root_path=root_path or settings.root_path,
        )
        await _run_loop(
            state=context,
            loop=loop_state,
            pending_tool_calls=pending_tool_calls,
        )
        return await _finalize_run(
            user_input,
            context=context,
            run=prepared.run,
            options=options,
            hooks=hooks,
            model=model,
            abort_signal=abort_signal,
        )
    except Exception as error:
        await _fail_run(context, prepared.run, error)
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
    terminated = False
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
            is_error=not result.is_success,
        )
        await commit_step(state, tool_step)

        if not terminated and result.termination_reason is not None:
            set_termination_reason(state, result.termination_reason)
            terminated = True
    return terminated or state.is_terminal


async def _run_loop(
    *,
    state: RunContext,
    loop: RunLoopState,
    pending_tool_calls: list[dict] | None,
) -> None:
    try:
        if pending_tool_calls:
            terminated = await _execute_tool_calls(
                state=state,
                tool_calls=pending_tool_calls,
                tools_map=loop.tools_map,
                abort_signal=loop.abort_signal,
            )
            if terminated:
                return

        while not state.is_terminal:
            should_stop = await _run_loop_iteration(state=state, loop=loop)
            if should_stop:
                return
    except asyncio.CancelledError:
        set_termination_reason(state, TerminationReason.CANCELLED)
        logger.info("agent_execution_cancelled", run_id=state.run_id)
    except Exception:
        set_termination_reason(
            state,
            TerminationReason.ERROR_WITH_CONTEXT
            if state.ledger.assistant_steps_count > 0
            else TerminationReason.ERROR,
        )
        logger.error(
            "agent_execution_failed",
            run_id=state.run_id,
            steps_completed=state.ledger.steps_count,
            termination_reason=state.ledger.termination_reason,
            exc_info=True,
        )
        raise


async def _run_loop_iteration(
    *,
    state: RunContext,
    loop: RunLoopState,
) -> bool:
    """Execute one iteration of the run loop. Returns True if the loop should stop."""
    reason = check_non_recoverable_limits(state, loop.options, loop.current_step)
    if reason is not None:
        set_termination_reason(state, reason)
        return True

    compact_start_seq, should_continue = await _run_compaction_cycle(
        state=state,
        loop=loop,
    )
    loop.compact_start_seq = compact_start_seq
    if should_continue or state.is_terminal:
        return state.is_terminal

    loop.current_step += 1
    step, llm_context = await _run_assistant_turn(
        state=state,
        model=loop.model,
        abort_signal=loop.abort_signal,
    )
    return await _handle_assistant_turn_result(
        state=state,
        step=step,
        llm_context=llm_context,
        tools_map=loop.tools_map,
        options=loop.options,
        max_input_tokens_per_call=loop.max_input_tokens_per_call,
        abort_signal=loop.abort_signal,
    )


async def _run_compaction_cycle(
    *,
    state: RunContext,
    loop: RunLoopState,
) -> tuple[int, bool]:
    compact_metadata = await compact_if_needed(
        state=state,
        model=loop.model,
        abort_signal=loop.abort_signal,
        max_context_window=loop.max_context_window,
        compact_prompt=loop.compact_prompt,
        compact_start_seq=loop.compact_start_seq,
        root_path=loop.root_path,
    )
    if compact_metadata is None:
        return loop.compact_start_seq, False

    new_start_seq = compact_metadata.end_seq + 1
    logger.info(
        "compact_triggered",
        run_id=state.run_id,
        before_messages=len(state.ledger.messages),
    )
    if (
        state.config.max_run_cost is not None
        and state.ledger.token_cost >= state.config.max_run_cost
    ):
        set_termination_reason(state, TerminationReason.MAX_RUN_COST)
    return new_start_seq, True


async def _run_assistant_turn(
    *,
    state: RunContext,
    model: Model,
    abort_signal: AbortSignal | None,
) -> tuple[StepRecord, LLMCallContext]:
    replace_messages(
        state,
        apply_steering_messages(
            state.copy_messages(), state.session_runtime.steering_queue
        ),
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
    reason = check_post_llm_limits(
        state,
        step,
        llm_context,
        options=options,
        max_input_tokens_per_call=max_input_tokens_per_call,
    )
    if reason is not None:
        set_termination_reason(state, reason)
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
