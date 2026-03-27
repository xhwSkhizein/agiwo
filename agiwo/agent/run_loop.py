"""Single-run execution engine — the core run loop."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from agiwo.agent.compaction import compact_if_needed
from agiwo.agent.config import AgentOptions
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.input import ChannelContext, UserInput
from agiwo.agent.input_codec import normalize_to_message
from agiwo.agent.llm_caller import stream_assistant_step
from agiwo.agent.memory_types import MemoryRecord
from agiwo.agent.run_mutations import (
    record_compaction_metadata,
    replace_messages,
    set_termination_reason,
)
from agiwo.agent.run_state import RunContext
from agiwo.agent.step_pipeline import commit_step
from agiwo.agent.tool_executor import execute_tool_batch
from agiwo.tool.base import BaseTool
from agiwo.agent.types import (
    LLMCallContext,
    Run,
    RunCompletedEvent,
    RunFailedEvent,
    RunMetrics,
    RunOutput,
    RunStartedEvent,
    RunStatus,
    StepRecord,
    TerminationReason,
    steps_to_messages,
)
from agiwo.config.settings import settings
from agiwo.llm.base import Model
from agiwo.llm.limits import (
    resolve_max_context_window,
    resolve_max_input_tokens_per_call,
)
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

_SUMMARY_REASONS = {
    TerminationReason.MAX_STEPS,
    TerminationReason.TIMEOUT,
    TerminationReason.MAX_OUTPUT_TOKENS,
    TerminationReason.MAX_INPUT_TOKENS_PER_CALL,
    TerminationReason.MAX_RUN_COST,
    TerminationReason.CANCELLED,
    TerminationReason.TOOL_LIMIT,
    TerminationReason.ERROR,
    TerminationReason.ERROR_WITH_CONTEXT,
}

DEFAULT_TERMINATION_USER_PROMPT = """**IMPORTANT: Execution Limit Reached**

The execution has been interrupted due to %s. This is NOT a normal completion.

Please provide a summary report that includes:
1. **Original Request**: What was the user asking for
2. **Work Completed**: What has been accomplished so far (be specific, cite actual results)
3. **Pending Work**: What remains incomplete or was interrupted
4. **Key Findings & Refs**: Any important results, data, or conclusions discovered with references

**Requirements**:
- Base your summary ONLY on the actual work done and results obtained - do not fabricate or assume
- If you must make any assumptions, clearly mark them as such
- Clearly indicate this is an INCOMPLETE/INTERRUPTED execution report
- Use the same language as the original request"""


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


def _assemble_messages(
    system_prompt: str,
    existing_steps: list[StepRecord] | None = None,
    memories: list[MemoryRecord] | None = None,
    before_run_hook_result: str | None = None,
    *,
    channel_context: ChannelContext | None = None,
) -> list[dict]:
    if existing_steps is None:
        existing_steps = []
    if memories is None:
        memories = []

    messages: list[dict] = steps_to_messages(existing_steps)
    filtered_memories = _filter_memories(messages, memories)

    preamble_parts: list[str] = []
    if channel_context:
        preamble_parts.append(_render_channel_context(channel_context))
    if filtered_memories:
        preamble_parts.append(_render_memories(filtered_memories))
    if before_run_hook_result:
        preamble_parts.append(_render_hook_result(before_run_hook_result))

    if preamble_parts and messages:
        last_msg = messages[-1]
        if last_msg.get("role") == "user":
            _prepend_to_user_message(last_msg, "\n\n".join(preamble_parts))

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    return messages


def _apply_steering_messages(
    messages: list[dict],
    steering_queue: asyncio.Queue[object] | None,
) -> None:
    if steering_queue is None or steering_queue.empty():
        return

    parts: list[str] = []
    while not steering_queue.empty():
        try:
            parts.append(str(steering_queue.get_nowait()))
        except asyncio.QueueEmpty:
            break

    if not parts:
        return

    steering_text = "\n".join(parts)
    tag = f"\n\n<system-steering>{steering_text}</system-steering>"
    last_message = messages[-1] if messages else None
    if last_message and last_message.get("role") in ("user", "tool"):
        content = last_message.get("content", "")
        if isinstance(content, str):
            last_message["content"] = content + tag
        elif isinstance(content, list):
            content.append({"type": "text", "text": tag})
        else:
            last_message["content"] = tag
        return

    messages.append({"role": "user", "content": steering_text})


def _render_channel_context(ctx: ChannelContext) -> str:
    lines = [f"source: {ctx.source}"]
    for key, value in ctx.metadata.items():
        if key in ("recent_dm_messages", "recent_group_messages") and isinstance(
            value, list
        ):
            if value:
                lines.append(f"{key}:")
                for msg in value:
                    lines.append(f"  - {msg}")
        elif isinstance(value, (str, int, float, bool)):
            lines.append(f"{key}: {value}")
    return "<channel-context>\n" + "\n".join(lines) + "\n</channel-context>"


def _render_memories(memories: list[MemoryRecord]) -> str:
    content = "\n\n".join(memory.content for memory in memories)
    return f"<relevant-memories>\n{content}\n</relevant-memories>"


def _render_hook_result(result: str) -> str:
    return f"<before_run_hook_result>\n{result}\n</before_run_hook_result>"


def _prepend_to_user_message(msg: dict[str, Any], preamble: str) -> None:
    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = preamble + "\n\n" + content
    elif isinstance(content, list):
        msg["content"] = [{"type": "text", "text": preamble}] + content
    else:
        msg["content"] = preamble


def _filter_memories(
    messages: list[dict], memories: list[MemoryRecord]
) -> list[MemoryRecord]:
    if not memories:
        return []

    min_relevance_score = 0.5
    similarity_threshold = 0.8

    existing_texts: list[str] = [
        msg.get("content", "")
        for msg in messages[:-1]
        if isinstance(msg.get("content"), str)
    ]

    def _is_similar_to_history(content: str) -> bool:
        content_lower = content.lower()
        for text in existing_texts:
            if _text_similarity(content_lower, text.lower()) > similarity_threshold:
                return True
        return False

    filtered: list[MemoryRecord] = []
    seen_contents: set[str] = set()

    for memory in sorted(
        [m for m in memories if m.relevance_score is not None],
        key=lambda m: m.relevance_score or 0,
        reverse=True,
    ):
        if memory.relevance_score < min_relevance_score:
            continue

        content_normalized = memory.content.strip()
        if content_normalized in seen_contents:
            continue
        seen_contents.add(content_normalized)

        if _is_similar_to_history(content_normalized):
            continue

        filtered.append(memory)

    return filtered


def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 0.9

    a_words = set(a.split())
    b_words = set(b.split())
    if not a_words or not b_words:
        return 0.0

    intersection = a_words & b_words
    union = a_words | b_words
    return len(intersection) / len(union)


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
    user_message = normalize_to_message(user_input)
    context.config = options
    context.hooks = hooks
    replace_messages(
        context,
        _assemble_messages(
            system_prompt,
            existing_steps,
            memories,
            before_run_hook_result,
            channel_context=user_message.context,
        ),
    )
    context.tool_schemas = tool_schemas
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
        await _maybe_generate_summary(
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
        _check_non_recoverable_limits(state, options, current_step),
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
    _apply_steering_messages(state.messages, state.steering_queue)
    if state.hooks.on_before_llm_call:
        modified = await state.hooks.on_before_llm_call(state.messages)
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
        _check_post_llm_limits(
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


def _check_non_recoverable_limits(
    state: RunContext,
    options: AgentOptions,
    current_step: int,
) -> TerminationReason | None:
    if current_step >= options.max_steps:
        logger.warning(
            "limit_hit_max_steps",
            current_step=current_step,
            max_steps=options.max_steps,
            run_id=state.run_id,
        )
        return TerminationReason.MAX_STEPS

    if options.run_timeout and state.elapsed > options.run_timeout:
        logger.warning(
            "limit_hit_timeout",
            elapsed=state.elapsed,
            run_timeout=options.run_timeout,
            run_id=state.run_id,
        )
        return TerminationReason.TIMEOUT

    if state.timeout_at and time.time() >= state.timeout_at:
        logger.warning(
            "limit_hit_context_timeout",
            timeout_at=state.timeout_at,
            run_id=state.run_id,
        )
        return TerminationReason.TIMEOUT

    return None


def _check_post_llm_limits(
    state: RunContext,
    step: StepRecord,
    llm_context: LLMCallContext,
    *,
    options: AgentOptions,
    max_input_tokens_per_call: int,
) -> TerminationReason | None:
    input_tokens = step.metrics.input_tokens if step.metrics else 0
    if input_tokens and input_tokens > max_input_tokens_per_call:
        logger.warning(
            "limit_hit_max_input_tokens_per_call",
            input_tokens=input_tokens,
            max_input_tokens_per_call=max_input_tokens_per_call,
            run_id=state.run_id,
        )
        return TerminationReason.MAX_INPUT_TOKENS_PER_CALL

    if options.max_run_cost is not None and state.token_cost >= options.max_run_cost:
        logger.warning(
            "limit_hit_max_run_cost",
            token_cost=state.token_cost,
            max_run_cost=options.max_run_cost,
            run_id=state.run_id,
        )
        return TerminationReason.MAX_RUN_COST

    finish_reason = llm_context.finish_reason
    if finish_reason and finish_reason.strip().lower() in {"length", "max_tokens"}:
        logger.warning(
            "limit_hit_max_output_tokens",
            finish_reason=finish_reason,
            run_id=state.run_id,
        )
        return TerminationReason.MAX_OUTPUT_TOKENS

    return None


async def _maybe_generate_summary(
    *,
    state: RunContext,
    options: AgentOptions,
    model: Model,
    abort_signal: AbortSignal | None,
) -> None:
    if not options.enable_termination_summary:
        return
    if state.termination_reason not in _SUMMARY_REASONS:
        return

    prompt_template = (
        options.termination_summary_prompt or DEFAULT_TERMINATION_USER_PROMPT
    )
    termination_reason_str = _format_termination_reason(state.termination_reason)
    user_prompt = (
        prompt_template % termination_reason_str
        if "%s" in prompt_template
        else prompt_template
    )

    sequence = await state.session_runtime.allocate_sequence()
    summary_user_step = StepRecord.user(
        state,
        sequence=sequence,
        content=user_prompt,
        name="summary_request",
    )
    await commit_step(state, summary_user_step, append_message=True)

    step, llm_context = await stream_assistant_step(
        model,
        state,
        abort_signal,
        messages=state.messages,
        tools=None,
    )
    step.name = "summary"
    await commit_step(state, step, llm=llm_context, append_message=False)

    logger.info(
        "summary_generated",
        tokens=step.metrics.total_tokens if step.metrics else 0,
    )


def _format_termination_reason(reason: TerminationReason | str) -> str:
    reason_val = reason.value if isinstance(reason, TerminationReason) else reason
    reason_mapping = {
        TerminationReason.MAX_STEPS.value: "reaching the maximum number of execution steps",
        TerminationReason.TIMEOUT.value: "execution timeout",
        TerminationReason.MAX_OUTPUT_TOKENS.value: "reaching model output token limit for one LLM call",
        TerminationReason.MAX_INPUT_TOKENS_PER_CALL.value: "reaching model input token limit for one LLM call",
        TerminationReason.MAX_RUN_COST.value: "reaching maximum token cost budget for this run",
        TerminationReason.CANCELLED.value: "user cancellation",
        TerminationReason.TOOL_LIMIT.value: "reaching the tool call limit",
        TerminationReason.ERROR.value: "internal error",
        TerminationReason.ERROR_WITH_CONTEXT.value: "error with context",
        TerminationReason.COMPLETED.value: "completed successfully",
        TerminationReason.SLEEPING.value: "sleeping/waiting",
    }
    return reason_mapping.get(reason_val, reason_val)


__all__ = ["execute_run"]
