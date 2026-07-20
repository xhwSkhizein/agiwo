"""
LLM caller — unified model-call boundary with attempt ledger and streaming.

All provider requests go through ``execute_model_call`` so every real attempt
produces Started/Completed/Failed facts with stable logical_call_id and phase.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from agiwo.agent.budget_gate import (
    LlmAttemptAdmitRequest,
    LlmAttemptCostEvent,
    LlmBudgetDenied,
    MissingBudgetGateError,
)
from agiwo.agent.models.model_call import (
    FINALIZATION_PHASES,
    ModelCallPhase,
    new_logical_call_id,
)
from agiwo.agent.models.step import (
    LLMCallContext,
    StepDelta,
    StepMetrics,
    StepView,
)
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.state_writer import RunStateWriter
from agiwo.agent.models.stream import StepDeltaEvent
from agiwo.agent.termination.run_limit import RunLimitPolicy
from agiwo.llm.base import Model, StreamChunk
from agiwo.llm.event_normalizer import normalize_usage_metrics
from agiwo.llm.usage_resolver import ModelUsageEstimator, UsageEstimate
from agiwo.agent.retry import (
    IdempotencyKind,
    RetryCoordinator,
    RunBlockingFaultError,
    map_provider_exception,
)
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

_CHUNK_TIMEOUT_SECONDS = 120


class ModelCallLimitExceeded(Exception):
    """Raised when RunLimitPolicy refuses a new provider attempt."""

    def __init__(self, phase: ModelCallPhase, reason: str) -> None:
        self.phase = phase
        self.reason = reason
        super().__init__(f"model call limit exceeded for {phase.value}: {reason}")


class ModelCallResult:
    __slots__ = ("step", "llm_context", "logical_call_id")

    def __init__(
        self,
        step: StepView,
        llm_context: LLMCallContext,
        *,
        logical_call_id: str,
    ) -> None:
        self.step = step
        self.llm_context = llm_context
        self.logical_call_id = logical_call_id


async def execute_model_call(
    *,
    model: Model,
    state: RunContext,
    writer: RunStateWriter,
    phase: ModelCallPhase,
    abort_signal: AbortSignal | None,
    messages: list[dict] | None = None,
    tools: list[dict] | None = None,
    use_state_tools: bool = True,
    name: str | None = None,
    logical_call_id: str | None = None,
    limit_policy: RunLimitPolicy | None = None,
    retry_coordinator: RetryCoordinator | None = None,
) -> ModelCallResult:
    """Run a model call through the unified attempt boundary."""
    policy = limit_policy or RunLimitPolicy()
    call_id = logical_call_id or new_logical_call_id()
    coordinator = retry_coordinator or RetryCoordinator()

    attempt_no = 1
    last_fault = None
    attempts: list = []
    while True:
        # ADR P0-04: every real provider attempt — including retries — must
        # pass RunLimitPolicy against the current ledger count.
        decision = policy.check_before_attempt(state.ledger.model_calls, phase)
        if not decision.allowed:
            raise ModelCallLimitExceeded(phase, decision.reason or "denied")

        if state.ledger.model_calls.at_or_over_limit() and phase in FINALIZATION_PHASES:
            # Over-limit finalization slots are single-shot: provider retries
            # of the same phase do not get a second over-limit allowance.
            state.ledger.model_calls.mark_finalization_consumed(phase)

        retry_reason = (
            None
            if last_fault is None
            else f"provider_retry:{last_fault.provider_code or last_fault.disposition.value}"
        )
        try:
            step, llm_context = await _stream_single_attempt(
                model=model,
                state=state,
                writer=writer,
                phase=phase,
                abort_signal=abort_signal,
                messages=messages,
                tools=tools,
                use_state_tools=use_state_tools,
                name=name,
                logical_call_id=call_id,
                attempt_no=attempt_no,
                retry_reason=retry_reason,
            )
            return ModelCallResult(step, llm_context, logical_call_id=call_id)
        except (ModelCallLimitExceeded, LlmBudgetDenied, MissingBudgetGateError):
            raise
        except RunBlockingFaultError:
            raise
        except Exception as exc:
            fault = map_provider_exception(
                exc,
                response_observed=False,
                logical_call_id=call_id,
                attempt_no=attempt_no,
            )
            attempts.append(fault)
            last_fault = fault
            if not coordinator.can_retry(
                fault,
                attempt_no=attempt_no,
                idempotency=IdempotencyKind.GUARANTEED,
            ):
                coordinator.raise_boundary(
                    fault,
                    attempts=attempts,
                    exhausted=attempt_no >= coordinator.policy.max_attempts
                    and fault.disposition.value == "retryable",
                )
                raise
            await coordinator.ensure_progress_allowed()
            if state.pause_request is not None:
                raise
            wait_seconds = min(
                coordinator.policy.max_backoff_seconds,
                coordinator.policy.min_backoff_seconds * (2 ** (attempt_no - 1)),
            )
            await writer.record_retry_backoff(
                operation="llm",
                attempt_no=attempt_no,
                wait_seconds=wait_seconds,
                reason=retry_reason or fault.disposition.value,
                logical_call_id=call_id,
            )
            logger.warning(
                "model_call_provider_retry",
                run_id=state.run_id,
                logical_call_id=call_id,
                phase=phase.value,
                attempt_no=attempt_no,
                retry_reason=retry_reason,
                disposition=fault.disposition.value,
                wait_seconds=wait_seconds,
            )
            attempt_no += 1
            await coordinator.wait_before_retry(attempt_no - 1)


async def _stream_single_attempt(
    *,
    model: Model,
    state: RunContext,
    writer: RunStateWriter,
    phase: ModelCallPhase,
    abort_signal: AbortSignal | None,
    messages: list[dict] | None,
    tools: list[dict] | None,
    use_state_tools: bool,
    name: str | None,
    logical_call_id: str,
    attempt_no: int,
    retry_reason: str | None,
) -> tuple[StepView, LLMCallContext]:
    ledger = state.ledger.model_calls
    resolved_messages = messages if messages is not None else state.snapshot_messages()
    resolved_tools = (
        state.copy_tool_schemas() if use_state_tools and tools is None else tools
    )

    metrics_resolver = ModelUsageEstimator(model)
    request_estimate = metrics_resolver.estimate_request(
        resolved_messages, resolved_tools
    )
    request_tokens = int(request_estimate.input_tokens or 0)
    max_output_tokens = int(model.max_output_tokens or 0)
    price_snapshot = metrics_resolver.price_snapshot()
    call_cost_ceiling = metrics_resolver.compute_call_cost_ceiling(
        request_tokens=request_tokens,
        max_output_tokens=max_output_tokens,
        cache_read_tokens=int(request_estimate.cache_read_tokens or 0),
        cache_creation_tokens=int(request_estimate.cache_creation_tokens or 0),
    )

    await _admit_objective_llm_attempt(
        state=state,
        model=model,
        phase=phase,
        logical_call_id=logical_call_id,
        attempt_no=attempt_no,
        call_ordinal=ledger.next_ordinal(),
        request_tokens=request_tokens,
        max_output_tokens=max_output_tokens,
        call_cost_ceiling=call_cost_ceiling,
        price_snapshot=price_snapshot,
    )

    call_ordinal = ledger.record_attempt_started(phase)
    await writer.record_llm_call_started(
        messages=resolved_messages,
        tools=resolved_tools,
        logical_call_id=logical_call_id,
        phase=phase,
        attempt_no=attempt_no,
        call_ordinal=call_ordinal,
        retry_reason=retry_reason,
        request_tokens=request_tokens,
        call_cost_ceiling=call_cost_ceiling,
        price_snapshot=price_snapshot,
    )

    try:
        step, llm_context = await _stream_assistant_step_inner(
            model,
            state,
            abort_signal,
            messages=resolved_messages,
            tools=resolved_tools,
            use_state_tools=False,
            name=name,
        )
    except Exception as exc:
        await writer.record_llm_call_failed(
            logical_call_id=logical_call_id,
            phase=phase,
            attempt_no=attempt_no,
            call_ordinal=call_ordinal,
            retry_reason=retry_reason,
            error=str(exc),
            response_observed=False,
            request_tokens=request_tokens,
            call_cost_ceiling=call_cost_ceiling,
            price_snapshot=price_snapshot,
        )
        ledger.record_attempt_failed(phase)
        await _record_objective_llm_cost(
            state=state,
            phase=phase,
            logical_call_id=logical_call_id,
            attempt_no=attempt_no,
            call_ordinal=call_ordinal,
            retry_reason=retry_reason,
            request_tokens=request_tokens,
            accepted_output_tokens=0,
            call_cost_ceiling=call_cost_ceiling,
            cost_usd=0.0,
            response_observed=False,
            source="no_response",
            price_snapshot=price_snapshot,
        )
        raise

    await writer.record_llm_call_completed(
        step=step,
        llm=llm_context,
        logical_call_id=logical_call_id,
        phase=phase,
        attempt_no=attempt_no,
        call_ordinal=call_ordinal,
        retry_reason=retry_reason,
        response_observed=True,
        request_tokens=request_tokens,
        call_cost_ceiling=call_cost_ceiling,
        price_snapshot=price_snapshot,
    )
    ledger.record_attempt_completed(phase)
    cost_usd = float(step.metrics.token_cost or 0.0) if step.metrics else 0.0
    accepted_output = int(step.metrics.output_tokens or 0) if step.metrics else 0
    usage_source = (
        str(step.metrics.usage_source)
        if step.metrics and step.metrics.usage_source
        else "estimated"
    )
    await _record_objective_llm_cost(
        state=state,
        phase=phase,
        logical_call_id=logical_call_id,
        attempt_no=attempt_no,
        call_ordinal=call_ordinal,
        retry_reason=retry_reason,
        request_tokens=request_tokens,
        accepted_output_tokens=accepted_output,
        call_cost_ceiling=call_cost_ceiling,
        cost_usd=cost_usd,
        response_observed=True,
        source=usage_source,
        price_snapshot=price_snapshot,
    )
    return step, llm_context


async def _admit_objective_llm_attempt(
    *,
    state: RunContext,
    model: Model,
    phase: ModelCallPhase,
    logical_call_id: str,
    attempt_no: int,
    call_ordinal: int,
    request_tokens: int,
    max_output_tokens: int,
    call_cost_ceiling: float,
    price_snapshot: dict[str, float],
) -> None:
    del model  # prices already snapshotted by caller
    if state.objective_id is None:
        return
    gate = state.llm_budget_gate
    if gate is None:
        raise MissingBudgetGateError(state.objective_id)
    await gate.check_before_attempt(
        LlmAttemptAdmitRequest(
            objective_id=state.objective_id,
            run_id=state.run_id,
            logical_call_id=logical_call_id,
            phase=phase.value,
            attempt_no=attempt_no,
            call_ordinal=call_ordinal,
            request_tokens=request_tokens,
            max_output_tokens=max_output_tokens,
            call_cost_ceiling=call_cost_ceiling,
            price_snapshot=price_snapshot,
        )
    )


async def _record_objective_llm_cost(
    *,
    state: RunContext,
    phase: ModelCallPhase,
    logical_call_id: str,
    attempt_no: int,
    call_ordinal: int,
    retry_reason: str | None,
    request_tokens: int,
    accepted_output_tokens: int,
    call_cost_ceiling: float,
    cost_usd: float,
    response_observed: bool,
    source: str,
    price_snapshot: dict[str, float],
) -> None:
    if state.objective_id is None:
        return
    gate = state.llm_budget_gate
    if gate is None:
        raise MissingBudgetGateError(state.objective_id)
    await gate.record_attempt_cost(
        LlmAttemptCostEvent(
            objective_id=state.objective_id,
            run_id=state.run_id,
            logical_call_id=logical_call_id,
            phase=phase.value,
            attempt_no=attempt_no,
            call_ordinal=call_ordinal,
            request_tokens=request_tokens,
            accepted_output_tokens=accepted_output_tokens,
            call_cost_ceiling=call_cost_ceiling,
            cost_usd=cost_usd,
            response_observed=response_observed,
            source=source,
            price_snapshot=price_snapshot,
            retry_reason=retry_reason,
        )
    )


async def _stream_assistant_step_inner(
    model: Model,
    state: RunContext,
    abort_signal: AbortSignal | None,
    *,
    messages: list[dict] | None = None,
    tools: list[dict] | None = None,
    use_state_tools: bool = True,
    name: str | None = None,
) -> tuple[StepView, LLMCallContext]:
    messages = messages if messages is not None else state.snapshot_messages()
    tools_resolved = (
        state.copy_tool_schemas() if use_state_tools and tools is None else tools
    )

    metrics_resolver = ModelUsageEstimator(model)
    request_estimate = metrics_resolver.estimate_request(messages, tools_resolved)

    logger.debug(
        "llm_request",
        model=model,
        messages_count=len(messages),
        tools_count=len(tools_resolved) if tools_resolved else 0,
        detail=_get_request_params(model),
    )

    llm_context = LLMCallContext(
        messages=list(messages),
        tools=list(tools_resolved) if tools_resolved else None,
        request_params=_get_request_params(model),
    )

    step_start_time = time.time()
    sequence = await state.session_runtime.allocate_sequence()
    step = StepView.assistant(
        state,
        sequence=sequence,
        content="",
        tool_calls=None,
        metrics=StepMetrics(start_at=datetime.now(timezone.utc)),
        name=name,
    )
    if step.metrics is not None:
        step.metrics.model_name = model.name
        step.metrics.provider = model.provider

    first_token_received = False
    finish_reason: str | None = None
    tool_calls_acc: dict[int, dict] = {}

    stream = model.arun_stream(messages, tools=tools_resolved)
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(
                    stream.__anext__(), timeout=_CHUNK_TIMEOUT_SECONDS
                )
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"LLM stream stalled: no chunk received for {_CHUNK_TIMEOUT_SECONDS}s"
                ) from exc

            _check_abort(abort_signal)
            delta, has_content, chunk_finish_reason = _apply_chunk_to_step(
                step=step,
                chunk=chunk,
                tool_calls_acc=tool_calls_acc,
            )

            if has_content and not first_token_received:
                first_token_received = True
                if step.metrics:
                    step.metrics.first_token_latency_ms = (
                        time.time() - step_start_time
                    ) * 1000

            if chunk_finish_reason:
                finish_reason = chunk_finish_reason

            if has_content or delta.usage:
                await state.session_runtime.publish(
                    StepDeltaEvent.from_context(state, step_id=step.id, delta=delta),
                )
    finally:
        await stream.aclose()

    step.content = step.content or None
    step.reasoning_content = step.reasoning_content or None
    step.tool_calls = _finalize_tool_calls(tool_calls_acc) or None

    if step.metrics:
        step.metrics.end_at = datetime.now(timezone.utc)
        step.metrics.duration_ms = (time.time() - step_start_time) * 1000

    _resolve_step_metrics(step, metrics_resolver, request_estimate)
    llm_context.finish_reason = finish_reason
    return step, llm_context


def _accumulate_tool_calls(
    tool_calls_acc: dict[int, dict], delta_calls: list[dict]
) -> None:
    for tc in delta_calls:
        idx = tc.get("index", 0)

        if idx not in tool_calls_acc:
            tool_calls_acc[idx] = {
                "id": None,
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }

        acc = tool_calls_acc[idx]

        if tc.get("id"):
            acc["id"] = tc["id"]

        if tc.get("type"):
            acc["type"] = tc["type"]

        if tc.get("function"):
            fn = tc["function"]
            if fn.get("name"):
                acc["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                acc["function"]["arguments"] += fn["arguments"]


def _finalize_tool_calls(tool_calls_acc: dict[int, dict]) -> list[dict]:
    return [call for call in tool_calls_acc.values() if call["id"] is not None]


def _apply_chunk_to_step(
    *,
    step: StepView,
    chunk: StreamChunk,
    tool_calls_acc: dict[int, dict],
) -> tuple[StepDelta, bool, str | None]:
    delta = StepDelta()
    has_content = bool(chunk.content or chunk.reasoning_content or chunk.tool_calls)

    if chunk.content:
        step.content = (step.content or "") + chunk.content
        delta.content = chunk.content

    if chunk.reasoning_content:
        step.reasoning_content = (
            step.reasoning_content or ""
        ) + chunk.reasoning_content
        delta.reasoning_content = chunk.reasoning_content

    if chunk.tool_calls:
        _accumulate_tool_calls(tool_calls_acc, chunk.tool_calls)
        delta.tool_calls = chunk.tool_calls

    if chunk.usage and step.metrics:
        normalized = normalize_usage_metrics(chunk.usage)
        step.metrics.input_tokens = normalized["input_tokens"]
        step.metrics.output_tokens = normalized["output_tokens"]
        step.metrics.total_tokens = normalized["total_tokens"]
        step.metrics.cache_read_tokens = normalized["cache_read_tokens"]
        step.metrics.cache_creation_tokens = normalized["cache_creation_tokens"]
        step.metrics.usage_source = "provider"
        delta.usage = normalized

    return delta, has_content, chunk.finish_reason


def _resolve_step_metrics(
    step: StepView,
    metrics_resolver: ModelUsageEstimator,
    request_estimate: UsageEstimate | None,
) -> None:
    if step.metrics is None:
        return

    had_provider_usage = any(
        value is not None
        for value in (
            step.metrics.input_tokens,
            step.metrics.output_tokens,
            step.metrics.total_tokens,
            step.metrics.cache_read_tokens,
            step.metrics.cache_creation_tokens,
        )
    )

    estimated_output = metrics_resolver.estimate_assistant_output(
        content=step.content if isinstance(step.content, str) else None,
        reasoning_content=step.reasoning_content,
        tool_calls=step.tool_calls,
    )

    if step.metrics.input_tokens is None and request_estimate is not None:
        step.metrics.input_tokens = request_estimate.input_tokens
    if step.metrics.output_tokens is None:
        step.metrics.output_tokens = estimated_output
    if step.metrics.total_tokens is None:
        step.metrics.total_tokens = (step.metrics.input_tokens or 0) + (
            step.metrics.output_tokens or 0
        )
    if step.metrics.cache_read_tokens is None:
        step.metrics.cache_read_tokens = (
            request_estimate.cache_read_tokens if request_estimate else 0
        )
    if step.metrics.cache_creation_tokens is None:
        step.metrics.cache_creation_tokens = (
            request_estimate.cache_creation_tokens if request_estimate else 0
        )

    if had_provider_usage:
        if request_estimate is not None and (
            step.metrics.input_tokens == request_estimate.input_tokens
            or step.metrics.output_tokens == estimated_output
        ):
            step.metrics.usage_source = "mixed"
        else:
            step.metrics.usage_source = "provider"
    else:
        step.metrics.usage_source = "estimated"

    step.metrics.token_cost = metrics_resolver.compute_cost(
        input_tokens=step.metrics.input_tokens,
        output_tokens=step.metrics.output_tokens,
        cache_read_tokens=step.metrics.cache_read_tokens,
        cache_creation_tokens=step.metrics.cache_creation_tokens,
    )


def _get_request_params(model: Model) -> dict[str, Any]:
    return {
        "model_id": model.id,
        "model_name": model.name,
        "temperature": model.temperature,
        "max_output_tokens": model.max_output_tokens,
        "top_p": model.top_p,
    }


def _check_abort(abort_signal: AbortSignal | None) -> None:
    if abort_signal and abort_signal.is_aborted():
        raise asyncio.CancelledError(abort_signal.reason)


__all__ = [
    "LlmBudgetDenied",
    "MissingBudgetGateError",
    "ModelCallLimitExceeded",
    "ModelCallResult",
    "execute_model_call",
]
