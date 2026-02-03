"""
Trace collector - builds Trace from StepEvent stream.

Uses middleware pattern to wrap event streams without modifying core execution logic.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator
from uuid import uuid4

from agiwo.agent.schema import Step, StepEvent, StepEventType
from agiwo.observability.trace import Span, SpanKind, SpanStatus, Trace
from agiwo.observability.otlp_exporter import get_otlp_exporter
from agiwo.utils.logging import get_logger
from agiwo.observability.store import TraceStore

logger = get_logger(__name__)


class TraceCollector:
    """
    Trace collector - constructs Trace from StepEvent stream.

    Usage:
        collector = TraceCollector(store)

        async for event in collector.wrap_stream(event_stream):
            yield event  # Events are passed through while building trace
    """

    PREVIEW_LENGTH = 500  # Input/output preview length

    def __init__(self, store: "TraceStore | None" = None) -> None:
        """
        Initialize collector.

        Args:
            store: Optional TraceStore for persistence
        """
        self.store = store

        # Internal state for push mode
        self._trace: Trace | None = None
        self._span_stack: dict[str, Span] = {}
        self._current_span: Span | None = None
        self._assistant_step_cache: dict[str, Step] = {}

    def start(
        self,
        trace_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        input_query: str | None = None,
    ) -> None:
        """Initialize trace collection state."""
        self._trace = Trace(
            trace_id=trace_id or str(uuid4()),
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            input_query=input_query,
        )
        self._span_stack = {}
        self._current_span = None
        self._assistant_step_cache = {}

    async def collect(self, event: StepEvent) -> None:
        """Process a single event in push mode."""
        if not self._trace:
            # If start() wasn't called, we can't collect.
            # Ideally should raise error or auto-start, but safe fallback is log/ignore?
            # For strictness, let's assume start() must be called.
            return

        current_span, updated_cache = self._process_event(
            event,
            self._trace,
            self._span_stack,
            self._current_span,
            self._assistant_step_cache,
        )

        self._current_span = current_span
        if updated_cache:
            self._assistant_step_cache.update(updated_cache)

        # Incremental save at critical checkpoints
        should_save = event.type in {
            StepEventType.RUN_STARTED,
            StepEventType.STEP_COMPLETED,
            StepEventType.RUN_COMPLETED,
            StepEventType.RUN_FAILED,
        }

        if should_save and self.store:
            # Save asynchronously in background to avoid blocking
            asyncio.create_task(self._save_trace_safe())

        # Inject trace fields into event
        event.trace_id = self._trace.trace_id
        if current_span:
            event.span_id = current_span.span_id
            event.parent_span_id = current_span.parent_span_id

    async def stop(self) -> None:
        """Finalize and save trace."""
        if not self._trace:
            return

        trace = self._trace

        # Cleanup
        if not trace.end_time:
            # If active span exists, mark as error or OK?
            # wrap_stream defaulted to OK if not explicitly set.
            status = SpanStatus.OK
            if self._current_span and self._current_span.status == SpanStatus.UNSET:
                # If we are stopping abruptly, maybe we should respect current state?
                pass
            trace.complete(status=status)

        # Final save to ensure complete state
        # Note: Trace is also saved incrementally during execution (see collect())
        # This final save ensures the complete end state and OTLP export
        if self.store:
            try:
                await self.store.save_trace(trace)
            except Exception as e:
                logger.error("trace_save_failed", trace_id=trace.trace_id, error=str(e))

        # Export to OTLP (async, non-blocking)
        try:
            exporter = get_otlp_exporter()
            if exporter.enabled:
                await exporter.export_trace(trace)
        except Exception as e:
            logger.warning("otlp_export_failed", trace_id=trace.trace_id, error=str(e))

        # Reset state
        self._trace = None
        self._span_stack = {}
        self._current_span = None
        self._assistant_step_cache = {}

    def fail(self, error: Exception) -> None:
        """Mark trace as failed."""
        if not self._trace:
            return

        self._trace.complete(status=SpanStatus.ERROR)
        if self._current_span:
            self._current_span.complete(
                status=SpanStatus.ERROR,
                error_message=str(error),
            )
        logger.error(
            "trace_collection_failed", trace_id=self._trace.trace_id, error=str(error)
        )

    async def _save_trace_safe(self) -> None:
        """Save trace to store with error handling (non-blocking)."""
        if not self._trace or not self.store:
            return

        try:
            await self.store.save_trace(self._trace)
        except Exception as e:
            # Log but don't raise - incremental saves should not break execution
            logger.warning(
                "trace_incremental_save_failed",
                trace_id=self._trace.trace_id,
                error=str(e),
            )

    async def wrap_stream(
        self,
        event_stream: AsyncIterator[StepEvent],
        trace_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        input_query: str | None = None,
    ) -> AsyncIterator[StepEvent]:
        """
        Wrap event stream to automatically collect trace information.

        Args:
            event_stream: Original event stream
            trace_id: Optional trace ID
            agent_id: Agent ID
            session_id: Session ID
            user_id: User ID
            input_query: User input

        Yields:
            StepEvent: Original events with injected trace_id/span_id
        """
        self.start(
            trace_id=trace_id,
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            input_query=input_query,
        )

        try:
            async for event in event_stream:
                await self.collect(event)
                yield event

        except Exception as e:
            self.fail(e)
            raise
        finally:
            await self.stop()

    def _process_event(
        self,
        event: StepEvent,
        trace: Trace,
        span_stack: dict[str, Span],
        current_span: Span | None,
        assistant_step_cache: dict[str, Step],
    ) -> tuple[Span | None, dict[str, Step]]:
        """
        Process single event, return currently active span and updated assistant_step_cache.

        Returns:
            tuple[Span | None, dict[str, Step]]: (current_span, updated_cache)
        """

        event_type = event.type

        # === RUN_STARTED ===
        if event_type == StepEventType.RUN_STARTED:
            data = event.data or {}

            # Check if this is a nested execution
            if event.agent_id:
                # Find parent span for nested execution
                parent_span = self._find_parent_span_for_nested(
                    event, span_stack, current_span
                )

                # Create nested Agent Span
                span = Span(
                    trace_id=trace.trace_id,
                    parent_span_id=parent_span.span_id if parent_span else None,
                    kind=SpanKind.AGENT,
                    name=event.agent_id,
                    depth=(parent_span.depth + 1) if parent_span else 1,
                    attributes={
                        "agent_id": event.agent_id,
                        "nested": True,
                        "parent_run_id": event.parent_run_id,
                        "session_id": data.get("session_id"),
                    },
                    run_id=event.run_id,
                )
                trace.add_span(span)
                span_stack[event.run_id] = span
                return span, {}

            # Agent span (top-level or nested)
            parent = current_span
            span = Span(
                trace_id=trace.trace_id,
                parent_span_id=parent.span_id if parent else None,
                kind=SpanKind.AGENT,
                name=data.get("agent_id", "agent"),
                depth=(parent.depth + 1) if parent else 0,
                attributes={
                    "agent_id": data.get("agent_id"),
                    "session_id": data.get("session_id"),
                },
            )
            if not trace.root_span_id:
                trace.root_span_id = span.span_id

            trace.add_span(span)
            span_stack[event.run_id] = span
            return span, {}

        # === STEP_COMPLETED ===
        elif event_type == StepEventType.STEP_COMPLETED:
            if not event.snapshot:
                return current_span, {}

            step = event.snapshot
            updated_cache = {}

            # Cache Assistant Step for tool call parameter extraction
            if step.role.value == "assistant" and step.tool_calls:
                for tool_call in step.tool_calls:
                    tool_call_id = tool_call.get("id")
                    if tool_call_id:
                        updated_cache[tool_call_id] = step

            if step.role.value == "tool":
                # Tool 调用 Span - 使用 Step 的时间戳
                from datetime import timezone

                start_time = (
                    step.metrics.start_at
                    if step.metrics and step.metrics.start_at
                    else step.created_at
                )
                # Ensure start_time is timezone-aware
                if start_time and start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=timezone.utc)
                end_time = (
                    step.metrics.end_at
                    if step.metrics and step.metrics.end_at
                    else None
                )
                # Ensure end_time is timezone-aware
                if end_time and end_time.tzinfo is None:
                    end_time = end_time.replace(tzinfo=timezone.utc)
                duration_ms = step.metrics.duration_ms if step.metrics else None

                # Determine parent span - improved for nested execution
                parent_span_id = step.parent_span_id
                if not parent_span_id:
                    # For nested execution, try to find parent Agent Span first
                    if step.parent_run_id:
                        parent_agent_span = span_stack.get(step.parent_run_id)
                        if parent_agent_span:
                            parent_span_id = parent_agent_span.span_id

                    # Fallback: find parent from span_stack
                    if not parent_span_id:
                        # Find Agent Span for this run_id
                        run_span = span_stack.get(step.run_id)
                        parent_span_id = run_span.span_id if run_span else None
                    if not parent_span_id and current_span:
                        parent_span_id = current_span.span_id

                # Get parent span for depth calculation
                parent = None
                if parent_span_id:
                    # Find parent span by span_id
                    for span in span_stack.values():
                        if span.span_id == parent_span_id:
                            parent = span
                            break
                if not parent:
                    parent = span_stack.get(step.run_id) or current_span
                parent_depth = parent.depth if parent else 0

                # Extract tool input arguments from Assistant Step
                tool_input_args = {}
                if step.tool_call_id:
                    assistant_step = assistant_step_cache.get(step.tool_call_id)
                    if assistant_step and assistant_step.tool_calls:
                        # Find matching tool_call in Assistant Step
                        for tc in assistant_step.tool_calls:
                            if tc.get("id") == step.tool_call_id:
                                fn_args_str = tc.get("function", {}).get(
                                    "arguments", "{}"
                                )
                                try:
                                    if isinstance(fn_args_str, str):
                                        tool_input_args = json.loads(fn_args_str)
                                    else:
                                        tool_input_args = fn_args_str or {}
                                except json.JSONDecodeError:
                                    tool_input_args = {}
                                break

                # Build tool_details
                tool_details = self._build_tool_details(step, tool_input_args)

                # Determine status
                is_error = step.content and step.content.startswith("Error:")
                status = SpanStatus.ERROR if is_error else SpanStatus.OK
                error_message = step.content if is_error else None

                span = Span(
                    trace_id=trace.trace_id,
                    parent_span_id=parent_span_id,
                    kind=SpanKind.TOOL_CALL,
                    name=step.name or "tool",
                    depth=step.depth if step.depth > 0 else (parent_depth + 1),
                    attributes={
                        "tool_name": step.name,
                        "tool_call_id": step.tool_call_id,
                        "tool_id": step.name,
                    },
                    tool_details=tool_details,
                    step_id=step.id,
                    run_id=step.run_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=duration_ms
                    or (step.metrics.duration_ms if step.metrics else None),
                    status=status,
                    error_message=error_message,
                )

                if step.metrics:
                    span.metrics = {
                        "tool.exec_time_ms": step.metrics.duration_ms,
                        "duration_ms": span.duration_ms,
                    }

                trace.add_span(span)
                return current_span, updated_cache

            elif step.role.value == "assistant":
                # LLM 调用 Span - 使用 Step 的时间戳和上下文
                from datetime import timezone

                start_time = (
                    step.metrics.start_at
                    if step.metrics and step.metrics.start_at
                    else step.created_at
                )
                # Ensure start_time is timezone-aware
                if start_time and start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=timezone.utc)
                end_time = (
                    step.metrics.end_at
                    if step.metrics and step.metrics.end_at
                    else None
                )
                # Ensure end_time is timezone-aware
                if end_time and end_time.tzinfo is None:
                    end_time = end_time.replace(tzinfo=timezone.utc)
                duration_ms = step.metrics.duration_ms if step.metrics else None

                # Determine parent span - improved for nested execution
                parent_span_id = step.parent_span_id
                if not parent_span_id:
                    # For nested execution, try to find parent Agent Span first
                    if step.parent_run_id:
                        parent_agent_span = span_stack.get(step.parent_run_id)
                        if parent_agent_span:
                            parent_span_id = parent_agent_span.span_id

                    # Fallback: find parent from span_stack
                    if not parent_span_id:
                        # Find Agent Span for this run_id
                        run_span = span_stack.get(step.run_id)
                        parent_span_id = run_span.span_id if run_span else None
                    if not parent_span_id and current_span:
                        parent_span_id = current_span.span_id

                # Get parent span for depth calculation
                parent = None
                if parent_span_id:
                    # Find parent span by span_id
                    for span in span_stack.values():
                        if span.span_id == parent_span_id:
                            parent = span
                            break
                if not parent:
                    parent = span_stack.get(step.run_id) or current_span
                parent_depth = parent.depth if parent else 0

                # Build name
                name = (
                    step.metrics.model_name
                    if step.metrics and step.metrics.model_name
                    else "llm"
                )

                span = Span(
                    trace_id=trace.trace_id,
                    parent_span_id=parent_span_id,
                    kind=SpanKind.LLM_CALL,
                    name=name,
                    depth=step.depth if step.depth > 0 else (parent_depth + 1),
                    attributes={
                        "model_name": step.metrics.model_name if step.metrics else None,
                        "provider": step.metrics.provider if step.metrics else None,
                        "has_tool_calls": bool(step.tool_calls),
                    },
                    step_id=step.id,
                    run_id=step.run_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=duration_ms,
                    status=SpanStatus.OK,
                    output_preview=step.content[: self.PREVIEW_LENGTH]
                    if step.content
                    else None,
                    llm_details=self._build_llm_details(step),
                )

                if step.metrics:
                    span.metrics = {
                        "tokens.input": step.metrics.input_tokens,
                        "tokens.output": step.metrics.output_tokens,
                        "tokens.total": step.metrics.total_tokens,
                        "tokens.cache_read": step.metrics.cache_read_tokens,
                        "tokens.cache_creation": step.metrics.cache_creation_tokens,
                        "first_token_ms": step.metrics.first_token_latency_ms,
                        "duration_ms": duration_ms,
                        "model": step.metrics.model_name,
                        "provider": step.metrics.provider,
                    }

                if span.end_time:
                    span.complete(status=SpanStatus.OK)

                trace.add_span(span)
                return current_span, updated_cache
            else:
                return current_span, updated_cache

        # === RUN_COMPLETED ===
        elif event_type == StepEventType.RUN_COMPLETED:
            span = span_stack.get(event.run_id)
            if span:
                response = event.data.get("response") if event.data else None
                span.complete(
                    status=SpanStatus.OK,
                    output_preview=response[: self.PREVIEW_LENGTH]
                    if response
                    else None,
                )
                trace.final_output = response
            return span_stack.get(event.run_id), {}

        # === RUN_FAILED ===
        elif event_type == StepEventType.RUN_FAILED:
            span = span_stack.get(event.run_id)
            if span:
                error = event.data.get("error") if event.data else "Unknown error"
                span.complete(status=SpanStatus.ERROR, error_message=error)
            return span_stack.get(event.run_id), {}

        return current_span, {}

    def _find_parent_span_for_nested(
        self,
        event: StepEvent,
        span_stack: dict[str, Span],
        current_span: Span | None,
    ) -> Span | None:
        """
        Find parent span for nested execution.

        Strategy:
        1. Look for parent_run_id in span_stack (parent Agent Span)
        2. Look for current_span (might be a tool call span)
        3. Fallback to root span
        """
        # First, try to find parent Agent Span by parent_run_id
        if event.parent_run_id:
            parent_agent_span = span_stack.get(event.parent_run_id)
            if parent_agent_span:
                return parent_agent_span

        # Second, check if current_span is a tool call span (nested agent called via tool)
        if current_span and current_span.kind == SpanKind.TOOL_CALL:
            return current_span

        # Third, use current_span if available
        if current_span:
            return current_span

        # Fallback: find root span
        for span in span_stack.values():
            if span.depth == 0:
                return span

        return None

    def _build_tool_details(
        self,
        tool_step: Step,
        input_args: dict[str, Any],
    ) -> dict[str, Any]:
        """Build complete tool call details from Tool Step and input arguments."""
        is_error = tool_step.content and tool_step.content.startswith("Error:")

        details: dict[str, Any] = {
            "tool_name": tool_step.name,
            "tool_id": tool_step.name,
            "tool_call_id": tool_step.tool_call_id,
            "input_args": input_args,  # Complete arguments, not truncated
            "output": tool_step.content,  # Complete execution result, not truncated
            "content_for_user": tool_step.content_for_user,
            "error": tool_step.content if is_error else None,
            "status": "error" if is_error else "completed",
        }

        # Add metrics if available
        if tool_step.metrics:
            details["metrics"] = {
                "duration_ms": tool_step.metrics.duration_ms,
            }

        return details

    def _build_llm_details(self, step: Step) -> dict[str, Any]:
        """Build complete LLM call details from Step."""
        if not step.llm_messages:
            return {}

        details: dict[str, Any] = {
            "request": step.llm_request_params or {},
            "messages": step.llm_messages,
            "tools": step.llm_tools,
            "response_content": step.content,
            "response_tool_calls": step.tool_calls,
            "finish_reason": getattr(step, "finish_reason", None),
            "status": "completed",
        }

        # Add metrics if available
        if step.metrics:
            details["metrics"] = {
                "duration_ms": step.metrics.duration_ms,
                "first_token_ms": step.metrics.first_token_latency_ms,
                "input_tokens": step.metrics.input_tokens,
                "output_tokens": step.metrics.output_tokens,
                "total_tokens": step.metrics.total_tokens,
                "cache_read_tokens": step.metrics.cache_read_tokens,
                "cache_creation_tokens": step.metrics.cache_creation_tokens,
            }

        return details


def create_collector(store=None) -> TraceCollector:
    """Create TraceCollector instance"""
    return TraceCollector(store)


__all__ = ["TraceCollector", "create_collector"]
