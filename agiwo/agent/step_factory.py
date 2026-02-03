"""
StepFactory - Context-bound step factory.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from agiwo.agent.schema import MessageRole, Step, StepMetrics
from agiwo.agent.execution_context import ExecutionContext


class StepFactory:
    def __init__(self, ctx: ExecutionContext) -> None:
        self._ctx = ctx

    @property
    def ctx(self) -> ExecutionContext:
        return self._ctx

    def _build_context_attrs(self, overrides: dict[str, Any]) -> dict[str, Any]:
        return {
            "session_id": self._ctx.session_id,
            "run_id": self._ctx.run_id,
            "agent_id": overrides.get("agent_id", self._ctx.agent_id),
            # Extract metadata from ExecutionContext
            "parent_run_id": overrides.get("parent_run_id", self._ctx.parent_run_id),
            # Observability
            "trace_id": overrides.get("trace_id", self._ctx.trace_id),
            "span_id": overrides.get("span_id"),
            "parent_span_id": overrides.get("parent_span_id", self._ctx.span_id),
            "depth": overrides.get("depth", self._ctx.depth),
        }

    def user_step(
        self,
        sequence: int,
        content: str,
        **overrides,
    ) -> Step:
        """Create a USER step."""
        context_attrs = self._build_context_attrs(overrides)
        return Step(
            id=str(uuid4()),
            sequence=sequence,
            role=MessageRole.USER,
            content=content,
            **context_attrs,
        )

    def assistant_step(
        self,
        sequence: int,
        content: str | None = None,
        tool_calls: list[dict] | None = None,
        reasoning_content: str | None = None,
        llm_messages: list[dict] | None = None,
        llm_tools: list[dict] | None = None,
        llm_request_params: dict[str, Any] | None = None,
        metrics: StepMetrics | None = None,
        **overrides,
    ) -> Step:
        """Create an ASSISTANT step."""
        context_attrs = self._build_context_attrs(overrides)
        return Step(
            id=str(uuid4()),
            sequence=sequence,
            role=MessageRole.ASSISTANT,
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            # LLM call context
            llm_messages=llm_messages,
            llm_tools=llm_tools,
            llm_request_params=llm_request_params,
            # Metadata
            metrics=metrics,
            **context_attrs,
        )

    def tool_step(
        self,
        sequence: int,
        tool_call_id: str,
        name: str,
        content: str,
        content_for_user: str | None = None,
        metrics: StepMetrics | None = None,
        **overrides,
    ) -> Step:
        """Create a TOOL step."""
        context_attrs = self._build_context_attrs(overrides)
        return Step(
            id=str(uuid4()),
            sequence=sequence,
            role=MessageRole.TOOL,
            content=content,
            content_for_user=content_for_user,
            tool_call_id=tool_call_id,
            name=name,
            # Metadata
            metrics=metrics,
            **context_attrs,
        )


__all__ = ["StepFactory"]
