"""
EventFactory - Context-bound event factory.
"""

from agiwo.agent.schema import (
    Step,
    StepDelta,
    StepEventType,
    StepEvent,
)
from agiwo.agent.execution_context import ExecutionContext

# ============================================================================
# Event Factory Functions
# ============================================================================


def create_run_started_event(
    run_id: str,
    query: str,
    session_id: str,
    *,
    depth: int = 0,
    parent_run_id: str | None = None,
    agent_id: str | None = None,
) -> StepEvent:
    """Create a RUN_STARTED event with optional nested context"""
    return StepEvent(
        type=StepEventType.RUN_STARTED,
        run_id=run_id,
        data={"query": query, "session_id": session_id},
        depth=depth,
        parent_run_id=parent_run_id,
        agent_id=agent_id,
    )


def create_run_completed_event(
    run_id: str,
    response: str,
    metrics: dict,
    termination_reason: str | None = None,
    max_steps: int | None = None,
    *,
    depth: int = 0,
    parent_run_id: str | None = None,
    agent_id: str | None = None,
) -> StepEvent:
    """Create a RUN_COMPLETED event with optional nested context"""
    data = {"response": response, "metrics": metrics}
    if termination_reason:
        data["termination_reason"] = termination_reason
    if max_steps:
        data["max_steps"] = max_steps
    return StepEvent(
        type=StepEventType.RUN_COMPLETED,
        run_id=run_id,
        data=data,
        depth=depth,
        parent_run_id=parent_run_id,
        agent_id=agent_id,
    )


def create_run_failed_event(
    run_id: str,
    error: str,
    *,
    depth: int = 0,
    parent_run_id: str | None = None,
    agent_id: str | None = None,
) -> StepEvent:
    """Create a RUN_FAILED event with optional nested context"""
    return StepEvent(
        type=StepEventType.RUN_FAILED,
        run_id=run_id,
        data={"error": error},
        depth=depth,
        parent_run_id=parent_run_id,
        agent_id=agent_id,
    )


def create_step_delta_event(
    step_id: str,
    run_id: str,
    delta: StepDelta,
    *,
    depth: int = 0,
    parent_run_id: str | None = None,
    agent_id: str | None = None,
) -> StepEvent:
    """Create a STEP_DELTA event with optional nested context"""
    return StepEvent(
        type=StepEventType.STEP_DELTA,
        step_id=step_id,
        run_id=run_id,
        delta=delta,
        depth=depth,
        parent_run_id=parent_run_id,
        agent_id=agent_id,
    )


def create_step_completed_event(
    step_id: str,
    run_id: str,
    snapshot: Step,
    *,
    depth: int = 0,
    parent_run_id: str | None = None,
    agent_id: str | None = None,
) -> StepEvent:
    """Create a STEP_COMPLETED event with optional nested context"""
    return StepEvent(
        type=StepEventType.STEP_COMPLETED,
        step_id=step_id,
        run_id=run_id,
        snapshot=snapshot,
        depth=depth,
        parent_run_id=parent_run_id,
        agent_id=agent_id,
    )


def create_error_event(
    run_id: str, error: str, error_type: str = "unknown"
) -> StepEvent:
    """Create an ERROR event"""
    return StepEvent(
        type=StepEventType.ERROR,
        run_id=run_id,
        data={"error": error, "error_type": error_type},
    )


#############
## 事件工厂类
#############


class EventFactory:
    def __init__(self, ctx: "ExecutionContext") -> None:
        """
        Initialize factory with execution context.

        Args:
            ctx: ExecutionContext to bind to
        """
        self._ctx = ctx

    @property
    def ctx(self) -> "ExecutionContext":
        """Get the bound execution context."""
        return self._ctx

    def run_started(self, query: str) -> StepEvent:
        """Create a RUN_STARTED event."""
        return create_run_started_event(
            run_id=self._ctx.run_id,
            query=query,
            session_id=self._ctx.session_id,
            depth=self._ctx.depth,
            parent_run_id=self._ctx.parent_run_id,
            agent_id=self._ctx.agent_id,
        )

    def run_completed(
        self,
        response: str,
        metrics: dict,
        termination_reason: str | None = None,
        max_steps: int | None = None,
    ) -> StepEvent:
        """Create a RUN_COMPLETED event."""
        return create_run_completed_event(
            run_id=self._ctx.run_id,
            response=response,
            metrics=metrics,
            termination_reason=termination_reason,
            max_steps=max_steps,
            depth=self._ctx.depth,
            parent_run_id=self._ctx.parent_run_id,
            agent_id=self._ctx.agent_id,
        )

    def run_failed(self, error: str) -> StepEvent:
        """Create a RUN_FAILED event."""
        return create_run_failed_event(
            run_id=self._ctx.run_id,
            error=error,
            depth=self._ctx.depth,
            parent_run_id=self._ctx.parent_run_id,
            agent_id=self._ctx.agent_id,
        )

    def step_delta(self, step_id: str, delta: StepDelta) -> StepEvent:
        """Create a STEP_DELTA event."""
        return create_step_delta_event(
            step_id=step_id,
            run_id=self._ctx.run_id,
            delta=delta,
            depth=self._ctx.depth,
            parent_run_id=self._ctx.parent_run_id,
            agent_id=self._ctx.agent_id,
        )

    def step_completed(self, step_id: str, snapshot: Step) -> StepEvent:
        """Create a STEP_COMPLETED event."""
        return create_step_completed_event(
            step_id=step_id,
            run_id=self._ctx.run_id,
            snapshot=snapshot,
            depth=self._ctx.depth,
            parent_run_id=self._ctx.parent_run_id,
            agent_id=self._ctx.agent_id,
        )

    def error(self, error: str, error_type: str = "unknown") -> StepEvent:
        """Create an ERROR event."""
        return create_error_event(
            run_id=self._ctx.run_id,
            error=error,
            error_type=error_type,
        )


__all__ = ["EventFactory"]
