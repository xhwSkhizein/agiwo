"""Stream event serialization utilities."""

from typing import Any

from agiwo.agent import AgentStreamItem, StepCompletedEvent
from server.models.step_run import StepResponse


def stream_event_to_payload(event: AgentStreamItem) -> dict[str, Any]:
    """Convert an AgentStreamItem to a SSE payload dictionary."""
    if isinstance(event, StepCompletedEvent):
        return {
            "type": "step",
            "step": StepResponse.from_sdk(event.step).model_dump(),
        }
    return {"type": "unknown", "data": str(event)}
