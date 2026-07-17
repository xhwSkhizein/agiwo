"""Agent introspection subsystem."""

from agiwo.agent.introspect.models import (
    IntrospectionCheckpoint,
    IntrospectionNotice,
    IntrospectionOutcome,
    IntrospectionState,
    Milestone,
    PendingIntrospectionNotice,
    ToolUsefulnessEntry,
)
from agiwo.agent.models.plan import RunPlan, RunPlanUpdate


def __getattr__(name: str) -> object:
    if name == "ReviewTrajectoryTool":
        from agiwo.agent.introspect.tool import (  # noqa: PLC0415
            ReviewTrajectoryTool,
        )

        return ReviewTrajectoryTool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "IntrospectionCheckpoint",
    "IntrospectionNotice",
    "IntrospectionOutcome",
    "IntrospectionState",
    "Milestone",
    "PendingIntrospectionNotice",
    "ReviewTrajectoryTool",
    "RunPlan",
    "RunPlanUpdate",
    "ToolUsefulnessEntry",
]
