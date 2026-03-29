"""Console domain models for aggregated session summaries."""

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field

from agiwo.agent import UserInput
from agiwo.agent.models.run import Run
from agiwo.agent.models.input import UserMessage
from agiwo.utils.serialization import serialize_optional_datetime

from server.domain.run_metrics import RunMetricsSummary


@dataclass
class SessionAggregate:
    session_id: str
    agent_id: str | None
    last_run: Run | None
    metrics: RunMetricsSummary
    created_at: datetime | None
    updated_at: datetime | None


class SessionSummaryData(BaseModel):
    session_id: str
    agent_id: str | None = None
    last_user_input: UserInput | None = None
    last_response: str | None = None
    run_count: int = 0
    step_count: int = 0
    metrics: RunMetricsSummary = Field(default_factory=RunMetricsSummary)
    created_at: str | None = None
    updated_at: str | None = None


def session_aggregate_to_summary_data(session: SessionAggregate) -> SessionSummaryData:
    last_run = session.last_run
    last_user_input = (
        UserMessage.to_transport_payload(last_run.user_input)
        if last_run is not None
        else None
    )
    return SessionSummaryData(
        session_id=session.session_id,
        agent_id=session.agent_id,
        last_user_input=last_user_input,
        last_response=(
            last_run.response_content[:200]
            if last_run and last_run.response_content
            else None
        ),
        run_count=session.metrics.run_count,
        step_count=session.metrics.step_count,
        metrics=session.metrics,
        created_at=serialize_optional_datetime(session.created_at),
        updated_at=serialize_optional_datetime(session.updated_at),
    )


def session_aggregate_to_chat_summary(session: SessionAggregate) -> dict[str, object]:
    summary = session_aggregate_to_summary_data(session)
    last_input = None
    if session.last_run is not None:
        raw = UserMessage.to_storage_value(session.last_run.user_input)
        if isinstance(raw, str) and raw:
            last_input = raw[:200]
    return {
        "session_id": summary.session_id,
        "run_count": summary.run_count,
        "last_input": last_input,
        "last_response": summary.last_response,
        "updated_at": summary.updated_at,
    }


__all__ = [
    "SessionAggregate",
    "SessionSummaryData",
    "session_aggregate_to_chat_summary",
    "session_aggregate_to_summary_data",
]
