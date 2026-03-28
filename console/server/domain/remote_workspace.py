"""Console-side domain models for session-first workspace semantics."""

from dataclasses import dataclass


@dataclass
class WorkspaceTaskSummary:
    """Default task-facing summary projected from session records + RunStep-backed execution facts."""

    session_id: str
    task_id: str | None
    message_count: int
    status: str
    run_count: int
    last_response: str | None
    source_session_id: str | None

    def to_default_view(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "message_count": self.message_count,
            "status": self.status,
            "run_count": self.run_count,
            "last_response": self.last_response,
            "source_session_id": self.source_session_id,
        }
