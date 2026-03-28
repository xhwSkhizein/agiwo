"""Projection helpers for building task summaries from SDK execution facts."""

from server.domain.remote_workspace import WorkspaceTaskSummary


def project_session_task_summary(session, run_steps) -> WorkspaceTaskSummary:
    """Build a task-facing summary from a session record and its associated RunStep records."""
    run_ids = {step.run_id for step in run_steps}
    visible_steps = [
        step for step in run_steps if getattr(step, "content_for_user", None)
    ]
    last_response = visible_steps[-1].content_for_user if visible_steps else None
    status = "completed" if last_response else "idle"
    return WorkspaceTaskSummary(
        session_id=session.id,
        task_id=session.current_task_id,
        message_count=session.task_message_count,
        status=status,
        run_count=len(run_ids),
        last_response=last_response,
        source_session_id=session.source_session_id,
    )
