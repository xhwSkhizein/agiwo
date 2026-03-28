"""Unit tests for task/run projection from SDK execution facts."""

from types import SimpleNamespace

from server.domain.remote_workspace import WorkspaceTaskSummary
from server.services.task_projection import project_session_task_summary


def test_project_session_task_summary_uses_run_steps_as_source_of_truth() -> None:
    session = SimpleNamespace(
        id="sess-1",
        current_task_id="task-1",
        task_message_count=2,
        source_session_id=None,
    )
    run_steps = [
        SimpleNamespace(session_id="sess-1", run_id="run-1", content_for_user="thinking"),
        SimpleNamespace(session_id="sess-1", run_id="run-1", content_for_user="done"),
    ]

    summary = project_session_task_summary(session, run_steps)

    assert summary.session_id == "sess-1"
    assert summary.task_id == "task-1"
    assert summary.message_count == 2
    assert summary.last_response == "done"
    assert summary.run_count == 1


def test_project_session_task_summary_idle_when_no_visible_steps() -> None:
    session = SimpleNamespace(
        id="sess-1",
        current_task_id=None,
        task_message_count=0,
        source_session_id=None,
    )

    summary = project_session_task_summary(session, [])

    assert summary.status == "idle"
    assert summary.last_response is None
    assert summary.run_count == 0


def test_workspace_task_summary_to_default_view() -> None:
    summary = WorkspaceTaskSummary(
        session_id="sess-1",
        task_id="task-1",
        message_count=1,
        status="completed",
        run_count=2,
        last_response="done",
        source_session_id=None,
    )

    payload = summary.to_default_view()

    assert payload == {
        "session_id": "sess-1",
        "task_id": "task-1",
        "message_count": 1,
        "status": "completed",
        "run_count": 2,
        "last_response": "done",
        "source_session_id": None,
    }


def test_project_session_task_summary_multiple_runs() -> None:
    session = SimpleNamespace(
        id="sess-1",
        current_task_id="task-1",
        task_message_count=3,
        source_session_id="sess-0",
    )
    run_steps = [
        SimpleNamespace(session_id="sess-1", run_id="run-1", content_for_user="first"),
        SimpleNamespace(session_id="sess-1", run_id="run-2", content_for_user="second"),
        SimpleNamespace(session_id="sess-1", run_id="run-2", content_for_user=None),
    ]

    summary = project_session_task_summary(session, run_steps)

    assert summary.run_count == 2
    assert summary.last_response == "second"
    assert summary.source_session_id == "sess-0"
