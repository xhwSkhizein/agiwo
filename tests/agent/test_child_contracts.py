from uuid import uuid4

from agiwo.agent.models.run import RunIdentity
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.storage.base import InMemoryRunLogStorage


def test_child_context_reuses_session_runtime_and_increments_depth() -> None:
    parent = RunContext(
        identity=RunIdentity(
            run_id="root-run",
            agent_id="root-agent",
            agent_name="root-agent",
            metadata={"scope": "root"},
        ),
        session_runtime=SessionRuntime(
            session_id="child-session",
            run_log_storage=InMemoryRunLogStorage(),
        ),
    )

    child = parent.__class__(
        identity=RunIdentity(
            run_id=str(uuid4()),
            agent_id="child-agent",
            agent_name="child-agent",
            user_id=parent.user_id,
            depth=parent.depth + 1,
            parent_run_id=parent.run_id,
            timeout_at=parent.timeout_at,
            metadata=dict(parent.metadata),
        ),
        session_runtime=parent.session_runtime,
    )

    assert child.session_runtime is parent.session_runtime
    assert child.session_id == parent.session_id
    assert child.parent_run_id == parent.run_id
    assert child.depth == parent.depth + 1
    assert child.metadata == parent.metadata
    assert child.metadata is not parent.metadata


def test_run_context_update_metadata_replaces_identity_metadata() -> None:
    context = RunContext(
        identity=RunIdentity(
            run_id="root-run",
            agent_id="root-agent",
            agent_name="root-agent",
            metadata={"scope": "root"},
        ),
        session_runtime=SessionRuntime(
            session_id="child-session",
            run_log_storage=InMemoryRunLogStorage(),
        ),
    )

    context.update_metadata({"child": "nested"})

    assert context.metadata == {"scope": "root", "child": "nested"}
