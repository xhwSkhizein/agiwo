from tests.utils.agent_context import build_agent_context


def test_child_context_reuses_session_runtime_and_increments_depth() -> None:
    parent = build_agent_context(
        session_id="child-session",
        run_id="root-run",
        agent_id="root-agent",
        agent_name="root-agent",
        metadata={"scope": "root"},
    )

    child = parent.new_child(
        agent_id="child-agent",
        agent_name="child-agent",
    )

    assert child.session_runtime is parent.session_runtime
    assert child.session_id == parent.session_id
    assert child.parent_run_id == parent.run_id
    assert child.depth == parent.depth + 1
    assert child.metadata == parent.metadata
    assert child.metadata is not parent.metadata
