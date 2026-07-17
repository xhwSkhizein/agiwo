import pytest

from agiwo.agent import (
    AssistantStepCommitted,
    MessageRole,
    RunFinished,
    RunStarted,
    TerminationDecided,
    TerminationReason,
    UserStepCommitted,
)
from agiwo.agent.storage.base import InMemoryRunLogStorage

from server.services.runtime.run_query_service import RunQueryService


@pytest.mark.asyncio
async def test_console_run_query_service_matches_run_log_replay_views() -> None:
    storage = InMemoryRunLogStorage()
    await storage.append_entries(
        [
            RunStarted(
                sequence=1,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                user_input="hello",
            ),
            UserStepCommitted(
                sequence=2,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                step_id="step-2",
                role=MessageRole.USER,
                content="hello",
                user_input="hello",
            ),
            AssistantStepCommitted(
                sequence=3,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                step_id="step-3",
                role=MessageRole.ASSISTANT,
                content="world",
            ),
            TerminationDecided(
                sequence=4,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                termination_reason=TerminationReason.COMPLETED,
                phase="after_tool_batch",
                source="finished",
            ),
            RunFinished(
                sequence=5,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                response="world",
                termination_reason=TerminationReason.COMPLETED,
            ),
        ]
    )
    service = RunQueryService(run_storage=storage)

    run_page = await service.list_runs(session_id="sess-1", limit=20, offset=0)
    step_page = await service.list_session_steps("sess-1", limit=20, order="asc")
    decision_state = await service.get_runtime_decision_state("sess-1")

    assert run_page.items == await storage.list_run_views(session_id="sess-1", limit=20)
    assert step_page.items == await storage.list_step_views(
        session_id="sess-1", limit=20
    )
    assert decision_state == await storage.get_runtime_decision_state(
        session_id="sess-1"
    )
