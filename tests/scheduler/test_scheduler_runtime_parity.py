from types import SimpleNamespace

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
from agiwo.scheduler.models import AgentState, AgentStateStatus
from agiwo.scheduler.runtime_facts import SchedulerRuntimeFacts
from agiwo.scheduler.runtime_state import RuntimeState


@pytest.mark.asyncio
async def test_scheduler_runtime_facts_match_run_log_replay_views() -> None:
    storage = InMemoryRunLogStorage()
    await storage.append_entries(
        [
            RunStarted(
                sequence=1,
                session_id="sess-1",
                run_id="run-1",
                agent_id="root",
                user_input="hello",
            ),
            UserStepCommitted(
                sequence=2,
                session_id="sess-1",
                run_id="run-1",
                agent_id="root",
                step_id="step-2",
                role=MessageRole.USER,
                content="hello",
                user_input="hello",
            ),
            AssistantStepCommitted(
                sequence=3,
                session_id="sess-1",
                run_id="run-1",
                agent_id="root",
                step_id="step-3",
                role=MessageRole.ASSISTANT,
                content="world",
            ),
            TerminationDecided(
                sequence=4,
                session_id="sess-1",
                run_id="run-1",
                agent_id="root",
                termination_reason=TerminationReason.COMPLETED,
                phase="after_tool_batch",
                source="finished",
            ),
            RunFinished(
                sequence=5,
                session_id="sess-1",
                run_id="run-1",
                agent_id="root",
                response="world",
                termination_reason=TerminationReason.COMPLETED,
            ),
        ]
    )
    state = AgentState(
        id="root",
        session_id="sess-1",
        status=AgentStateStatus.IDLE,
        task="hello",
        is_persistent=True,
    )
    facts = SchedulerRuntimeFacts(
        RuntimeState(agents={"root": SimpleNamespace(run_log_storage=storage)})
    )

    latest_run = await facts.get_latest_run_view(state)
    step_views = await facts.list_step_views(state)
    decisions = await facts.get_runtime_decision_state(state)

    assert latest_run == await storage.get_latest_run_view("sess-1")
    assert step_views == await storage.list_step_views(
        session_id="sess-1",
        agent_id="root",
    )
    assert decisions == await storage.get_runtime_decision_state(
        session_id="sess-1",
        agent_id="root",
    )
