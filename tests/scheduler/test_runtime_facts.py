from types import SimpleNamespace

import pytest

from agiwo.agent import (
    AssistantStepCommitted,
    CompactionApplied,
    MessageRole,
    RunFinished,
    RunStarted,
    StepBackApplied,
    TerminationDecided,
    TerminationReason,
    UserStepCommitted,
)
from agiwo.agent.storage.base import InMemoryRunLogStorage
from agiwo.scheduler.models import AgentState, AgentStateStatus
from agiwo.scheduler.runtime_facts import SchedulerRuntimeFacts
from agiwo.scheduler.runtime_state import RuntimeState


@pytest.mark.asyncio
async def test_runtime_facts_reads_latest_run_view_from_runtime_agent():
    storage = InMemoryRunLogStorage()
    state = AgentState(
        id="root",
        session_id="sess-1",
        status=AgentStateStatus.IDLE,
        task="work",
        is_persistent=True,
    )
    await storage.append_entries(
        [
            RunStarted(
                sequence=1,
                session_id="sess-1",
                run_id="run-1",
                agent_id="root",
                user_input="hello",
            ),
            RunFinished(
                sequence=2,
                session_id="sess-1",
                run_id="run-1",
                agent_id="root",
                response="fresh response",
                termination_reason=TerminationReason.COMPLETED,
            ),
        ]
    )
    facts = SchedulerRuntimeFacts(
        RuntimeState(agents={"root": SimpleNamespace(run_log_storage=storage)})
    )

    run_view = await facts.get_latest_run_view(state)

    assert run_view is not None
    assert run_view.run_id == "run-1"
    assert run_view.response == "fresh response"


@pytest.mark.asyncio
async def test_runtime_facts_hide_rolled_back_steps_by_default():
    storage = InMemoryRunLogStorage()
    state = AgentState(
        id="child-1",
        session_id="sess-parent",
        status=AgentStateStatus.WAITING,
        task="work",
        parent_id="root",
    )
    await storage.append_entries(
        [
            UserStepCommitted(
                sequence=10,
                session_id="child-1",
                run_id="run-1",
                agent_id="child-1",
                step_id="step-10",
                role=MessageRole.USER,
                content="u1",
                user_input="u1",
            ),
            AssistantStepCommitted(
                sequence=11,
                session_id="child-1",
                run_id="run-1",
                agent_id="child-1",
                step_id="step-11",
                role=MessageRole.ASSISTANT,
                content="a1",
            ),
        ]
    )
    await storage.append_run_rollback(
        session_id="child-1",
        run_id="run-1",
        agent_id="child-1",
        start_sequence=10,
        end_sequence=11,
        reason="scheduler_no_progress_periodic",
    )
    facts = SchedulerRuntimeFacts(
        RuntimeState(agents={"child-1": SimpleNamespace(run_log_storage=storage)})
    )

    visible_steps = await facts.list_step_views(state)
    all_steps = await facts.list_step_views(state, include_rolled_back=True)

    assert visible_steps == []
    assert [step.id for step in all_steps] == ["step-10", "step-11"]


@pytest.mark.asyncio
async def test_runtime_facts_reads_runtime_decision_state_from_runtime_agent():
    storage = InMemoryRunLogStorage()
    state = AgentState(
        id="root",
        session_id="sess-1",
        status=AgentStateStatus.IDLE,
        task="work",
        is_persistent=True,
    )
    await storage.append_entries(
        [
            CompactionApplied(
                sequence=1,
                session_id="sess-1",
                run_id="run-1",
                agent_id="root",
                start_sequence=1,
                end_sequence=5,
                before_token_estimate=800,
                after_token_estimate=200,
                message_count=3,
                transcript_path="/tmp/transcript.json",
                summary="compacted",
            ),
            StepBackApplied(
                sequence=2,
                session_id="sess-1",
                run_id="run-1",
                agent_id="root",
                affected_count=2,
                checkpoint_seq=5,
                experience="switch plan",
            ),
            TerminationDecided(
                sequence=3,
                session_id="sess-1",
                run_id="run-1",
                agent_id="root",
                termination_reason=TerminationReason.MAX_STEPS,
                phase="before_termination",
                source="limit",
            ),
        ]
    )
    facts = SchedulerRuntimeFacts(
        RuntimeState(agents={"root": SimpleNamespace(run_log_storage=storage)})
    )

    decision_state = await facts.get_runtime_decision_state(state)

    assert decision_state.latest_compaction is not None
    assert decision_state.latest_compaction.summary == "compacted"
    assert decision_state.latest_step_back is not None
    assert decision_state.latest_step_back.experience == "switch plan"
    assert decision_state.latest_termination is not None
    assert decision_state.latest_termination.reason is TerminationReason.MAX_STEPS
