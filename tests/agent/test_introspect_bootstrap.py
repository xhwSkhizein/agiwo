import tempfile
from typing import cast

import pytest

from agiwo.agent.introspect.models import Milestone
from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.log import (
    CompactionApplied,
    GoalMilestonesUpdated,
    IntrospectionCheckpointRecorded,
    MessagesRebuilt,
    ToolStepCommitted,
)
from agiwo.agent.models.run import RunIdentity
from agiwo.agent.models.step import MessageRole
from agiwo.agent.run_bootstrap import prepare_run_context
from agiwo.agent.runtime.context import RunContext, RunRuntime
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.runtime.state_writer import RunStateWriter
from agiwo.agent.storage.base import InMemoryRunLogStorage
from agiwo.utils.abort_signal import AbortSignal


@pytest.mark.asyncio
async def test_prepare_run_context_restores_introspect_state_from_run_log() -> None:
    storage = InMemoryRunLogStorage()
    await storage.append_entries(
        [
            GoalMilestonesUpdated(
                sequence=1,
                session_id="sess-1",
                run_id="run-old",
                agent_id="agent-1",
                milestones=[
                    Milestone(id="inspect", description="Inspect", status="active")
                ],
                active_milestone_id="inspect",
                reason="declared",
            ),
            IntrospectionCheckpointRecorded(
                sequence=2,
                session_id="sess-1",
                run_id="run-old",
                agent_id="agent-1",
                checkpoint_seq=2,
                milestone_id="inspect",
                review_tool_call_id="tc-review",
                review_step_id="step-review",
            ),
            ToolStepCommitted(
                sequence=3,
                session_id="sess-1",
                run_id="run-old",
                agent_id="agent-1",
                step_id="step-search",
                role=MessageRole.TOOL,
                tool_call_id="tc-search",
                name="web_search",
                content="results",
            ),
        ]
    )
    context = RunContext(
        identity=RunIdentity(
            run_id="run-new",
            agent_id="agent-1",
            agent_name="agent",
        ),
        session_runtime=SessionRuntime(
            session_id="sess-1",
            run_log_storage=storage,
        ),
    )
    context.config = AgentOptions()
    runtime = RunRuntime(
        session_runtime=context.session_runtime,
        config=context.config,
        hooks=context.hooks,
        model=cast(object, None),
        tools_map={},
        abort_signal=AbortSignal(),
        root_path=tempfile.gettempdir(),
        compact_start_seq=0,
        max_input_tokens_per_call=1000,
        max_context_window=None,
        compact_prompt=None,
    )

    await prepare_run_context(
        context=context,
        runtime=runtime,
        user_input="continue",
        system_prompt="sys",
        writer=RunStateWriter(context),
    )

    assert [milestone.id for milestone in context.ledger.goal.milestones] == ["inspect"]
    assert context.ledger.goal.active_milestone_id == "inspect"
    assert context.ledger.introspection.latest_aligned_checkpoint is not None
    assert context.ledger.introspection.latest_aligned_checkpoint.seq == 2
    assert context.ledger.introspection.last_boundary_seq == 2
    assert context.ledger.introspection.review_count_since_boundary == 1


@pytest.mark.asyncio
async def test_prepare_run_context_restores_latest_rebuilt_messages_after_compaction() -> (
    None
):
    storage = InMemoryRunLogStorage()
    await storage.append_entries(
        [
            MessagesRebuilt(
                sequence=1,
                session_id="sess-1",
                run_id="run-old",
                agent_id="agent-1",
                reason="compaction",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "compressed history"},
                    {"role": "assistant", "content": "Understood."},
                ],
            ),
            CompactionApplied(
                sequence=2,
                session_id="sess-1",
                run_id="run-old",
                agent_id="agent-1",
                start_sequence=0,
                end_sequence=1,
                before_token_estimate=1000,
                after_token_estimate=100,
                message_count=3,
                transcript_path="/tmp/compaction-transcript.jsonl",
                analysis={"summary": "compressed history"},
            ),
        ]
    )
    context = RunContext(
        identity=RunIdentity(
            run_id="run-new",
            agent_id="agent-1",
            agent_name="agent",
        ),
        session_runtime=SessionRuntime(
            session_id="sess-1",
            run_log_storage=storage,
        ),
    )
    context.config = AgentOptions()
    runtime = RunRuntime(
        session_runtime=context.session_runtime,
        config=context.config,
        hooks=context.hooks,
        model=cast(object, None),
        tools_map={},
        abort_signal=AbortSignal(),
        root_path=tempfile.gettempdir(),
        compact_start_seq=0,
        max_input_tokens_per_call=1000,
        max_context_window=None,
        compact_prompt=None,
    )

    await prepare_run_context(
        context=context,
        runtime=runtime,
        user_input="next question",
        system_prompt="sys",
        writer=RunStateWriter(context),
    )

    assert context.ledger.messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "compressed history"},
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "_sequence": 3, "content": "next question"},
    ]
