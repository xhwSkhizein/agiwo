from collections.abc import AsyncIterator
from typing import Any

import pytest

from agiwo.agent import Agent, AgentConfig, TerminationReason
from agiwo.agent.models.log import AssistantStepCommitted, ContextStepsHidden
from agiwo.agent.models.step import MessageRole
from agiwo.agent.models.stream import stream_items_from_entries
from agiwo.llm.base import Model, StreamChunk

_REPLAYABLE_TYPES = {
    "run_started",
    "step_completed",
    "context_steps_hidden",
    "messages_rebuilt",
    "compaction_applied",
    "step_back_applied",
    "termination_decided",
    "run_rolled_back",
    "run_completed",
    "run_failed",
}


class _FixedResponseModel(Model):
    def __init__(self, response: str = "ok") -> None:
        super().__init__(id="fixed-model", name="fixed-model", temperature=0.0)
        self._response = response

    async def arun_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        del messages, tools
        yield StreamChunk(content=self._response)
        yield StreamChunk(finish_reason="stop")


def test_context_steps_hidden_emits_public_stream_event_with_step_ids() -> None:
    entries = [
        AssistantStepCommitted(
            sequence=1,
            session_id="sess-1",
            run_id="run-1",
            agent_id="agent-1",
            step_id="step-review-call",
            role=MessageRole.ASSISTANT,
            content="Trajectory review: aligned=True.",
        ),
        ContextStepsHidden(
            sequence=2,
            session_id="sess-1",
            run_id="run-1",
            agent_id="agent-1",
            step_ids=["step-review-call"],
            reason="review_metadata",
        ),
    ]

    items = stream_items_from_entries(entries)

    assert [item.type for item in items] == ["context_steps_hidden"]
    assert items[0].step_ids == ["step-review-call"]


def test_stream_replay_persists_hidden_step_ids_across_pages() -> None:
    hidden_step_ids: set[str] = set()

    first_page = stream_items_from_entries(
        [
            ContextStepsHidden(
                sequence=2,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                step_ids=["step-review-call"],
                reason="review_metadata",
            ),
        ],
        persisted_hidden_step_ids=hidden_step_ids,
    )
    second_page = stream_items_from_entries(
        [
            AssistantStepCommitted(
                sequence=1,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                step_id="step-review-call",
                role=MessageRole.ASSISTANT,
                content="Trajectory review: aligned=True.",
            ),
        ],
        persisted_hidden_step_ids=hidden_step_ids,
    )

    assert [item.type for item in first_page] == ["context_steps_hidden"]
    assert second_page == []


@pytest.mark.asyncio
async def test_live_stream_matches_replayed_run_log() -> None:
    agent = Agent(
        AgentConfig(name="parity-test", description="replay parity test"),
        model=_FixedResponseModel(response="done"),
    )
    try:
        live_items = [
            item
            async for item in agent.run_stream("hello", session_id="sess-parity")
            if item.type in _REPLAYABLE_TYPES
        ]
        entries = await agent.run_log_storage.list_entries(session_id="sess-parity")
        replayed_items = [
            item
            for item in stream_items_from_entries(entries)
            if item.type in _REPLAYABLE_TYPES
        ]

        assert [item.type for item in live_items] == [
            item.type for item in replayed_items
        ]

        live_steps = [item.step for item in live_items if item.type == "step_completed"]
        replayed_steps = [
            item.step for item in replayed_items if item.type == "step_completed"
        ]
        assert [(step.role.value, step.sequence) for step in live_steps] == [
            (step.role.value, step.sequence) for step in replayed_steps
        ]

        live_termination = next(
            item for item in live_items if item.type == "termination_decided"
        )
        replayed_termination = next(
            item for item in replayed_items if item.type == "termination_decided"
        )
        assert live_termination.termination_reason is TerminationReason.COMPLETED
        assert (
            live_termination.termination_reason
            == replayed_termination.termination_reason
        )

        live_completed = next(
            item for item in live_items if item.type == "run_completed"
        )
        replayed_completed = next(
            item for item in replayed_items if item.type == "run_completed"
        )
        assert live_completed.response == replayed_completed.response == "done"
    finally:
        await agent.close()
