from collections.abc import AsyncIterator

import pytest

from agiwo.agent import Agent, AgentConfig, TerminationReason
from agiwo.agent.models.stream import stream_items_from_entries
from agiwo.llm.base import Model, StreamChunk

_REPLAYABLE_TYPES = {
    "run_started",
    "step_completed",
    "messages_rebuilt",
    "compaction_applied",
    "retrospect_applied",
    "termination_decided",
    "run_rolled_back",
    "run_completed",
    "run_failed",
}


class _FixedResponseModel(Model):
    def __init__(self, response: str = "ok") -> None:
        super().__init__(id="fixed-model", name="fixed-model", temperature=0.0)
        self._response = response

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        yield StreamChunk(content=self._response)
        yield StreamChunk(finish_reason="stop")


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
