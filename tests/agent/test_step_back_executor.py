# tests/agent/test_step_back_executor.py
import pytest
from unittest.mock import AsyncMock
from agiwo.agent.models.log import StepCondensedContentUpdated, ToolStepCommitted
from agiwo.agent.models.step import MessageRole
from agiwo.agent.review.step_back_executor import (
    StepBackOutcome,
    execute_step_back,
)
from agiwo.agent.storage.base import InMemoryRunLogStorage


def _make_messages():
    """Build a realistic message list for testing."""
    return [
        {"role": "system", "content": "You are an agent", "_sequence": 0},
        {"role": "user", "content": "Fix the bug", "_sequence": 1},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q":"session"}'},
                }
            ],
            "_sequence": 2,
        },
        {
            "role": "tool",
            "tool_call_id": "tc_1",
            "content": "Found SessionManager in auth.py",
            "_sequence": 3,
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc_2",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q":"jwt"}'},
                }
            ],
            "_sequence": 4,
        },
        {
            "role": "tool",
            "tool_call_id": "tc_2",
            "content": "Found 15 JWT references",
            "_sequence": 5,
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc_review",
                    "type": "function",
                    "function": {
                        "name": "review_trajectory",
                        "arguments": '{"aligned":false,"experience":"JWT search was off-track"}',
                    },
                }
            ],
            "_sequence": 6,
        },
        {
            "role": "tool",
            "tool_call_id": "tc_review",
            "content": "Review acknowledged",
            "_sequence": 7,
        },
    ]


class TestExecuteStepBack:
    @pytest.mark.asyncio
    async def test_condenses_tool_results_after_checkpoint(self):
        messages = _make_messages()
        storage = AsyncMock()
        storage.append_step_condensed_content = AsyncMock(return_value=True)

        outcome = await execute_step_back(
            messages=messages,
            checkpoint_seq=3,
            experience="JWT search was off-track. Token validation lives in auth.py.",
            step_lookup={
                "tc_1": {"id": "step_1", "sequence": 3},
                "tc_2": {"id": "step_2", "sequence": 5},
            },
            storage=storage,
            session_id="s1",
            run_id="r1",
            agent_id="a1",
        )

        assert outcome.applied is True
        assert outcome.affected_count == 1  # only tc_2 (seq 5 > checkpoint 3)
        assert outcome.checkpoint_seq == 3

        # tc_1 (seq 3) is at or before checkpoint — should be unchanged
        assert "Found SessionManager" in messages[3]["content"]

        # tc_2 (seq 5) is after checkpoint — should be condensed
        assert "[EXPERIENCE]" in messages[5]["content"]
        assert "off-track" in messages[5]["content"]

    @pytest.mark.asyncio
    async def test_preserves_tool_calls(self):
        messages = _make_messages()
        storage = AsyncMock()
        storage.append_step_condensed_content = AsyncMock(return_value=True)

        await execute_step_back(
            messages=messages,
            checkpoint_seq=2,
            experience="All steps were off-track.",
            step_lookup={
                "tc_1": {"id": "step_1", "sequence": 3},
                "tc_2": {"id": "step_2", "sequence": 5},
            },
            storage=storage,
            session_id="s1",
            run_id="r1",
            agent_id="a1",
        )

        # assistant messages with tool_calls are preserved
        assert messages[2]["tool_calls"][0]["function"]["name"] == "search"
        assert messages[4]["tool_calls"][0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_removes_review_trajectory_messages(self):
        messages = _make_messages()
        storage = AsyncMock()
        storage.append_step_condensed_content = AsyncMock(return_value=True)

        outcome = await execute_step_back(
            messages=messages,
            checkpoint_seq=2,
            experience="Done.",
            step_lookup={
                "tc_1": {"id": "step_1", "sequence": 3},
                "tc_2": {"id": "step_2", "sequence": 5},
            },
            storage=storage,
            session_id="s1",
            run_id="r1",
            agent_id="a1",
        )

        # No review_trajectory references remain
        for msg in outcome.messages:
            if msg.get("role") == "tool":
                assert msg.get("tool_call_id") != "tc_review"
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    assert tc["function"]["name"] != "review_trajectory"

    @pytest.mark.asyncio
    async def test_no_op_when_no_tool_results_after_checkpoint(self):
        messages = [
            {"role": "user", "content": "hello", "_sequence": 1},
            {"role": "assistant", "content": "hi", "_sequence": 2},
        ]
        storage = AsyncMock()

        outcome = await execute_step_back(
            messages=messages,
            checkpoint_seq=5,
            experience="Nothing to condense.",
            step_lookup={},
            storage=storage,
            session_id="s1",
            run_id="r1",
            agent_id="a1",
        )

        assert outcome.applied is True
        assert outcome.affected_count == 0
        assert len(outcome.messages) == len(messages)

    @pytest.mark.asyncio
    async def test_persists_condensed_content_to_storage(self):
        messages = _make_messages()
        storage = AsyncMock()
        storage.append_step_condensed_content = AsyncMock(return_value=True)

        await execute_step_back(
            messages=messages,
            checkpoint_seq=2,
            experience="Condensed.",
            step_lookup={
                "tc_1": {"id": "step_1", "sequence": 3},
                "tc_2": {"id": "step_2", "sequence": 5},
            },
            storage=storage,
            session_id="s1",
            run_id="r1",
            agent_id="a1",
        )

        # Should have called append_step_condensed_content for affected steps
        assert storage.append_step_condensed_content.call_count >= 2

    @pytest.mark.asyncio
    async def test_persists_condensed_content_for_prior_batch_tool_result(self):
        messages = [
            {
                "role": "tool",
                "tool_call_id": "tc_old",
                "content": "Verbose old result",
                "_sequence": 5,
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "tc_review",
                        "type": "function",
                        "function": {"name": "review_trajectory", "arguments": "{}"},
                    }
                ],
                "_sequence": 6,
            },
            {
                "role": "tool",
                "tool_call_id": "tc_review",
                "content": "Review acknowledged",
                "_sequence": 7,
            },
        ]
        storage = InMemoryRunLogStorage()
        await storage.append_entries(
            [
                ToolStepCommitted(
                    sequence=5,
                    session_id="s1",
                    run_id="r1",
                    agent_id="a1",
                    step_id="step_old",
                    role=MessageRole.TOOL,
                    content="Verbose old result",
                    tool_call_id="tc_old",
                    name="search",
                )
            ]
        )

        await execute_step_back(
            messages=messages,
            checkpoint_seq=3,
            experience="Old search was off-track.",
            review_tool_call_id="tc_review",
            step_lookup={},
            storage=storage,
            session_id="s1",
            run_id="r1",
            agent_id="a1",
        )

        entries = await storage.list_entries(session_id="s1")
        condensed_entries = [
            entry for entry in entries if isinstance(entry, StepCondensedContentUpdated)
        ]
        assert len(condensed_entries) == 1
        assert condensed_entries[0].step_id == "step_old"

    @pytest.mark.asyncio
    async def test_removes_only_current_review_pair(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "tc_old_review",
                        "type": "function",
                        "function": {"name": "review_trajectory", "arguments": "{}"},
                    }
                ],
                "_sequence": 2,
            },
            {
                "role": "tool",
                "tool_call_id": "tc_old_review",
                "content": "Trajectory review: aligned=True.",
                "_sequence": 3,
            },
            {
                "role": "tool",
                "tool_call_id": "tc_search",
                "content": "Verbose search result",
                "_sequence": 5,
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "tc_current_review",
                        "type": "function",
                        "function": {"name": "review_trajectory", "arguments": "{}"},
                    }
                ],
                "_sequence": 6,
            },
            {
                "role": "tool",
                "tool_call_id": "tc_current_review",
                "content": "Trajectory review: aligned=False.",
                "_sequence": 7,
            },
        ]
        storage = AsyncMock()
        storage.get_step_by_tool_call_id = AsyncMock(return_value=None)
        storage.append_step_condensed_content = AsyncMock(return_value=True)

        outcome = await execute_step_back(
            messages=messages,
            checkpoint_seq=4,
            experience="Search drifted.",
            review_tool_call_id="tc_current_review",
            step_lookup={},
            storage=storage,
            session_id="s1",
            run_id="r1",
            agent_id="a1",
        )

        tool_call_ids = {
            msg.get("tool_call_id")
            for msg in outcome.messages
            if msg.get("role") == "tool"
        }
        assert "tc_old_review" in tool_call_ids
        assert "tc_current_review" not in tool_call_ids

    @pytest.mark.asyncio
    async def test_preserves_assistant_content_when_removing_review_call(self):
        messages = [
            {
                "role": "assistant",
                "content": "I checked the previous results.",
                "tool_calls": [
                    {
                        "id": "tc_review",
                        "type": "function",
                        "function": {"name": "review_trajectory", "arguments": "{}"},
                    }
                ],
                "_sequence": 6,
            },
            {
                "role": "tool",
                "tool_call_id": "tc_review",
                "content": "Trajectory review: aligned=False.",
                "_sequence": 7,
            },
        ]
        storage = AsyncMock()
        storage.get_step_by_tool_call_id = AsyncMock(return_value=None)
        storage.append_step_condensed_content = AsyncMock(return_value=True)

        outcome = await execute_step_back(
            messages=messages,
            checkpoint_seq=6,
            experience="Search drifted.",
            review_tool_call_id="tc_review",
            step_lookup={},
            storage=storage,
            session_id="s1",
            run_id="r1",
            agent_id="a1",
        )

        assert len(outcome.messages) == 1
        assert outcome.messages[0]["role"] == "assistant"
        assert outcome.messages[0]["content"] == "I checked the previous results."
        assert outcome.messages[0]["tool_calls"] == []


class TestStepBackOutcome:
    def test_default_not_applied(self):
        outcome = StepBackOutcome()
        assert outcome.applied is False
        assert outcome.affected_count == 0
        assert outcome.messages == []
