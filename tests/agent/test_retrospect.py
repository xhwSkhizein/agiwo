"""Tests for the retrospect package — trigger, notice injection, offload, and execution."""

import pytest

from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.log import build_committed_step_entry
from agiwo.agent.models.run import RetrospectState, RunLedger
from agiwo.agent.models.step import MessageRole, StepView
from agiwo.agent.retrospect.triggers import (
    RetrospectTrigger,
    check_retrospect_trigger,
    inject_system_notice,
    update_retrospect_tracking,
)
from agiwo.agent.retrospect.executor import (
    execute_retrospect,
    offload_to_disk,
)
from agiwo.agent.storage.base import InMemoryRunLogStorage


class TestCheckRetrospectTrigger:
    def test_disabled_returns_none(self):
        config = AgentOptions(enable_tool_retrospect=False)
        ledger = RunLedger()
        assert (
            check_retrospect_trigger(config, ledger, "x" * 8000, "bash")
            is RetrospectTrigger.NONE
        )

    def test_retrospect_tool_itself_never_triggers(self):
        config = AgentOptions(
            enable_tool_retrospect=True, retrospect_token_threshold=10
        )
        ledger = RunLedger()
        assert (
            check_retrospect_trigger(
                config, ledger, "x" * 8000, "retrospect_tool_result"
            )
            is RetrospectTrigger.NONE
        )

    def test_single_large_result_triggers(self):
        config = AgentOptions(
            enable_tool_retrospect=True, retrospect_token_threshold=100
        )
        ledger = RunLedger()
        content = "x" * 500  # ~125 tokens
        assert (
            check_retrospect_trigger(config, ledger, content, "bash")
            is RetrospectTrigger.LARGE_RESULT
        )

    def test_small_result_does_not_trigger(self):
        config = AgentOptions(
            enable_tool_retrospect=True, retrospect_token_threshold=100
        )
        ledger = RunLedger()
        content = "x" * 200  # ~50 tokens
        assert (
            check_retrospect_trigger(config, ledger, content, "bash")
            is RetrospectTrigger.NONE
        )

    def test_round_interval_triggers(self):
        config = AgentOptions(
            enable_tool_retrospect=True,
            retrospect_token_threshold=99999,
            retrospect_round_interval=3,
            retrospect_accumulated_token_threshold=99999,
        )
        ledger = RunLedger(retrospect=RetrospectState(pending_rounds=3))
        assert (
            check_retrospect_trigger(config, ledger, "tiny", "bash")
            is RetrospectTrigger.ROUND_INTERVAL
        )

    def test_accumulated_token_triggers(self):
        config = AgentOptions(
            enable_tool_retrospect=True,
            retrospect_token_threshold=99999,
            retrospect_round_interval=99999,
            retrospect_accumulated_token_threshold=500,
        )
        ledger = RunLedger(retrospect=RetrospectState(pending_tokens=600))
        assert (
            check_retrospect_trigger(config, ledger, "tiny", "bash")
            is RetrospectTrigger.TOKEN_ACCUMULATED
        )


class TestUpdateRetrospectTracking:
    def test_accumulates_tokens_and_rounds(self):
        ledger = RunLedger()
        update_retrospect_tracking(ledger, "a" * 400)  # ~100 tokens
        assert ledger.retrospect.pending_tokens == 100
        assert ledger.retrospect.pending_rounds == 1

        update_retrospect_tracking(ledger, "b" * 800)  # ~200 tokens
        assert ledger.retrospect.pending_tokens == 300
        assert ledger.retrospect.pending_rounds == 2


class TestInjectSystemNotice:
    def test_large_result_notice(self):
        result = inject_system_notice("original output", RetrospectTrigger.LARGE_RESULT)
        assert result.startswith("original output")
        assert "<system-notice>" in result
        assert "large" in result.lower()

    def test_round_interval_notice(self):
        result = inject_system_notice(
            "original output", RetrospectTrigger.ROUND_INTERVAL
        )
        assert "<system-notice>" in result
        assert "goal" in result.lower()

    def test_token_accumulated_notice(self):
        result = inject_system_notice(
            "original output", RetrospectTrigger.TOKEN_ACCUMULATED
        )
        assert "<system-notice>" in result
        assert "context" in result.lower()


class TestInjectSystemNoticeNone:
    def test_none_trigger_returns_content_unchanged(self):
        result = inject_system_notice("original output", RetrospectTrigger.NONE)
        assert result == "original output"


class TestOffloadToDisk:
    @pytest.mark.asyncio
    async def test_writes_file_and_returns_placeholder(self, tmp_path):
        target = tmp_path / "sub" / "out.txt"
        placeholder = await offload_to_disk("big content here", target)
        assert target.read_text() == "big content here"
        assert str(target) in placeholder


class TestExecuteRetrospect:
    @pytest.mark.asyncio
    async def test_offloads_and_appends_feedback(self, tmp_path):
        storage = InMemoryRunLogStorage()
        session_id = "sess-1"

        step1 = StepView(
            session_id=session_id,
            run_id="run-1",
            sequence=1,
            role=MessageRole.TOOL,
            content="big result A",
            tool_call_id="tc-1",
            name="bash",
        )
        step2 = StepView(
            session_id=session_id,
            run_id="run-1",
            sequence=2,
            role=MessageRole.TOOL,
            content="big result B",
            tool_call_id="tc-2",
            name="bash",
        )
        await storage.append_entries(
            [
                build_committed_step_entry(step1),
                build_committed_step_entry(step2),
            ]
        )

        messages = [
            {
                "role": "tool",
                "content": "big result A",
                "tool_call_id": "tc-1",
                "_sequence": 1,
            },
            {
                "role": "tool",
                "content": "big result B",
                "tool_call_id": "tc-2",
                "_sequence": 2,
            },
        ]
        ledger = RunLedger(retrospect=RetrospectState(last_seq=0))
        step_lookup = {
            "tc-1": {"id": step1.id, "sequence": 1},
            "tc-2": {"id": step2.id, "sequence": 2},
        }

        outcome = await execute_retrospect(
            feedback="Plan A and B both failed, need to try table Y.",
            messages=messages,
            ledger=ledger,
            storage=storage,
            session_id=session_id,
            run_id="run-1",
            agent_id="agent-1",
            offload_dir=tmp_path / "offload",
            step_lookup=step_lookup,
        )

        assert outcome.applied is True
        assert outcome.offloaded_count == 2

        assert (tmp_path / "offload" / "tc-1.txt").read_text() == "big result A"
        assert (tmp_path / "offload" / "tc-2.txt").read_text() == "big result B"

        assert outcome.messages[0]["content"].startswith("[")
        assert "Retrospect:" in outcome.messages[1]["content"]
        assert "Plan A and B" in outcome.messages[1]["content"]
        assert outcome.affected_sequences == [1, 2]
        assert outcome.affected_step_ids == [step1.id, step2.id]
        assert outcome.feedback == "Plan A and B both failed, need to try table Y."
        assert outcome.replacement is not None
        assert "Retrospect:" in outcome.replacement

        assert ledger.retrospect.pending_tokens == 0
        assert ledger.retrospect.pending_rounds == 0
        assert ledger.retrospect.last_seq == 2

    @pytest.mark.asyncio
    async def test_original_messages_not_mutated(self, tmp_path):
        """execute_retrospect must work on a copy, leaving originals intact."""
        storage = InMemoryRunLogStorage()
        session_id = "sess-1"

        messages = [
            {
                "role": "tool",
                "content": "original content",
                "tool_call_id": "tc-1",
                "_sequence": 1,
            },
        ]
        ledger = RunLedger(retrospect=RetrospectState(last_seq=0))
        step_lookup = {"tc-1": {"id": "s1", "sequence": 1}}

        outcome = await execute_retrospect(
            feedback="summary",
            messages=messages,
            ledger=ledger,
            storage=storage,
            session_id=session_id,
            run_id="run-1",
            agent_id="agent-1",
            offload_dir=tmp_path / "offload",
            step_lookup=step_lookup,
        )

        assert messages[0]["content"] == "original content"
        assert outcome.applied is True
        assert outcome.messages[0]["content"] != "original content"

    @pytest.mark.asyncio
    async def test_feedback_persisted_to_condensed_content(self, tmp_path):
        """condensed_content in storage must include the feedback suffix."""
        storage = InMemoryRunLogStorage()
        session_id = "sess-1"

        step = StepView(
            session_id=session_id,
            run_id="run-1",
            sequence=5,
            role=MessageRole.TOOL,
            content="verbose output",
            tool_call_id="tc-5",
            name="bash",
        )
        await storage.append_entries([build_committed_step_entry(step)])

        messages = [
            {
                "role": "tool",
                "content": "verbose output",
                "tool_call_id": "tc-5",
                "_sequence": 5,
            },
        ]
        ledger = RunLedger(retrospect=RetrospectState(last_seq=0))
        step_lookup = {"tc-5": {"id": step.id, "sequence": 5}}

        await execute_retrospect(
            feedback="dead-end, switch approach",
            messages=messages,
            ledger=ledger,
            storage=storage,
            session_id=session_id,
            run_id="run-1",
            agent_id="agent-1",
            offload_dir=tmp_path / "offload",
            step_lookup=step_lookup,
        )

        stored_steps = await storage.list_step_views(session_id=session_id)
        stored_step = stored_steps[0]
        assert stored_step.condensed_content is not None
        assert "Retrospect:" in stored_step.condensed_content
        assert "dead-end" in stored_step.condensed_content

    @pytest.mark.asyncio
    async def test_removes_retrospect_tool_call(self, tmp_path):
        storage = InMemoryRunLogStorage()
        session_id = "sess-1"

        messages = [
            {"role": "tool", "content": "data", "tool_call_id": "tc-1", "_sequence": 1},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "tc-r",
                        "function": {
                            "name": "retrospect_tool_result",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {"role": "tool", "content": "ok", "tool_call_id": "tc-r", "_sequence": 3},
        ]
        ledger = RunLedger(retrospect=RetrospectState(last_seq=0))
        step_lookup = {"tc-1": {"id": "s1", "sequence": 1}}

        outcome = await execute_retrospect(
            feedback="summary",
            messages=messages,
            ledger=ledger,
            storage=storage,
            session_id=session_id,
            run_id="run-1",
            agent_id="agent-1",
            offload_dir=tmp_path / "offload",
            step_lookup=step_lookup,
        )

        roles = [m["role"] for m in outcome.messages]
        assert "assistant" not in roles or all(
            tc["function"]["name"] != "retrospect_tool_result"
            for m in outcome.messages
            if m.get("tool_calls")
            for tc in m["tool_calls"]
        )
        assert not any(
            m.get("tool_call_id") == "tc-r"
            for m in outcome.messages
            if m.get("role") == "tool"
        )


class TestStepViewToMessageSequence:
    def test_to_message_includes_sequence(self):
        step = StepView(
            session_id="s1",
            run_id="r1",
            sequence=42,
            role=MessageRole.TOOL,
            content="result",
            tool_call_id="tc-1",
            name="bash",
        )
        msg = step.to_message()
        assert msg["_sequence"] == 42

    def test_to_message_sequence_zero_for_default(self):
        step = StepView(
            session_id="s1",
            run_id="r1",
            sequence=0,
            role=MessageRole.USER,
            content="hello",
        )
        msg = step.to_message()
        assert "_sequence" in msg
        assert msg["_sequence"] == 0


class TestRemoveRetrospectMultiCall:
    @pytest.mark.asyncio
    async def test_multi_call_removes_only_retrospect_entry(self, tmp_path):
        """When assistant has [search, retrospect], only retrospect tc is removed."""
        storage = InMemoryRunLogStorage()
        session_id = "sess-1"

        messages = [
            {"role": "tool", "content": "data", "tool_call_id": "tc-1", "_sequence": 1},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "tc-search",
                        "function": {"name": "search_db", "arguments": "{}"},
                    },
                    {
                        "id": "tc-r",
                        "function": {
                            "name": "retrospect_tool_result",
                            "arguments": "{}",
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "content": "search result",
                "tool_call_id": "tc-search",
                "_sequence": 3,
            },
            {"role": "tool", "content": "ok", "tool_call_id": "tc-r", "_sequence": 4},
        ]
        ledger = RunLedger(retrospect=RetrospectState(last_seq=0))
        step_lookup = {"tc-1": {"id": "s1", "sequence": 1}}

        outcome = await execute_retrospect(
            feedback="summary",
            messages=messages,
            ledger=ledger,
            storage=storage,
            session_id=session_id,
            run_id="run-1",
            agent_id="agent-1",
            offload_dir=tmp_path / "offload",
            step_lookup=step_lookup,
        )

        assistant_msgs = [m for m in outcome.messages if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        assert len(assistant_msgs[0]["tool_calls"]) == 1
        assert assistant_msgs[0]["tool_calls"][0]["function"]["name"] == "search_db"

        tool_ids = [
            m.get("tool_call_id") for m in outcome.messages if m.get("role") == "tool"
        ]
        assert "tc-r" not in tool_ids


class TestStorageAppendCondensedContent:
    @pytest.mark.asyncio
    async def test_in_memory_append(self):
        storage = InMemoryRunLogStorage()
        step = StepView(
            session_id="s1",
            run_id="r1",
            sequence=1,
            role=MessageRole.TOOL,
            content="original",
            tool_call_id="tc-1",
            name="bash",
        )
        await storage.append_entries([build_committed_step_entry(step)])

        updated = await storage.append_step_condensed_content(
            "s1", "r1", "agent-1", step.id, "condensed"
        )
        assert updated is True
        steps = await storage.list_step_views(session_id="s1")
        assert steps[0].condensed_content == "condensed"

    @pytest.mark.asyncio
    async def test_in_memory_append_nonexistent(self):
        storage = InMemoryRunLogStorage()
        updated = await storage.append_step_condensed_content(
            "s1", "r1", "agent-1", "ghost", "x"
        )
        assert updated is False
