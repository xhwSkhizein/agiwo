"""Tests for the retrospect module — trigger, notice injection, offload, and execution."""

import pytest

from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.run import RunLedger
from agiwo.agent.models.step import MessageRole, StepRecord
from agiwo.agent.retrospect import (
    check_retrospect_trigger,
    execute_retrospect,
    inject_system_notice,
    offload_to_disk,
    update_retrospect_tracking,
)
from agiwo.agent.storage.base import InMemoryRunStepStorage


class TestCheckRetrospectTrigger:
    def test_disabled_returns_false(self):
        config = AgentOptions(enable_tool_retrospect=False)
        ledger = RunLedger()
        assert not check_retrospect_trigger(config, ledger, "x" * 8000, "bash")

    def test_retrospect_tool_itself_never_triggers(self):
        config = AgentOptions(
            enable_tool_retrospect=True, retrospect_token_threshold=10
        )
        ledger = RunLedger()
        assert not check_retrospect_trigger(
            config, ledger, "x" * 8000, "retrospect_tool_result"
        )

    def test_single_large_result_triggers(self):
        config = AgentOptions(
            enable_tool_retrospect=True, retrospect_token_threshold=100
        )
        ledger = RunLedger()
        content = "x" * 500  # ~125 tokens
        assert check_retrospect_trigger(config, ledger, content, "bash")

    def test_small_result_does_not_trigger(self):
        config = AgentOptions(
            enable_tool_retrospect=True, retrospect_token_threshold=100
        )
        ledger = RunLedger()
        content = "x" * 200  # ~50 tokens
        assert not check_retrospect_trigger(config, ledger, content, "bash")

    def test_round_interval_triggers(self):
        config = AgentOptions(
            enable_tool_retrospect=True,
            retrospect_token_threshold=99999,
            retrospect_round_interval=3,
            retrospect_accumulated_token_threshold=99999,
        )
        ledger = RunLedger(retrospect_pending_rounds=3)
        assert check_retrospect_trigger(config, ledger, "tiny", "bash")

    def test_accumulated_token_triggers(self):
        config = AgentOptions(
            enable_tool_retrospect=True,
            retrospect_token_threshold=99999,
            retrospect_round_interval=99999,
            retrospect_accumulated_token_threshold=500,
        )
        ledger = RunLedger(retrospect_pending_tokens=600)
        assert check_retrospect_trigger(config, ledger, "tiny", "bash")


class TestUpdateRetrospectTracking:
    def test_accumulates_tokens_and_rounds(self):
        ledger = RunLedger()
        update_retrospect_tracking(ledger, "a" * 400)  # ~100 tokens
        assert ledger.retrospect_pending_tokens == 100
        assert ledger.retrospect_pending_rounds == 1

        update_retrospect_tracking(ledger, "b" * 800)  # ~200 tokens
        assert ledger.retrospect_pending_tokens == 300
        assert ledger.retrospect_pending_rounds == 2


class TestInjectSystemNotice:
    def test_appends_notice_tag(self):
        result = inject_system_notice("original output")
        assert result.startswith("original output")
        assert "<system-notice>" in result
        assert "retrospect_tool_result" in result


class TestOffloadToDisk:
    def test_writes_file_and_returns_placeholder(self, tmp_path):
        target = tmp_path / "sub" / "out.txt"
        placeholder = offload_to_disk("big content here", target)
        assert target.read_text() == "big content here"
        assert str(target) in placeholder


class TestExecuteRetrospect:
    @pytest.mark.asyncio
    async def test_offloads_and_appends_feedback(self, tmp_path):
        storage = InMemoryRunStepStorage()
        session_id = "sess-1"

        step1 = StepRecord(
            session_id=session_id,
            run_id="run-1",
            sequence=1,
            role=MessageRole.TOOL,
            content="big result A",
            tool_call_id="tc-1",
            name="bash",
        )
        step2 = StepRecord(
            session_id=session_id,
            run_id="run-1",
            sequence=2,
            role=MessageRole.TOOL,
            content="big result B",
            tool_call_id="tc-2",
            name="bash",
        )
        await storage.save_step(step1)
        await storage.save_step(step2)

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
        ledger = RunLedger(last_retrospect_seq=0)
        step_lookup = {
            "tc-1": {"id": step1.id, "sequence": 1},
            "tc-2": {"id": step2.id, "sequence": 2},
        }

        result = await execute_retrospect(
            feedback="Plan A and B both failed, need to try table Y.",
            messages=messages,
            ledger=ledger,
            storage=storage,
            session_id=session_id,
            offload_dir=tmp_path / "offload",
            step_lookup=step_lookup,
        )

        assert "2 tool result" in result

        assert (tmp_path / "offload" / "tc-1.txt").read_text() == "big result A"
        assert (tmp_path / "offload" / "tc-2.txt").read_text() == "big result B"

        assert messages[0]["content"].startswith("[")
        assert "Retrospect:" in messages[1]["content"]
        assert "Plan A and B" in messages[1]["content"]

        assert ledger.retrospect_pending_tokens == 0
        assert ledger.retrospect_pending_rounds == 0
        assert ledger.last_retrospect_seq == 2

    @pytest.mark.asyncio
    async def test_feedback_persisted_to_condensed_content(self, tmp_path):
        """condensed_content in storage must include the feedback suffix."""
        storage = InMemoryRunStepStorage()
        session_id = "sess-1"

        step = StepRecord(
            session_id=session_id,
            run_id="run-1",
            sequence=5,
            role=MessageRole.TOOL,
            content="verbose output",
            tool_call_id="tc-5",
            name="bash",
        )
        await storage.save_step(step)

        messages = [
            {
                "role": "tool",
                "content": "verbose output",
                "tool_call_id": "tc-5",
                "_sequence": 5,
            },
        ]
        ledger = RunLedger(last_retrospect_seq=0)
        step_lookup = {"tc-5": {"id": step.id, "sequence": 5}}

        await execute_retrospect(
            feedback="dead-end, switch approach",
            messages=messages,
            ledger=ledger,
            storage=storage,
            session_id=session_id,
            offload_dir=tmp_path / "offload",
            step_lookup=step_lookup,
        )

        stored_step = storage.steps[session_id][0]
        assert stored_step.condensed_content is not None
        assert "Retrospect:" in stored_step.condensed_content
        assert "dead-end" in stored_step.condensed_content

    @pytest.mark.asyncio
    async def test_removes_retrospect_tool_call(self, tmp_path):
        storage = InMemoryRunStepStorage()
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
        ledger = RunLedger(last_retrospect_seq=0)
        step_lookup = {"tc-1": {"id": "s1", "sequence": 1}}

        await execute_retrospect(
            feedback="summary",
            messages=messages,
            ledger=ledger,
            storage=storage,
            session_id=session_id,
            offload_dir=tmp_path / "offload",
            step_lookup=step_lookup,
        )

        roles = [m["role"] for m in messages]
        assert "assistant" not in roles or all(
            tc["function"]["name"] != "retrospect_tool_result"
            for m in messages
            if m.get("tool_calls")
            for tc in m["tool_calls"]
        )
        assert not any(
            m.get("tool_call_id") == "tc-r" for m in messages if m.get("role") == "tool"
        )


class TestStorageUpdateCondensedContent:
    @pytest.mark.asyncio
    async def test_in_memory_update(self):
        storage = InMemoryRunStepStorage()
        step = StepRecord(
            session_id="s1",
            run_id="r1",
            sequence=1,
            role=MessageRole.TOOL,
            content="original",
            tool_call_id="tc-1",
            name="bash",
        )
        await storage.save_step(step)

        updated = await storage.update_step_condensed_content(
            "s1", step.id, "condensed"
        )
        assert updated is True
        assert storage.steps["s1"][0].condensed_content == "condensed"

    @pytest.mark.asyncio
    async def test_in_memory_update_nonexistent(self):
        storage = InMemoryRunStepStorage()
        updated = await storage.update_step_condensed_content("s1", "ghost", "x")
        assert updated is False
