"""
Tests for Context Compact functionality.
"""

from datetime import datetime

import pytest

from agiwo.agent.compaction import _compact, build_compacted_messages
from agiwo.agent.compact_types import CompactMetadata
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.run_state import RunContext, SessionRuntime
from agiwo.agent.storage.base import InMemoryRunStepStorage
from agiwo.agent.storage.session import InMemorySessionStorage
from agiwo.llm.base import Model, StreamChunk


class _CompactModel(Model):
    async def arun_stream(self, messages, tools=None):
        del messages, tools
        yield StreamChunk(content='{"summary": "compressed"}')
        yield StreamChunk(finish_reason="stop")


class TestCompactMetadata:
    def test_create_metadata(self):
        metadata = CompactMetadata(
            session_id="session-1",
            agent_id="agent-1",
            start_seq=0,
            end_seq=10,
            before_token_estimate=5000,
            after_token_estimate=500,
            message_count=10,
            transcript_path="/path/to/transcript.jsonl",
            analysis={"summary": "Test summary", "key_decisions": ["decision1"]},
            compact_model="gpt-4o-mini",
            compact_tokens=1000,
        )

        assert metadata.session_id == "session-1"
        assert metadata.agent_id == "agent-1"
        assert metadata.start_seq == 0
        assert metadata.end_seq == 10
        assert metadata.before_token_estimate == 5000
        assert metadata.after_token_estimate == 500
        assert metadata.get_summary() == "Test summary"

    def test_get_summary_missing(self):
        metadata = CompactMetadata(
            session_id="session-1",
            agent_id="agent-1",
            start_seq=0,
            end_seq=10,
            before_token_estimate=5000,
            after_token_estimate=500,
            message_count=10,
            transcript_path="/path/to/transcript.jsonl",
            analysis={},
        )
        assert metadata.get_summary() == ""


class TestInMemorySessionStorage:
    @pytest.fixture
    def storage(self):
        return InMemorySessionStorage()

    @pytest.mark.asyncio
    async def test_save_and_get_latest(self, storage):
        metadata1 = CompactMetadata(
            session_id="session-1",
            agent_id="agent-1",
            start_seq=0,
            end_seq=5,
            before_token_estimate=3000,
            after_token_estimate=300,
            message_count=5,
            transcript_path="/path/1.jsonl",
            analysis={"summary": "First summary"},
            created_at=datetime(2024, 1, 1, 10, 0, 0),
        )

        metadata2 = CompactMetadata(
            session_id="session-1",
            agent_id="agent-1",
            start_seq=6,
            end_seq=10,
            before_token_estimate=4000,
            after_token_estimate=400,
            message_count=5,
            transcript_path="/path/2.jsonl",
            analysis={"summary": "Second summary"},
            created_at=datetime(2024, 1, 1, 11, 0, 0),
        )

        await storage.save_compact_metadata("session-1", "agent-1", metadata1)
        await storage.save_compact_metadata("session-1", "agent-1", metadata2)

        latest = await storage.get_latest_compact_metadata("session-1", "agent-1")
        assert latest is not None
        assert latest.get_summary() == "Second summary"

    @pytest.mark.asyncio
    async def test_get_latest_not_found(self, storage):
        result = await storage.get_latest_compact_metadata("nonexistent", "agent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_compact_history(self, storage):
        for i in range(3):
            metadata = CompactMetadata(
                session_id="session-1",
                agent_id="agent-1",
                start_seq=i * 5,
                end_seq=(i + 1) * 5,
                before_token_estimate=1000,
                after_token_estimate=100,
                message_count=5,
                transcript_path=f"/path/{i}.jsonl",
                analysis={"summary": f"Summary {i}"},
            )
            await storage.save_compact_metadata("session-1", "agent-1", metadata)

        history = await storage.get_compact_history("session-1", "agent-1")
        assert len(history) == 3
        assert history[0].get_summary() == "Summary 0"
        assert history[2].get_summary() == "Summary 2"

    @pytest.mark.asyncio
    async def test_isolation_by_agent_id(self, storage):
        metadata1 = CompactMetadata(
            session_id="session-1",
            agent_id="agent-1",
            start_seq=0,
            end_seq=5,
            before_token_estimate=1000,
            after_token_estimate=100,
            message_count=5,
            transcript_path="/path/1.jsonl",
            analysis={"summary": "Agent 1 summary"},
        )

        metadata2 = CompactMetadata(
            session_id="session-1",
            agent_id="agent-2",
            start_seq=0,
            end_seq=5,
            before_token_estimate=1000,
            after_token_estimate=100,
            message_count=5,
            transcript_path="/path/2.jsonl",
            analysis={"summary": "Agent 2 summary"},
        )

        await storage.save_compact_metadata("session-1", "agent-1", metadata1)
        await storage.save_compact_metadata("session-1", "agent-2", metadata2)

        latest1 = await storage.get_latest_compact_metadata("session-1", "agent-1")
        latest2 = await storage.get_latest_compact_metadata("session-1", "agent-2")

        assert latest1 is not None
        assert latest2 is not None
        assert latest1.get_summary() == "Agent 1 summary"
        assert latest2.get_summary() == "Agent 2 summary"


class TestDefaultPrompt:
    def test_build_compacted_messages_includes_summary_and_default_response(self):
        messages = build_compacted_messages(
            system_prompt="system prompt",
            summary="compressed summary",
            transcript_path="/tmp/transcript.jsonl",
        )

        assert messages[0] == {"role": "system", "content": "system prompt"}
        assert messages[1]["role"] == "user"
        assert "compressed summary" in messages[1]["content"]
        assert "/tmp/transcript.jsonl" in messages[1]["content"]
        assert messages[2]["role"] == "assistant"
        assert "Understood" in messages[2]["content"]


@pytest.mark.asyncio
async def test_compact_uses_the_same_step_commit_pipeline_as_normal_runs(tmp_path):
    step_storage = InMemoryRunStepStorage()
    session_storage = InMemorySessionStorage()
    session_runtime = SessionRuntime(
        session_id="sess-1",
        run_step_storage=step_storage,
        session_storage=session_storage,
    )
    published = []

    async def publish(item):
        published.append(item)

    session_runtime.publish = publish  # type: ignore[method-assign]

    seen_steps = []

    async def on_step(step):
        seen_steps.append(step.name or step.role.value)

    state = RunContext(
        session_runtime=session_runtime,
        run_id="run-1",
        agent_id="agent-1",
        agent_name="agent",
        hooks=AgentHooks(on_step=on_step),
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ],
    )

    metadata = await _compact(
        state,
        _CompactModel(id="compact-model", name="compact-model", provider="openai"),
        session_storage,
        abort_signal=None,
        compact_prompt=None,
        compact_start_seq=1,
        root_path=str(tmp_path),
    )

    steps = await step_storage.get_steps("sess-1", run_id="run-1")

    assert [step.name for step in steps[-2:]] == ["compact_request", "compact"]
    assert seen_steps[-2:] == ["compact_request", "compact"]
    assert [item.type for item in published if item.type == "step_completed"][-2:] == [
        "step_completed",
        "step_completed",
    ]
    assert metadata.get_summary() == "compressed"
