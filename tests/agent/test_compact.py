"""
Tests for Context Compact functionality.
"""

import pytest
from datetime import datetime

from agiwo.agent import CompactMetadata, CompactResult
from agiwo.agent.inner.compaction.prompt import (
    DEFAULT_ASSISTANT_RESPONSE,
    DEFAULT_COMPACT_PROMPT,
)
from agiwo.agent.storage.session import InMemorySessionStorage


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


class TestCompactResult:
    def test_create_result(self):
        metadata = CompactMetadata(
            session_id="session-1",
            agent_id="agent-1",
            start_seq=0,
            end_seq=10,
            before_token_estimate=5000,
            after_token_estimate=500,
            message_count=10,
            transcript_path="/path/to/transcript.jsonl",
            analysis={"summary": "Test summary"},
        )

        result = CompactResult(
            compacted_messages=[
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Summary"},
                {"role": "assistant", "content": "Understood"},
            ],
            metadata=metadata,
        )

        assert len(result.compacted_messages) == 3
        assert result.metadata.session_id == "session-1"


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
    def test_default_prompt_has_placeholders(self):
        # Only previous_summary placeholder is needed (conversation history is in context)
        assert "{previous_summary}" in DEFAULT_COMPACT_PROMPT

    def test_default_assistant_response(self):
        assert "Understood" in DEFAULT_ASSISTANT_RESPONSE
