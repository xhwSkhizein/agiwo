"""Test message chunking in BaseChannelService."""

import re

import pytest

from server.channels.base import BaseChannelService
from server.channels.session.models import BatchContext


class _MinimalChannelService(BaseChannelService):
    """Minimal implementation for testing chunking logic."""

    async def _build_user_message(self, context, messages):
        return "test"

    async def _deliver_reply(self, context: BatchContext, text: str) -> None:
        pass

    async def _deliver_message(self, context: BatchContext, text: str) -> None:
        pass

    def _to_user_facing_error(self, error: Exception) -> str:
        return str(error)


@pytest.fixture
def service():
    """Create a minimal service instance for testing."""
    return _MinimalChannelService(
        session_service=None,
        agent_pool=None,
        executor=None,
        debounce_ms=1,
        max_batch_window_ms=1,
    )


def test_split_text_short_message(service):
    """Short messages should not be split."""
    text = "Hello, world!"
    chunks = service._split_text_into_chunks(text)
    assert chunks == [text]


def test_split_text_exact_max_length(service):
    """Messages exactly at max length should not be split."""
    text = "x" * 6000
    chunks = service._split_text_into_chunks(text)
    assert chunks == [text]


def test_split_text_long_message_without_newlines(service):
    """Long messages without newlines should be split at max_len."""
    text = "x" * 15000
    chunks = service._split_text_into_chunks(text)
    
    assert len(chunks) == 3
    # First two chunks should have continuation markers
    assert "[续 1/3]" in chunks[0]
    assert "[续 2/3]" in chunks[1]
    # Last chunk should not have a marker
    assert "[续" not in chunks[2]
    
    # Verify all content is preserved (minus markers)
    reconstructed = "".join(
        chunk.replace("\n\n[续 1/3]", "").replace("\n\n[续 2/3]", "")
        for chunk in chunks
    )
    assert reconstructed == text


def test_split_text_long_message_with_newlines(service):
    """Long messages with newlines should split at newline boundaries."""
    lines = [f"Line {i}" for i in range(1000)]
    text = "\n".join(lines)
    
    chunks = service._split_text_into_chunks(text)
    
    assert len(chunks) > 1
    # Each chunk should end with a newline (except possibly the last)
    for chunk in chunks[:-1]:
        # Remove continuation marker if present
        clean_chunk = chunk.split("\n\n[续")[0]
        assert clean_chunk.endswith("\n") or clean_chunk.endswith("]")


def test_split_text_custom_max_length(service):
    """Should respect custom max_len parameter."""
    text = "x" * 500
    chunks = service._split_text_into_chunks(text, max_len=200)
    
    assert len(chunks) == 3
    # Each chunk should be around 200 chars (plus marker)
    for chunk in chunks[:-1]:
        assert len(chunk) <= 220  # 200 + marker overhead


def test_split_text_preserves_content(service):
    """All content should be preserved across chunks."""
    text = "A" * 5000 + "B" * 5000 + "C" * 5000
    chunks = service._split_text_into_chunks(text)
    
    # Remove all continuation markers
    reconstructed = ""
    for chunk in chunks:
        clean = chunk
        clean = re.sub(r"\n\n\[续 \d+/\d+\]", "", clean)
        reconstructed += clean
    
    assert reconstructed == text


def test_split_text_empty_string(service):
    """Empty strings should return a single empty chunk."""
    chunks = service._split_text_into_chunks("")
    assert chunks == [""]


def test_split_text_single_character(service):
    """Single character should not be split."""
    chunks = service._split_text_into_chunks("x")
    assert chunks == ["x"]
