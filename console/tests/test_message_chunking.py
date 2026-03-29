"""Test message chunking utility."""

import re

from server.channels.utils import split_text_into_chunks


def test_split_text_short_message():
    """Short messages should not be split."""
    text = "Hello, world!"
    chunks = split_text_into_chunks(text)
    assert chunks == [text]


def test_split_text_exact_max_length():
    """Messages exactly at max length should not be split."""
    text = "x" * 6000
    chunks = split_text_into_chunks(text)
    assert chunks == [text]


def test_split_text_long_message_without_newlines():
    """Long messages without newlines should be split at max_len."""
    text = "x" * 15000
    chunks = split_text_into_chunks(text)

    assert len(chunks) == 3
    assert "[续 1/3]" in chunks[0]
    assert "[续 2/3]" in chunks[1]
    assert "[续" not in chunks[2]

    reconstructed = "".join(
        chunk.replace("\n\n[续 1/3]", "").replace("\n\n[续 2/3]", "")
        for chunk in chunks
    )
    assert reconstructed == text


def test_split_text_long_message_with_newlines():
    """Long messages with newlines should split at newline boundaries."""
    lines = [f"Line {i}" for i in range(1000)]
    text = "\n".join(lines)

    chunks = split_text_into_chunks(text)

    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        clean_chunk = chunk.split("\n\n[续")[0]
        assert clean_chunk.endswith("\n") or clean_chunk.endswith("]")


def test_split_text_custom_max_length():
    """Should respect custom max_len parameter."""
    text = "x" * 500
    chunks = split_text_into_chunks(text, max_len=200)

    assert len(chunks) == 3
    for chunk in chunks[:-1]:
        assert len(chunk) <= 220


def test_split_text_preserves_content():
    """All content should be preserved across chunks."""
    text = "A" * 5000 + "B" * 5000 + "C" * 5000
    chunks = split_text_into_chunks(text)

    reconstructed = ""
    for chunk in chunks:
        clean = chunk
        clean = re.sub(r"\n\n\[续 \d+/\d+\]", "", clean)
        reconstructed += clean

    assert reconstructed == text


def test_split_text_empty_string():
    """Empty strings should return a single empty chunk."""
    chunks = split_text_into_chunks("")
    assert chunks == [""]


def test_split_text_single_character():
    """Single character should not be split."""
    chunks = split_text_into_chunks("x")
    assert chunks == ["x"]
