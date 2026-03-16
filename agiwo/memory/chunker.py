"""
MemoryChunker - Split markdown files into overlapping chunks for indexing.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from agiwo.utils.token_encoding import resolve_text_encoding


@dataclass
class MemoryChunk:
    """A chunk of text from a memory file."""

    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str
    content_hash: str


def _resolve_encoding():
    return resolve_text_encoding(default_encoding="cl100k_base")


class MemoryChunker:
    """Split markdown content into overlapping token-based chunks."""

    def __init__(self, chunk_tokens: int = 400, overlap_tokens: int = 80):
        self._chunk_tokens = chunk_tokens
        self._overlap_tokens = overlap_tokens
        self._encoding = _resolve_encoding()

    def chunk_file(self, path: Path, content: str) -> list[MemoryChunk]:
        """Split file content into chunks with line tracking."""
        if not content.strip():
            return []

        lines = content.splitlines(keepends=True)
        chunks: list[MemoryChunk] = []

        current_tokens: list[int] = []
        current_lines: list[str] = []
        current_start_line = 1

        for line_idx, line in enumerate(lines):
            line_tokens = self._encoding.encode(line)

            if len(current_tokens) + len(line_tokens) > self._chunk_tokens:
                if current_lines:
                    chunk = self._create_chunk(
                        path=str(path),
                        start_line=current_start_line,
                        end_line=current_start_line + len(current_lines) - 1,
                        lines=current_lines,
                    )
                    chunks.append(chunk)

                    overlap_lines, overlap_tokens = self._get_overlap(
                        current_lines, current_tokens
                    )
                    current_lines = overlap_lines
                    current_tokens = overlap_tokens
                    current_start_line = line_idx + 1 - len(overlap_lines)

            current_lines.append(line)
            current_tokens.extend(line_tokens)

        if current_lines:
            chunk = self._create_chunk(
                path=str(path),
                start_line=current_start_line,
                end_line=current_start_line + len(current_lines) - 1,
                lines=current_lines,
            )
            chunks.append(chunk)

        return chunks

    def _get_overlap(
        self, lines: list[str], tokens: list[int]
    ) -> tuple[list[str], list[int]]:
        """Get overlap lines from the end of current chunk."""
        if self._overlap_tokens <= 0:
            return [], []

        overlap_lines: list[str] = []
        overlap_token_count = 0

        for line in reversed(lines):
            line_tokens = self._encoding.encode(line)
            if overlap_token_count + len(line_tokens) > self._overlap_tokens:
                break
            overlap_lines.insert(0, line)
            overlap_token_count += len(line_tokens)

        overlap_tokens = self._encoding.encode("".join(overlap_lines))
        return overlap_lines, overlap_tokens

    def _create_chunk(
        self, path: str, start_line: int, end_line: int, lines: list[str]
    ) -> MemoryChunk:
        """Create a MemoryChunk from lines."""
        text = "".join(lines).strip()
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        chunk_id = hashlib.sha256(f"{path}:{start_line}".encode()).hexdigest()[:16]

        return MemoryChunk(
            chunk_id=chunk_id,
            path=path,
            start_line=start_line,
            end_line=end_line,
            text=text,
            content_hash=content_hash,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._encoding.encode(text))


__all__ = ["MemoryChunk", "MemoryChunker"]
