"""Tests for memory retrieval system."""

import tempfile
from pathlib import Path

import pytest

from agiwo.memory import MemoryChunker, MemoryIndexStore


class TestMemoryChunker:
    """Tests for MemoryChunker."""

    def test_chunk_empty_content(self):
        chunker = MemoryChunker(chunk_tokens=100, overlap_tokens=20)
        chunks = chunker.chunk_file(Path("test.md"), "")
        assert chunks == []

    def test_chunk_small_content(self):
        chunker = MemoryChunker(chunk_tokens=100, overlap_tokens=20)
        content = "This is a small test file.\nWith two lines."
        chunks = chunker.chunk_file(Path("test.md"), content)

        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 2
        assert "small test file" in chunks[0].text

    def test_chunk_large_content_creates_multiple_chunks(self):
        chunker = MemoryChunker(chunk_tokens=50, overlap_tokens=10)
        lines = [f"Line {i}: This is some content for testing.\n" for i in range(20)]
        content = "".join(lines)

        chunks = chunker.chunk_file(Path("test.md"), content)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.content_hash
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line

    def test_chunk_id_is_deterministic(self):
        chunker = MemoryChunker(chunk_tokens=100, overlap_tokens=20)
        content = "Test content"

        chunks1 = chunker.chunk_file(Path("test.md"), content)
        chunks2 = chunker.chunk_file(Path("test.md"), content)

        assert chunks1[0].chunk_id == chunks2[0].chunk_id
        assert chunks1[0].content_hash == chunks2[0].content_hash

    def test_different_paths_different_chunk_ids(self):
        chunker = MemoryChunker(chunk_tokens=100, overlap_tokens=20)
        content = "Test content"

        chunks1 = chunker.chunk_file(Path("file1.md"), content)
        chunks2 = chunker.chunk_file(Path("file2.md"), content)

        assert chunks1[0].chunk_id != chunks2[0].chunk_id
        assert chunks1[0].content_hash == chunks2[0].content_hash

    def test_count_tokens(self):
        chunker = MemoryChunker()
        tokens = chunker.count_tokens("Hello world")
        assert tokens > 0
        assert isinstance(tokens, int)


class TestMemoryIndexStore:
    """Tests for MemoryIndexStore."""

    @pytest.fixture
    def temp_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            memory_dir = workspace / "MEMORY"
            memory_dir.mkdir()
            yield workspace

    @pytest.mark.asyncio
    async def test_sync_empty_directory(self, temp_workspace):
        store = MemoryIndexStore(
            temp_workspace,
            embedding_provider="disabled",
            chunk_tokens=100,
            chunk_overlap_tokens=20,
        )
        await store.sync_files()
        results = await store.search("test")
        assert results == []
        await store.close()

    @pytest.mark.asyncio
    async def test_index_single_file(self, temp_workspace):
        memory_dir = temp_workspace / "MEMORY"
        (memory_dir / "2025-01-15.md").write_text(
            "# Project Notes\n\nDecided to use SQLite for storage.\n"
        )

        store = MemoryIndexStore(
            temp_workspace,
            embedding_provider="disabled",
            chunk_tokens=100,
            chunk_overlap_tokens=20,
        )
        await store.sync_files()

        results = await store.search("SQLite storage")
        assert len(results) >= 1
        assert "SQLite" in results[0].text
        await store.close()

    @pytest.mark.asyncio
    async def test_incremental_sync(self, temp_workspace):
        memory_dir = temp_workspace / "MEMORY"
        file1 = memory_dir / "notes.md"
        file1.write_text("Initial content about Python.")

        store = MemoryIndexStore(
            temp_workspace,
            embedding_provider="disabled",
            chunk_tokens=100,
            chunk_overlap_tokens=20,
        )
        await store.sync_files()

        results1 = await store.search("Python")
        assert len(results1) >= 1

        file1.write_text("Updated content about JavaScript.")
        await store.sync_files()

        results2 = await store.search("JavaScript")
        assert len(results2) >= 1
        assert "JavaScript" in results2[0].text

        results3 = await store.search("Python")
        assert len(results3) == 0 or "Python" not in results3[0].text

        await store.close()

    @pytest.mark.asyncio
    async def test_file_deletion_removes_from_index(self, temp_workspace):
        memory_dir = temp_workspace / "MEMORY"
        file1 = memory_dir / "temp.md"
        file1.write_text("Temporary content about Redis.")

        store = MemoryIndexStore(
            temp_workspace,
            embedding_provider="disabled",
            chunk_tokens=100,
            chunk_overlap_tokens=20,
        )
        await store.sync_files()

        results1 = await store.search("Redis")
        assert len(results1) >= 1

        file1.unlink()
        await store.sync_files()

        results2 = await store.search("Redis")
        assert len(results2) == 0

        await store.close()

    @pytest.mark.asyncio
    async def test_bm25_only_mode(self, temp_workspace):
        memory_dir = temp_workspace / "MEMORY"
        (memory_dir / "test.md").write_text(
            "# Architecture Decision\n\n"
            "We chose PostgreSQL over MySQL for better JSON support.\n"
        )

        store = MemoryIndexStore(
            temp_workspace,
            embedding_provider="disabled",
            chunk_tokens=100,
            chunk_overlap_tokens=20,
        )
        await store.sync_files()

        results = await store.search("PostgreSQL JSON")
        assert len(results) >= 1
        assert results[0].bm25_score > 0
        assert results[0].vector_score == 0.0

        await store.close()

    @pytest.mark.asyncio
    async def test_search_returns_line_numbers(self, temp_workspace):
        memory_dir = temp_workspace / "MEMORY"
        content = "\n".join([f"Line {i}" for i in range(1, 11)])
        (memory_dir / "lines.md").write_text(content)

        store = MemoryIndexStore(
            temp_workspace,
            embedding_provider="disabled",
            chunk_tokens=100,
            chunk_overlap_tokens=20,
        )
        await store.sync_files()

        results = await store.search("Line")
        assert len(results) >= 1
        assert results[0].start_line >= 1
        assert results[0].end_line >= results[0].start_line

        await store.close()

    @pytest.mark.asyncio
    async def test_multiple_files(self, temp_workspace):
        memory_dir = temp_workspace / "MEMORY"
        (memory_dir / "file1.md").write_text("Content about apples and oranges.")
        (memory_dir / "file2.md").write_text("Content about bananas and grapes.")

        store = MemoryIndexStore(
            temp_workspace,
            embedding_provider="disabled",
            chunk_tokens=100,
            chunk_overlap_tokens=20,
        )
        await store.sync_files()

        results1 = await store.search("apples")
        assert len(results1) >= 1
        assert "file1.md" in results1[0].path

        results2 = await store.search("bananas")
        assert len(results2) >= 1
        assert "file2.md" in results2[0].path

        await store.close()
