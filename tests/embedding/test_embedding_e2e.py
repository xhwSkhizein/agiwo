"""
E2E tests for embedding and memory retrieval system.

These tests require actual API access or local models.
Skip markers are used to conditionally run based on environment.
"""

import math
import tempfile
from pathlib import Path

import pytest

from agiwo.config.settings import settings
from agiwo.embedding import EmbeddingFactory
from agiwo.memory import MemoryIndexStore


def has_openai_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(settings.openai_api_key or settings.embedding_api_key)


def has_local_model() -> bool:
    """Check if local embedding model is available."""
    path = settings.local_embedding_model_path or ""
    return bool(path) and Path(path).exists()


skip_without_openai = pytest.mark.skipif(
    not has_openai_key(), reason="OPENAI_API_KEY not set"
)

skip_without_local_model = pytest.mark.skipif(
    not has_local_model(),
    reason="AGIWO_LOCAL_EMBEDDING_MODEL_PATH not set or model not found",
)


class TestOpenAIEmbeddingE2E:
    """E2E tests for OpenAI embedding."""

    @skip_without_openai
    @pytest.mark.asyncio
    async def test_embed_single_text(self):
        """Test embedding a single text with real API."""
        model = EmbeddingFactory.create(provider="openai")
        assert model is not None

        result = await model.embed(["Hello, world!"])

        assert len(result) == 1
        assert len(result[0]) == model.dimensions
        assert all(isinstance(v, float) for v in result[0])

        await model.close()

    @skip_without_openai
    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self):
        """Test embedding multiple texts with real API."""
        model = EmbeddingFactory.create(provider="openai")
        assert model is not None

        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language.",
        ]
        result = await model.embed(texts)

        assert len(result) == 3
        for embedding in result:
            assert len(embedding) == model.dimensions

        await model.close()

    @skip_without_openai
    @pytest.mark.asyncio
    async def test_embed_chinese_text(self):
        """Test embedding Chinese text."""
        model = EmbeddingFactory.create(provider="openai")
        assert model is not None

        result = await model.embed(["你好，世界！这是一个测试。"])

        assert len(result) == 1
        assert len(result[0]) == model.dimensions

        await model.close()

    @skip_without_openai
    @pytest.mark.asyncio
    async def test_semantic_similarity(self):
        """Test that semantically similar texts have similar embeddings."""
        model = EmbeddingFactory.create(provider="openai")
        assert model is not None

        texts = [
            "The cat sat on the mat.",
            "A feline rested on the rug.",
            "Python is a programming language.",
        ]
        embeddings = await model.embed(texts)

        def cosine_sim(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b)

        sim_cat_feline = cosine_sim(embeddings[0], embeddings[1])
        sim_cat_python = cosine_sim(embeddings[0], embeddings[2])

        assert sim_cat_feline > sim_cat_python

        await model.close()


class TestLocalEmbeddingE2E:
    """E2E tests for local embedding model."""

    @skip_without_local_model
    @pytest.mark.asyncio
    async def test_embed_single_text(self):
        """Test embedding with local model."""
        model = EmbeddingFactory.create(provider="local")
        assert model is not None

        result = await model.embed(["Hello, world!"])

        assert len(result) == 1
        assert len(result[0]) > 0
        assert all(isinstance(v, float) for v in result[0])

        await model.close()

    @skip_without_local_model
    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self):
        """Test embedding multiple texts with local model."""
        model = EmbeddingFactory.create(provider="local")
        assert model is not None

        texts = ["Text one", "Text two", "Text three"]
        result = await model.embed(texts)

        assert len(result) == 3
        dims = len(result[0])
        for embedding in result:
            assert len(embedding) == dims

        await model.close()


class TestMemoryRetrievalE2E:
    """E2E tests for complete memory retrieval pipeline."""

    @pytest.fixture
    def temp_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            memory_dir = workspace / "MEMORY"
            memory_dir.mkdir()
            yield workspace

    @pytest.mark.asyncio
    async def test_bm25_only_retrieval(self, temp_workspace):
        """Test retrieval with BM25 only (no embedding)."""
        memory_dir = temp_workspace / "MEMORY"

        (memory_dir / "2025-01-15.md").write_text(
            """
# Architecture Decision Record

## Context
We need to choose a database for our new project.

## Decision
We decided to use PostgreSQL for the following reasons:
- Strong JSON support
- Excellent performance
- Good ecosystem

## Consequences
- Need to set up PostgreSQL server
- Team needs PostgreSQL training
"""
        )

        (memory_dir / "2025-01-20.md").write_text(
            """
# Meeting Notes

## Attendees
- Alice, Bob, Charlie

## Discussion
- Reviewed the PostgreSQL decision
- Discussed migration strategy
- Planned training sessions

## Action Items
- Bob: Set up PostgreSQL server
- Alice: Create migration scripts
"""
        )

        store = MemoryIndexStore(temp_workspace, embedding_provider="disabled")

        await store.sync_files()

        results = await store.search("PostgreSQL database decision")
        assert len(results) >= 1
        assert any("PostgreSQL" in r.text for r in results)

        results = await store.search("meeting attendees Alice Bob")
        assert len(results) >= 1
        assert any("Alice" in r.text or "Bob" in r.text for r in results)

        await store.close()

    @skip_without_openai
    @pytest.mark.asyncio
    async def test_hybrid_retrieval_with_openai(self, temp_workspace):
        """Test hybrid retrieval with OpenAI embeddings."""
        memory_dir = temp_workspace / "MEMORY"

        (memory_dir / "tech_decisions.md").write_text(
            """
# Technology Stack Decisions

## Frontend
We chose React for the frontend because:
- Component-based architecture
- Large ecosystem
- Good TypeScript support

## Backend
FastAPI was selected for the backend:
- High performance
- Automatic OpenAPI docs
- Native async support
"""
        )

        (memory_dir / "project_notes.md").write_text(
            """
# Project Notes

## Week 1
- Set up development environment
- Created initial project structure
- Configured CI/CD pipeline

## Week 2
- Implemented user authentication
- Added database migrations
- Set up monitoring
"""
        )

        store = MemoryIndexStore(temp_workspace, embedding_provider="openai")

        await store.sync_files()

        results = await store.search("frontend framework choice")
        assert len(results) >= 1
        found_react = any("React" in r.text for r in results)
        assert found_react, "Should find React in frontend discussion"

        results = await store.search("backend API framework")
        assert len(results) >= 1
        found_fastapi = any("FastAPI" in r.text for r in results)
        assert found_fastapi, "Should find FastAPI in backend discussion"

        results = await store.search("CI/CD setup")
        assert len(results) >= 1
        found_cicd = any("CI/CD" in r.text for r in results)
        assert found_cicd, "Should find CI/CD in project notes"

        await store.close()

    @pytest.mark.asyncio
    async def test_incremental_indexing(self, temp_workspace):
        """Test that incremental indexing works correctly."""
        memory_dir = temp_workspace / "MEMORY"
        store = MemoryIndexStore(temp_workspace, embedding_provider="disabled")

        (memory_dir / "notes.md").write_text("Initial content about Python.")
        await store.sync_files()

        results = await store.search("Python")
        assert len(results) >= 1

        (memory_dir / "notes.md").write_text("Updated content about JavaScript.")
        await store.sync_files()

        results = await store.search("JavaScript")
        assert len(results) >= 1
        assert "JavaScript" in results[0].text

        results = await store.search("Python")
        python_found = any("Python" in r.text for r in results)
        assert not python_found, "Old content should be replaced"

        await store.close()

    @pytest.mark.asyncio
    async def test_file_deletion_handling(self, temp_workspace):
        """Test that deleted files are removed from index."""
        memory_dir = temp_workspace / "MEMORY"
        store = MemoryIndexStore(temp_workspace, embedding_provider="disabled")

        temp_file = memory_dir / "temporary.md"
        temp_file.write_text("Temporary content about Redis caching.")
        await store.sync_files()

        results = await store.search("Redis caching")
        assert len(results) >= 1

        temp_file.unlink()
        await store.sync_files()

        results = await store.search("Redis caching")
        assert len(results) == 0

        await store.close()

    @pytest.mark.asyncio
    async def test_chinese_content_retrieval(self, temp_workspace):
        """Test retrieval of Chinese content."""
        memory_dir = temp_workspace / "MEMORY"

        (memory_dir / "chinese_notes.md").write_text(
            """
# 项目笔记

## 技术选型
我们选择了 PostgreSQL 作为数据库，因为它支持 JSON 类型。

## 架构设计
采用微服务架构，使用 Docker 容器化部署。

## 下一步计划
- 完成用户认证模块
- 实现数据同步功能
"""
        )

        store = MemoryIndexStore(temp_workspace, embedding_provider="disabled")

        await store.sync_files()

        results = await store.search("PostgreSQL 数据库")
        assert len(results) >= 1

        results = await store.search("微服务 Docker")
        assert len(results) >= 1

        await store.close()

    @pytest.mark.asyncio
    async def test_multiple_files_ranking(self, temp_workspace):
        """Test that results from multiple files are properly ranked."""
        memory_dir = temp_workspace / "MEMORY"

        (memory_dir / "file1.md").write_text(
            "This document is about cooking recipes and food preparation."
        )
        (memory_dir / "file2.md").write_text(
            """
# Machine Learning Guide

Machine learning is a powerful technique for building intelligent systems.
This document covers various machine learning algorithms including:
- Supervised learning
- Unsupervised learning
- Reinforcement learning

Machine learning applications are everywhere.
"""
        )
        (memory_dir / "file3.md").write_text("Unrelated content about gardening tips.")

        store = MemoryIndexStore(temp_workspace, embedding_provider="disabled")

        await store.sync_files()

        results = await store.search("machine learning algorithms", top_k=3)
        assert len(results) >= 1
        found_ml = any("file2.md" in r.path for r in results)
        assert found_ml, "Should find machine learning content"
        assert not any("file3.md" in r.path for r in results[:2]), (
            "Gardening should not rank high"
        )

        await store.close()

    @pytest.mark.asyncio
    async def test_line_number_accuracy(self, temp_workspace):
        """Test that line numbers in results are accurate."""
        memory_dir = temp_workspace / "MEMORY"

        content = "\n".join(
            [
                "Line 1: Introduction",
                "Line 2: Background",
                "Line 3: The important keyword appears here",
                "Line 4: More content",
                "Line 5: Conclusion",
            ]
        )
        (memory_dir / "test.md").write_text(content)

        store = MemoryIndexStore(temp_workspace, embedding_provider="disabled")

        await store.sync_files()

        results = await store.search("important keyword")
        assert len(results) >= 1
        assert results[0].start_line >= 1
        assert results[0].end_line >= results[0].start_line

        await store.close()
