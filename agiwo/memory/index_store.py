"""
MemoryIndexStore - SQLite-based memory indexing and retrieval.
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import aiosqlite

from agiwo.config.settings import get_settings
from agiwo.embedding import EmbeddingError, EmbeddingFactory, EmbeddingModel
from agiwo.memory.chunker import MemoryChunk, MemoryChunker
from agiwo.memory.searcher import HybridSearcher, SearchResult
from agiwo.utils.logging import get_logger
from agiwo.utils.storage_support.sqlite_runtime import (
    SQLiteConnectionRuntime,
    execute_statements,
)

logger = get_logger(__name__)


class MemoryIndexStore:
    """SQLite-based memory index store with hybrid search support."""

    def __init__(
        self,
        workspace_dir: Path,
        *,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        embedding_dims: int | None = None,
        embedding_api_key: str | None = None,
        embedding_api_base: str | None = None,
        chunk_tokens: int | None = None,
        chunk_overlap_tokens: int | None = None,
        top_k: int | None = None,
    ):
        self._workspace_dir = Path(workspace_dir)
        self._memory_dir = self._workspace_dir / "MEMORY"
        self._db_path = self._workspace_dir / "memory.db"

        _s = get_settings()
        self._embedding_provider = embedding_provider or _s.embedding_provider
        self._embedding_model = embedding_model or _s.embedding_model
        self._embedding_dims = (
            embedding_dims if embedding_dims is not None else _s.embedding_dimensions
        )
        self._embedding_api_key = embedding_api_key or _s.get_embedding_api_key() or ""
        self._embedding_api_base = embedding_api_base or _s.embedding_base_url or ""
        self._top_k = top_k

        self._rt = SQLiteConnectionRuntime(
            str(self._db_path),
            logger=logger,
            connect_event="memory_store_connected",
        )
        self._embedder: EmbeddingModel | None = None
        self._vec_available = False
        self._initialized = False

        self._chunker = MemoryChunker(
            chunk_tokens=(
                chunk_tokens if chunk_tokens is not None else _s.memory_chunk_tokens
            ),
            overlap_tokens=(
                chunk_overlap_tokens
                if chunk_overlap_tokens is not None
                else _s.memory_chunk_overlap
            ),
        )

    async def ensure_initialized(self) -> None:
        if self._initialized:
            return

        self._memory_dir.mkdir(parents=True, exist_ok=True)
        await self._rt.ensure_connection(self._initialize)
        self._initialized = True
        logger.info(
            "memory_store_initialized",
            db_path=str(self._db_path),
            vec_available=self._vec_available,
            embedder_available=self._embedder is not None,
        )

    async def _initialize(self, conn: aiosqlite.Connection) -> None:
        await execute_statements(
            conn,
            [
                """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                mtime INTEGER NOT NULL,
                size INTEGER NOT NULL
            )
        """,
                """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                model_id TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL,
                embedding TEXT NOT NULL DEFAULT '',
                updated_at INTEGER NOT NULL
            )
        """,
                "CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)",
                """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                content_hash TEXT NOT NULL,
                model_id TEXT NOT NULL,
                embedding TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (content_hash, model_id)
            )
        """,
            ],
        )
        await conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                chunk_id UNINDEXED,
                path UNINDEXED,
                start_line UNINDEXED,
                end_line UNINDEXED,
                tokenize = "unicode61"
            )
        """
        )
        await conn.commit()

        self._vec_available = await self._check_vec_extension(conn)

        try:
            self._embedder = EmbeddingFactory.create(
                provider=self._embedding_provider,
                model=self._embedding_model,
                dimensions=self._embedding_dims,
                api_key=self._embedding_api_key or None,
                base_url=self._embedding_api_base or None,
            )
        except EmbeddingError as exc:
            logger.warning("embedder_init_failed", error=str(exc))
            self._embedder = None

    async def _check_vec_extension(self, conn: aiosqlite.Connection) -> bool:
        try:
            await conn.enable_load_extension(True)
            await conn.load_extension("vec0")

            dims = self._embedding_dims
            await conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                    chunk_id TEXT PRIMARY KEY,
                    embedding FLOAT[{dims}]
                )
            """
            )
            await conn.commit()
            return True
        except (sqlite3.OperationalError, AttributeError) as exc:
            logger.info("sqlite_vec_not_available", reason=str(exc))
            return False

    async def sync_files(self) -> None:
        await self.ensure_initialized()

        if not self._memory_dir.exists():
            return

        conn = self._rt.connection
        assert conn is not None

        current_files: dict[str, tuple[int, int]] = {}
        for file_path in self._memory_dir.glob("*.md"):
            stat = file_path.stat()
            rel_path = str(file_path.relative_to(self._workspace_dir))
            current_files[rel_path] = (int(stat.st_mtime), stat.st_size)

        async with conn.execute("SELECT path, mtime, size FROM files") as cursor:
            rows = await cursor.fetchall()
        indexed_files = {row["path"]: (row["mtime"], row["size"]) for row in rows}

        new_or_changed = []
        for path, file_state in current_files.items():
            if path not in indexed_files or indexed_files[path] != file_state:
                new_or_changed.append(path)

        deleted = set(indexed_files.keys()) - set(current_files.keys())

        for path in new_or_changed:
            await self._index_file(path)

        for path in deleted:
            await self._remove_file(path)

        if new_or_changed or deleted:
            logger.info(
                "memory_sync_complete",
                indexed=len(new_or_changed),
                deleted=len(deleted),
            )

    async def _index_file(self, rel_path: str) -> None:
        full_path = self._workspace_dir / rel_path
        if not full_path.exists():
            return

        content = full_path.read_text(encoding="utf-8", errors="replace")
        file_hash = hashlib.sha256(content.encode()).hexdigest()

        conn = self._rt.connection
        assert conn is not None

        async with conn.execute(
            "SELECT hash FROM files WHERE path = ?", (rel_path,)
        ) as cursor:
            row = await cursor.fetchone()
        if row and row["hash"] == file_hash:
            return

        chunks = self._chunker.chunk_file(full_path, content)
        if not chunks:
            return

        embeddings: dict[str, list[float]] = {}
        if self._embedder:
            embeddings = await self._get_embeddings(chunks)

        await self._remove_file_chunks(rel_path)

        now = int(time.time())
        model_id = self._embedder.model_id if self._embedder else ""

        for chunk in chunks:
            embedding = embeddings.get(chunk.content_hash, [])
            embedding_json = json.dumps(embedding) if embedding else ""

            await conn.execute(
                """
                INSERT INTO chunks (chunk_id, path, start_line, end_line,
                    content_hash, model_id, text, embedding, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    rel_path,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.content_hash,
                    model_id,
                    chunk.text,
                    embedding_json,
                    now,
                ),
            )

            await conn.execute(
                """
                INSERT INTO chunks_fts (text, chunk_id, path, start_line, end_line)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    chunk.text,
                    chunk.chunk_id,
                    rel_path,
                    chunk.start_line,
                    chunk.end_line,
                ),
            )

            if self._vec_available and embedding:
                vec_str = "[" + ",".join(str(value) for value in embedding) + "]"
                await conn.execute(
                    "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                    (chunk.chunk_id, vec_str),
                )

        stat = full_path.stat()
        await conn.execute(
            """
            INSERT OR REPLACE INTO files (path, hash, mtime, size)
            VALUES (?, ?, ?, ?)
            """,
            (rel_path, file_hash, int(stat.st_mtime), stat.st_size),
        )

        await conn.commit()
        logger.debug("file_indexed", path=rel_path, chunks=len(chunks))

    async def _get_embeddings(
        self, chunks: list[MemoryChunk]
    ) -> dict[str, list[float]]:
        if not self._embedder:
            return {}

        model_id = self._embedder.model_id
        conn = self._rt.connection
        assert conn is not None

        content_hashes = [chunk.content_hash for chunk in chunks]
        placeholders = ",".join("?" * len(content_hashes))
        async with conn.execute(
            f"""
            SELECT content_hash, embedding FROM embedding_cache
            WHERE model_id = ? AND content_hash IN ({placeholders})
            """,
            [model_id] + content_hashes,
        ) as cursor:
            rows = await cursor.fetchall()

        cached: dict[str, list[float]] = {}
        for row in rows:
            try:
                cached[row["content_hash"]] = json.loads(row["embedding"])
            except json.JSONDecodeError:
                continue

        missing_chunks = [chunk for chunk in chunks if chunk.content_hash not in cached]

        if missing_chunks:
            try:
                texts = [chunk.text for chunk in missing_chunks]
                new_embeddings = await self._embedder.embed(texts)

                now = int(time.time())
                for chunk, embedding in zip(missing_chunks, new_embeddings):
                    cached[chunk.content_hash] = embedding
                    await conn.execute(
                        """
                        INSERT OR REPLACE INTO embedding_cache
                        (content_hash, model_id, embedding, updated_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (chunk.content_hash, model_id, json.dumps(embedding), now),
                    )
                await conn.commit()

            except EmbeddingError as exc:
                logger.warning("embedding_failed", error=str(exc))

        return cached

    async def _remove_file(self, rel_path: str) -> None:
        await self._remove_file_chunks(rel_path)
        conn = self._rt.connection
        assert conn is not None
        await conn.execute("DELETE FROM files WHERE path = ?", (rel_path,))
        await conn.commit()

    async def _remove_file_chunks(self, rel_path: str) -> None:
        conn = self._rt.connection
        assert conn is not None

        async with conn.execute(
            "SELECT chunk_id FROM chunks WHERE path = ?", (rel_path,)
        ) as cursor:
            rows = await cursor.fetchall()
        chunk_ids = [row["chunk_id"] for row in rows]

        if chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            await conn.execute(
                f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
            if self._vec_available:
                await conn.execute(
                    f"DELETE FROM chunks_vec WHERE chunk_id IN ({placeholders})",
                    chunk_ids,
                )

        await conn.execute("DELETE FROM chunks WHERE path = ?", (rel_path,))

    async def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        await self.ensure_initialized()

        conn = self._rt.connection
        assert conn is not None

        searcher = HybridSearcher(
            conn=conn,
            embedder=self._embedder,
            vec_available=self._vec_available,
        )

        return await searcher.search(query, top_k)

    async def close(self) -> None:
        await self._rt.disconnect()
        self._initialized = False


__all__ = ["MemoryIndexStore"]
