"""
MemoryIndexStore - SQLite-based memory indexing and retrieval.
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from agiwo.config.settings import settings
from agiwo.embedding import EmbeddingError, EmbeddingFactory, EmbeddingModel
from agiwo.memory.chunker import MemoryChunk, MemoryChunker
from agiwo.memory.searcher import HybridSearcher, SearchResult
from agiwo.utils.logging import get_logger

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

        self._embedding_provider = embedding_provider or settings.embedding_provider
        self._embedding_model = embedding_model or settings.embedding_model
        self._embedding_dims = (
            embedding_dims
            if embedding_dims is not None
            else settings.embedding_dimensions
        )
        self._embedding_api_key = (
            embedding_api_key or settings.get_embedding_api_key() or ""
        )
        self._embedding_api_base = (
            embedding_api_base or settings.embedding_base_url or ""
        )
        self._top_k = top_k

        self._conn: sqlite3.Connection | None = None
        self._embedder: EmbeddingModel | None = None
        self._vec_available = False
        self._initialized = False

        self._chunker = MemoryChunker(
            chunk_tokens=(
                chunk_tokens
                if chunk_tokens is not None
                else settings.memory_chunk_tokens
            ),
            overlap_tokens=(
                chunk_overlap_tokens
                if chunk_overlap_tokens is not None
                else settings.memory_chunk_overlap
            ),
        )

    async def ensure_initialized(self) -> None:
        """Initialize database and embedder lazily."""
        if self._initialized:
            return

        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        self._init_schema()
        self._vec_available = self._check_vec_extension()

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

        self._initialized = True
        logger.info(
            "memory_store_initialized",
            db_path=str(self._db_path),
            vec_available=self._vec_available,
            embedder_available=self._embedder is not None,
        )

    def _init_schema(self) -> None:
        """Initialize database schema."""
        cursor = self._conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                mtime INTEGER NOT NULL,
                size INTEGER NOT NULL
            )
        """
        )

        cursor.execute(
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
        """
        )

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                content_hash TEXT NOT NULL,
                model_id TEXT NOT NULL,
                embedding TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (content_hash, model_id)
            )
        """
        )

        cursor.execute(
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

        self._conn.commit()

    def _check_vec_extension(self) -> bool:
        """Check if sqlite-vec extension is available."""
        try:
            self._conn.enable_load_extension(True)
            self._conn.load_extension("vec0")

            dims = self._embedding_dims
            cursor = self._conn.cursor()
            cursor.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                    chunk_id TEXT PRIMARY KEY,
                    embedding FLOAT[{dims}]
                )
            """
            )
            self._conn.commit()
            return True
        except (sqlite3.OperationalError, AttributeError) as exc:
            logger.info("sqlite_vec_not_available", reason=str(exc))
            return False

    async def sync_files(self) -> None:
        """Sync MEMORY directory with index."""
        await self.ensure_initialized()

        if not self._memory_dir.exists():
            return

        current_files: dict[str, tuple[int, int]] = {}
        for file_path in self._memory_dir.glob("*.md"):
            stat = file_path.stat()
            rel_path = str(file_path.relative_to(self._workspace_dir))
            current_files[rel_path] = (int(stat.st_mtime), stat.st_size)

        cursor = self._conn.cursor()
        cursor.execute("SELECT path, mtime, size FROM files")
        indexed_files = {
            row["path"]: (row["mtime"], row["size"]) for row in cursor.fetchall()
        }

        new_or_changed = []
        for path, file_state in current_files.items():
            if path not in indexed_files or indexed_files[path] != file_state:
                new_or_changed.append(path)

        deleted = set(indexed_files.keys()) - set(current_files.keys())

        for path in new_or_changed:
            await self._index_file(path)

        for path in deleted:
            self._remove_file(path)

        if new_or_changed or deleted:
            logger.info(
                "memory_sync_complete",
                indexed=len(new_or_changed),
                deleted=len(deleted),
            )

    async def _index_file(self, rel_path: str) -> None:
        """Index a single file."""
        full_path = self._workspace_dir / rel_path
        if not full_path.exists():
            return

        content = full_path.read_text(encoding="utf-8", errors="replace")
        file_hash = hashlib.sha256(content.encode()).hexdigest()

        cursor = self._conn.cursor()
        cursor.execute("SELECT hash FROM files WHERE path = ?", (rel_path,))
        row = cursor.fetchone()
        if row and row["hash"] == file_hash:
            return

        chunks = self._chunker.chunk_file(full_path, content)
        if not chunks:
            return

        embeddings: dict[str, list[float]] = {}
        if self._embedder:
            embeddings = await self._get_embeddings(chunks)

        self._remove_file_chunks(rel_path)

        now = int(time.time())
        model_id = self._embedder.model_id if self._embedder else ""

        for chunk in chunks:
            embedding = embeddings.get(chunk.content_hash, [])
            embedding_json = json.dumps(embedding) if embedding else ""

            cursor.execute(
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

            cursor.execute(
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
                cursor.execute(
                    "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                    (chunk.chunk_id, vec_str),
                )

        stat = full_path.stat()
        cursor.execute(
            """
            INSERT OR REPLACE INTO files (path, hash, mtime, size)
            VALUES (?, ?, ?, ?)
            """,
            (rel_path, file_hash, int(stat.st_mtime), stat.st_size),
        )

        self._conn.commit()
        logger.debug("file_indexed", path=rel_path, chunks=len(chunks))

    async def _get_embeddings(
        self, chunks: list[MemoryChunk]
    ) -> dict[str, list[float]]:
        """Get embeddings for chunks, using cache when possible."""
        if not self._embedder:
            return {}

        model_id = self._embedder.model_id
        cursor = self._conn.cursor()

        content_hashes = [chunk.content_hash for chunk in chunks]
        placeholders = ",".join("?" * len(content_hashes))
        cursor.execute(
            f"""
            SELECT content_hash, embedding FROM embedding_cache
            WHERE model_id = ? AND content_hash IN ({placeholders})
            """,
            [model_id] + content_hashes,
        )

        cached: dict[str, list[float]] = {}
        for row in cursor.fetchall():
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
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO embedding_cache
                        (content_hash, model_id, embedding, updated_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (chunk.content_hash, model_id, json.dumps(embedding), now),
                    )
                self._conn.commit()

            except EmbeddingError as exc:
                logger.warning("embedding_failed", error=str(exc))

        return cached

    def _remove_file(self, rel_path: str) -> None:
        """Remove a file from the index."""
        self._remove_file_chunks(rel_path)
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM files WHERE path = ?", (rel_path,))
        self._conn.commit()

    def _remove_file_chunks(self, rel_path: str) -> None:
        """Remove chunks for a file."""
        cursor = self._conn.cursor()

        cursor.execute("SELECT chunk_id FROM chunks WHERE path = ?", (rel_path,))
        chunk_ids = [row["chunk_id"] for row in cursor.fetchall()]

        if chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            cursor.execute(
                f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
            if self._vec_available:
                cursor.execute(
                    f"DELETE FROM chunks_vec WHERE chunk_id IN ({placeholders})",
                    chunk_ids,
                )

        cursor.execute("DELETE FROM chunks WHERE path = ?", (rel_path,))

    async def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Search memories using hybrid search."""
        await self.ensure_initialized()

        searcher = HybridSearcher(
            conn=self._conn,
            embedder=self._embedder,
            vec_available=self._vec_available,
        )

        return await searcher.search(query, top_k)

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            self._initialized = False


__all__ = ["MemoryIndexStore"]
