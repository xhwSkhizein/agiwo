"""
HybridSearcher - BM25 + Vector hybrid search with optional MMR and temporal decay.
"""

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from agiwo.config.settings import get_settings
from agiwo.embedding import EmbeddingModel
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

STOP_WORDS_ZH = {
    "的",
    "了",
    "是",
    "在",
    "和",
    "与",
    "或",
    "也",
    "都",
    "很",
    "就",
    "不",
    "有",
    "这",
    "那",
    "我",
    "你",
    "他",
    "她",
    "它",
    "们",
    "个",
    "为",
    "以",
    "及",
    "等",
    "被",
    "把",
    "让",
    "给",
    "从",
    "到",
    "对",
    "于",
    "而",
    "但",
}
STOP_WORDS_EN = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    "dare",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "up",
    "about",
    "into",
    "over",
    "after",
    "and",
    "or",
    "but",
    "if",
    "then",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "what",
    "which",
    "who",
    "whom",
    "how",
    "when",
    "where",
    "why",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "as",
}
STOP_WORDS = STOP_WORDS_ZH | STOP_WORDS_EN


@dataclass
class SearchResult:
    """A search result from hybrid search."""

    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str
    score: float
    vector_score: float
    bm25_score: float


class HybridSearcher:
    """Hybrid BM25 + Vector searcher with optional MMR and temporal decay."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        embedder: EmbeddingModel | None,
        vec_available: bool,
        *,
        top_k: int | None = None,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
        temporal_decay_enabled: bool | None = None,
        temporal_decay_half_life_days: float | None = None,
        mmr_enabled: bool | None = None,
        mmr_lambda: float | None = None,
    ):
        self._conn = conn
        self._embedder = embedder
        self._vec_available = vec_available
        _s = get_settings()
        self._top_k = top_k if top_k is not None else _s.memory_top_k
        self._vector_weight = (
            vector_weight if vector_weight is not None else _s.memory_vector_weight
        )
        self._bm25_weight = (
            bm25_weight if bm25_weight is not None else _s.memory_bm25_weight
        )
        self._temporal_decay_enabled = (
            temporal_decay_enabled
            if temporal_decay_enabled is not None
            else _s.memory_temporal_decay
        )
        self._temporal_decay_half_life_days = (
            temporal_decay_half_life_days
            if temporal_decay_half_life_days is not None
            else _s.memory_temporal_decay_half_life
        )
        self._mmr_enabled = (
            mmr_enabled if mmr_enabled is not None else _s.memory_mmr_enabled
        )
        self._mmr_lambda = (
            mmr_lambda if mmr_lambda is not None else _s.memory_mmr_lambda
        )

    async def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Execute hybrid search with detailed logging."""
        top_k = top_k or self._top_k
        fetch_limit = top_k * 5

        logger.info(
            "hybrid_search_start", query=query, top_k=top_k, fetch_limit=fetch_limit
        )
        logger.debug(
            "search_config",
            vector_weight=self._vector_weight,
            bm25_weight=self._bm25_weight,
            mmr_enabled=self._mmr_enabled,
        )

        vector_results: dict[str, float] = {}
        bm25_results: dict[str, float] = {}

        if self._embedder:
            vec_start = datetime.now()
            vector_results = await self._vector_search(query, fetch_limit)
            vec_duration_ms = (datetime.now() - vec_start).total_seconds() * 1000
            logger.info(
                "vector_search_complete",
                query=query,
                result_count=len(vector_results),
                duration_ms=vec_duration_ms,
                top_results=list(vector_results.items())[:3] if vector_results else [],
            )
        else:
            logger.warning("vector_search_skipped", reason="no_embedder")

        bm25_start = datetime.now()
        bm25_results = self._bm25_search(
            query, fetch_limit, use_expansion=not self._embedder
        )
        bm25_duration_ms = (datetime.now() - bm25_start).total_seconds() * 1000
        logger.info(
            "bm25_search_complete",
            query=query,
            result_count=len(bm25_results),
            duration_ms=bm25_duration_ms,
            top_results=list(bm25_results.items())[:3] if bm25_results else [],
        )

        merged = self._merge_results(vector_results, bm25_results)
        logger.info(
            "results_merged",
            query=query,
            merged_count=len(merged),
            vector_only=len(
                [
                    k
                    for k, v in merged.items()
                    if v["vector_score"] > 0 and v["bm25_score"] == 0
                ]
            ),
            bm25_only=len(
                [
                    k
                    for k, v in merged.items()
                    if v["bm25_score"] > 0 and v["vector_score"] == 0
                ]
            ),
            both=len(
                [
                    k
                    for k, v in merged.items()
                    if v["vector_score"] > 0 and v["bm25_score"] > 0
                ]
            ),
            top_merged=(
                sorted(merged.items(), key=lambda x: x[1]["score"], reverse=True)[:3]
                if merged
                else []
            ),
        )

        if self._temporal_decay_enabled:
            merged = self._apply_temporal_decay(merged)
            logger.debug("temporal_decay_applied")

        sorted_results = sorted(
            merged.items(), key=lambda x: x[1]["score"], reverse=True
        )
        sorted_results = sorted_results[: top_k * 2]

        results = self._build_results(sorted_results)
        logger.debug("results_built", count=len(results))

        if self._mmr_enabled and len(results) > top_k:
            results_before_mmr = results
            results = self._mmr_rerank(results, top_k)
            logger.info(
                "mmr_rerank_complete",
                input_count=len(results_before_mmr),
                output_count=len(results),
                top_k=top_k,
            )
        else:
            results = results[:top_k]

        logger.info(
            "hybrid_search_complete",
            query=query,
            final_result_count=len(results),
            top_scores=[
                {
                    "path": result.path,
                    "score": round(result.score, 4),
                    "vector": round(result.vector_score, 4),
                    "bm25": round(result.bm25_score, 4),
                }
                for result in results[:3]
            ],
        )

        return results

    async def _vector_search(self, query: str, limit: int) -> dict[str, float]:
        """Search using vector similarity."""
        if not self._embedder:
            return {}

        query_embedding = (await self._embedder.embed([query]))[0]
        if self._vec_available:
            return self._vector_search_sqlite_vec(query_embedding, limit)
        return self._vector_search_memory(query_embedding, limit)

    def _vector_search_sqlite_vec(
        self, query_vec: list[float], limit: int
    ) -> dict[str, float]:
        """Vector search using sqlite-vec extension."""
        cursor = self._conn.cursor()
        vec_str = "[" + ",".join(str(value) for value in query_vec) + "]"

        cursor.execute(
            """
            SELECT chunk_id, vec_distance_cosine(embedding, ?) as distance
            FROM chunks_vec
            ORDER BY distance
            LIMIT ?
            """,
            (vec_str, limit),
        )

        results = {}
        for row in cursor.fetchall():
            chunk_id, distance = row
            score = 1.0 - distance
            results[chunk_id] = max(0.0, min(1.0, score))

        return results

    _VECTOR_FALLBACK_MAX = 10_000

    def _vector_search_memory(
        self, query_vec: list[float], limit: int
    ) -> dict[str, float]:
        """Fallback vector search in memory (no sqlite-vec)."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT chunk_id, embedding FROM chunks WHERE embedding != '' LIMIT ?",
            (self._VECTOR_FALLBACK_MAX,),
        )

        results: list[tuple[str, float]] = []
        for row in cursor.fetchall():
            chunk_id, embedding_json = row
            if not embedding_json:
                continue
            try:
                embedding = json.loads(embedding_json)
                score = self._cosine_similarity(query_vec, embedding)
                results.append((chunk_id, score))
            except (json.JSONDecodeError, TypeError):
                continue

        results.sort(key=lambda item: item[1], reverse=True)
        return {chunk_id: score for chunk_id, score in results[:limit]}

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _bm25_search(
        self, query: str, limit: int, use_expansion: bool = False
    ) -> dict[str, float]:
        """Search using FTS5 BM25."""
        cursor = self._conn.cursor()

        if use_expansion:
            keywords = self._extract_keywords(query)
            if not keywords:
                keywords = [query]
        else:
            keywords = [query]

        all_results: dict[str, float] = {}
        for keyword in keywords:
            safe_query = self._sanitize_fts_query(keyword)
            if not safe_query:
                continue

            try:
                cursor.execute(
                    """
                    SELECT chunk_id, bm25(chunks_fts) as rank
                    FROM chunks_fts
                    WHERE text MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (safe_query, limit),
                )
            except sqlite3.OperationalError as exc:
                logger.warning("fts_search_failed", query=safe_query, error=str(exc))
                continue

            for row in cursor.fetchall():
                chunk_id = row[0]
                rank = row[1]
                score = 1.0 / (1.0 + abs(rank))
                all_results[chunk_id] = max(all_results.get(chunk_id, 0.0), score)

        return all_results

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords for query expansion."""
        tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", text.lower())
        return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]

    def _sanitize_fts_query(self, query: str) -> str:
        """Sanitize query for FTS5."""
        query = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", query)
        query = re.sub(r"\s+", " ", query).strip()
        return query

    def _merge_results(
        self, vector_results: dict[str, float], bm25_results: dict[str, float]
    ) -> dict[str, dict[str, float]]:
        """Merge vector and BM25 results."""
        all_chunk_ids = set(vector_results.keys()) | set(bm25_results.keys())
        merged = {}

        for chunk_id in all_chunk_ids:
            vector_score = vector_results.get(chunk_id, 0.0)
            bm25_score = bm25_results.get(chunk_id, 0.0)
            merged[chunk_id] = {
                "vector_score": vector_score,
                "bm25_score": bm25_score,
                "score": (
                    self._vector_weight * vector_score + self._bm25_weight * bm25_score
                ),
            }

        return merged

    def _apply_temporal_decay(
        self, merged: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        """Apply temporal decay based on file mtime."""
        if not merged:
            return merged

        cursor = self._conn.cursor()
        now = datetime.now().timestamp()
        half_life_seconds = self._temporal_decay_half_life_days * 24 * 3600

        chunk_ids = list(merged.keys())
        placeholders = ",".join("?" * len(chunk_ids))
        cursor.execute(
            f"""
            SELECT c.chunk_id, f.mtime
            FROM chunks c
            JOIN files f ON c.path = f.path
            WHERE c.chunk_id IN ({placeholders})
            """,
            chunk_ids,
        )
        mtime_map = {row[0]: row[1] for row in cursor.fetchall()}

        for chunk_id, scores in merged.items():
            mtime = mtime_map.get(chunk_id)
            if mtime is not None:
                age_seconds = now - mtime
                decay = math.exp(-math.log(2) * age_seconds / half_life_seconds)
                scores["score"] *= decay

        return merged

    def _build_results(
        self, sorted_results: list[tuple[str, dict[str, float]]]
    ) -> list[SearchResult]:
        """Build SearchResult objects from sorted scores."""
        if not sorted_results:
            return []

        cursor = self._conn.cursor()
        chunk_ids = [cid for cid, _ in sorted_results]
        placeholders = ",".join("?" * len(chunk_ids))
        cursor.execute(
            f"""
            SELECT chunk_id, path, start_line, end_line, text
            FROM chunks
            WHERE chunk_id IN ({placeholders})
            """,
            chunk_ids,
        )
        chunk_data = {row[0]: row for row in cursor.fetchall()}

        results: list[SearchResult] = []
        for chunk_id, scores in sorted_results:
            row = chunk_data.get(chunk_id)
            if row:
                results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        path=row[1],
                        start_line=row[2],
                        end_line=row[3],
                        text=row[4],
                        score=scores["score"],
                        vector_score=scores["vector_score"],
                        bm25_score=scores["bm25_score"],
                    )
                )

        return results

    def _mmr_rerank(
        self, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """Apply Maximal Marginal Relevance re-ranking."""
        if not results:
            return []

        selected = [results[0]]
        remaining = results[1:]

        while remaining and len(selected) < top_k:
            mmr_scores = []

            for candidate in remaining:
                max_similarity = 0.0
                candidate_tokens = set(self._extract_keywords(candidate.text))
                for selected_result in selected:
                    selected_tokens = set(self._extract_keywords(selected_result.text))
                    if candidate_tokens and selected_tokens:
                        intersection = len(candidate_tokens & selected_tokens)
                        union = len(candidate_tokens | selected_tokens)
                        similarity = intersection / union if union > 0 else 0.0
                        max_similarity = max(max_similarity, similarity)

                mmr_score = (
                    self._mmr_lambda * candidate.score
                    - (1 - self._mmr_lambda) * max_similarity
                )
                mmr_scores.append((candidate, mmr_score))

            mmr_scores.sort(key=lambda item: item[1], reverse=True)
            best = mmr_scores[0][0]
            selected.append(best)
            remaining.remove(best)

        return selected


__all__ = ["HybridSearcher", "SearchResult"]
