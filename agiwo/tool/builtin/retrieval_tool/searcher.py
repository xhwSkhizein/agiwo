"""
HybridSearcher - BM25 + Vector hybrid search with optional MMR and temporal decay.
"""

import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from agiwo.embedding import EmbeddingModel
from agiwo.tool.builtin.config import MemoryConfig
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

STOP_WORDS_ZH = {
    "的", "了", "是", "在", "和", "与", "或", "也", "都", "很", "就", "不",
    "有", "这", "那", "我", "你", "他", "她", "它", "们", "个", "为", "以",
    "及", "等", "被", "把", "让", "给", "从", "到", "对", "于", "而", "但",
}
STOP_WORDS_EN = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "up",
    "about", "into", "over", "after", "and", "or", "but", "if", "then",
    "this", "that", "these", "those", "it", "its", "i", "you", "he",
    "she", "we", "they", "what", "which", "who", "whom", "how", "when",
    "where", "why", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "as",
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
        config: MemoryConfig,
        embedder: EmbeddingModel | None,
        vec_available: bool,
    ):
        self._conn = conn
        self._config = config
        self._embedder = embedder
        self._vec_available = vec_available

    async def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Execute hybrid search with detailed logging."""
        top_k = top_k or self._config.top_k
        fetch_limit = top_k * 5

        logger.info("hybrid_search_start", query=query, top_k=top_k, fetch_limit=fetch_limit)
        logger.debug("search_config", vector_weight=self._config.vector_weight, bm25_weight=self._config.bm25_weight, mmr_enabled=self._config.mmr_enabled)

        vector_results: dict[str, float] = {}
        bm25_results: dict[str, float] = {}

        # Vector search
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

        # BM25 search
        bm25_start = datetime.now()
        bm25_results = self._bm25_search(query, fetch_limit, use_expansion=not self._embedder)
        bm25_duration_ms = (datetime.now() - bm25_start).total_seconds() * 1000
        logger.info(
            "bm25_search_complete",
            query=query,
            result_count=len(bm25_results),
            duration_ms=bm25_duration_ms,
            top_results=list(bm25_results.items())[:3] if bm25_results else [],
        )

        # Merge results
        merged = self._merge_results(vector_results, bm25_results)
        logger.info(
            "results_merged",
            query=query,
            merged_count=len(merged),
            vector_only=len([k for k, v in merged.items() if v["vector_score"] > 0 and v["bm25_score"] == 0]),
            bm25_only=len([k for k, v in merged.items() if v["bm25_score"] > 0 and v["vector_score"] == 0]),
            both=len([k for k, v in merged.items() if v["vector_score"] > 0 and v["bm25_score"] > 0]),
            top_merged=sorted(merged.items(), key=lambda x: x[1]["score"], reverse=True)[:3] if merged else [],
        )

        if self._config.temporal_decay_enabled:
            merged = self._apply_temporal_decay(merged)
            logger.debug("temporal_decay_applied")

        sorted_results = sorted(merged.items(), key=lambda x: x[1]["score"], reverse=True)
        sorted_results = sorted_results[: top_k * 2]

        results = self._build_results(sorted_results)
        logger.debug("results_built", count=len(results))

        if self._config.mmr_enabled and len(results) > top_k:
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
            top_scores=[{"path": r.path, "score": round(r.score, 4), "vector": round(r.vector_score, 4), "bm25": round(r.bm25_score, 4)} for r in results[:3]],
        )

        return results

    async def _vector_search(
        self, query: str, limit: int
    ) -> dict[str, float]:
        """Search using vector similarity."""
        if not self._embedder:
            return {}

        query_embedding = (await self._embedder.embed([query]))[0]

        if self._vec_available:
            return self._vector_search_sqlite_vec(query_embedding, limit)
        else:
            return self._vector_search_memory(query_embedding, limit)

    def _vector_search_sqlite_vec(
        self, query_vec: list[float], limit: int
    ) -> dict[str, float]:
        """Vector search using sqlite-vec extension."""
        cursor = self._conn.cursor()
        vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"

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

    def _vector_search_memory(
        self, query_vec: list[float], limit: int
    ) -> dict[str, float]:
        """Fallback vector search in memory (no sqlite-vec)."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT chunk_id, embedding FROM chunks WHERE embedding != ''")

        results: list[tuple[str, float]] = []
        for row in cursor.fetchall():
            chunk_id, embedding_json = row
            if not embedding_json:
                continue
            try:
                import json
                embedding = json.loads(embedding_json)
                score = self._cosine_similarity(query_vec, embedding)
                results.append((chunk_id, score))
            except (json.JSONDecodeError, TypeError):
                continue

        results.sort(key=lambda x: x[1], reverse=True)
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

                for row in cursor.fetchall():
                    chunk_id, rank = row
                    score = 1.0 / (1.0 + abs(rank))
                    if chunk_id not in all_results or all_results[chunk_id] < score:
                        all_results[chunk_id] = score

            except sqlite3.OperationalError as e:
                logger.warning("fts_search_error", query=safe_query, error=str(e))
                continue

        return all_results

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract keywords from query for FTS expansion."""
        zh_pattern = re.compile(r"[\u4e00-\u9fff]+")
        zh_matches = zh_pattern.findall(query)

        remaining = zh_pattern.sub(" ", query)
        en_words = remaining.split()

        keywords = []
        for word in zh_matches:
            if word not in STOP_WORDS and len(word) >= 2:
                keywords.append(word)

        for word in en_words:
            word_lower = word.lower().strip()
            if word_lower not in STOP_WORDS and len(word_lower) >= 2:
                keywords.append(word_lower)

        return list(dict.fromkeys(keywords))

    def _sanitize_fts_query(self, query: str) -> str:
        """Sanitize query for FTS5."""
        query = re.sub(r'["\*\(\)\[\]\{\}]', " ", query)
        query = re.sub(r"\s+", " ", query).strip()
        if query:
            query = f'"{query}"'
        return query

    def _merge_results(
        self,
        vector_results: dict[str, float],
        bm25_results: dict[str, float],
    ) -> dict[str, dict]:
        """Merge vector and BM25 results with weighted fusion."""
        all_ids = set(vector_results.keys()) | set(bm25_results.keys())
        merged = {}

        for chunk_id in all_ids:
            v_score = vector_results.get(chunk_id, 0.0)
            bm_score = bm25_results.get(chunk_id, 0.0)

            has_vec = v_score > 0
            has_bm = bm_score > 0

            if has_vec and has_bm:
                final = (
                    self._config.vector_weight * v_score
                    + self._config.bm25_weight * bm_score
                )
            elif has_vec:
                final = v_score
            else:
                final = bm_score

            merged[chunk_id] = {
                "score": final,
                "vector_score": v_score,
                "bm25_score": bm_score,
            }

        return merged

    def _apply_temporal_decay(
        self, merged: dict[str, dict]
    ) -> dict[str, dict]:
        """Apply temporal decay based on file date."""
        cursor = self._conn.cursor()
        date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})")
        today = datetime.now().date()
        half_life = self._config.temporal_decay_half_life_days
        lambda_decay = math.log(2) / half_life

        for chunk_id, data in merged.items():
            cursor.execute("SELECT path FROM chunks WHERE chunk_id = ?", (chunk_id,))
            row = cursor.fetchone()
            if not row:
                continue

            path = row[0]
            match = date_pattern.search(path)
            if not match:
                continue

            try:
                file_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
                age_days = (today - file_date).days
                if age_days > 0:
                    decay = math.exp(-lambda_decay * age_days)
                    data["score"] *= decay
            except ValueError:
                continue

        return merged

    def _build_results(
        self, sorted_items: list[tuple[str, dict]]
    ) -> list[SearchResult]:
        """Build SearchResult objects from merged data."""
        cursor = self._conn.cursor()
        results = []

        for chunk_id, data in sorted_items:
            cursor.execute(
                """
                SELECT path, start_line, end_line, text
                FROM chunks
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            )
            row = cursor.fetchone()
            if not row:
                continue

            path, start_line, end_line, text = row
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    path=path,
                    start_line=start_line,
                    end_line=end_line,
                    text=text,
                    score=data["score"],
                    vector_score=data["vector_score"],
                    bm25_score=data["bm25_score"],
                )
            )

        return results

    def _mmr_rerank(
        self, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """Maximal Marginal Relevance re-ranking."""
        if not results:
            return results

        lambda_param = self._config.mmr_lambda
        selected: list[SearchResult] = []
        candidates = list(results)

        while len(selected) < top_k and candidates:
            best_idx = 0
            best_mmr = float("-inf")

            for i, candidate in enumerate(candidates):
                relevance = candidate.score

                if selected:
                    max_sim = max(
                        self._jaccard_similarity(candidate.text, s.text)
                        for s in selected
                    )
                else:
                    max_sim = 0.0

                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            selected.append(candidates.pop(best_idx))

        return selected

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0


__all__ = ["SearchResult", "HybridSearcher"]
