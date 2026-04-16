"""Runtime skill recommendation based on metadata recall and LLM judgment."""

import json
import math
from dataclasses import dataclass
from typing import Literal

from agiwo.config.settings import get_settings
from agiwo.embedding import EmbeddingFactory
from agiwo.embedding.base import EmbeddingModel
from agiwo.llm import Model, ModelSpec, create_model
from agiwo.skill.registry import SkillMetadata


DEFAULT_NO_RECOMMENDATION_REASON = (
    "No available skill is necessary or clearly matched for this request."
)


@dataclass(frozen=True)
class SkillSearchRecommendation:
    decision: Literal["recommend", "no_recommendation"]
    skill_name: str | None = None
    reason: str = DEFAULT_NO_RECOMMENDATION_REASON


@dataclass(frozen=True)
class SkillSearchCandidate:
    metadata: SkillMetadata
    score: float


class SkillSearchService:
    def __init__(
        self,
        *,
        embedding_model: EmbeddingModel | None = None,
        judge_model: Model | None = None,
        top_k: int | None = None,
    ) -> None:
        settings = get_settings()
        self._embedding_model = embedding_model
        self._judge_model = judge_model
        self._top_k = top_k or settings.skill_search_top_k

    async def search(
        self,
        *,
        query: str,
        metadata_items: list[SkillMetadata],
    ) -> SkillSearchRecommendation:
        if not query.strip() or not metadata_items:
            return SkillSearchRecommendation(decision="no_recommendation")

        try:
            candidates = await self._recall_top_k(
                query=query,
                metadata_items=metadata_items,
            )
        except Exception:  # noqa: BLE001 - search should degrade, not fail hard
            return SkillSearchRecommendation(decision="no_recommendation")

        if not candidates:
            return SkillSearchRecommendation(decision="no_recommendation")

        try:
            return await self._judge_candidates(query=query, candidates=candidates)
        except Exception:  # noqa: BLE001 - judge failures should downgrade safely
            return SkillSearchRecommendation(decision="no_recommendation")

    async def _recall_top_k(
        self,
        *,
        query: str,
        metadata_items: list[SkillMetadata],
    ) -> list[SkillSearchCandidate]:
        embedding_model = self._embedding_model or EmbeddingFactory.create()
        if embedding_model is None:
            raise RuntimeError("embedding unavailable")

        corpus = [self._build_search_text(item) for item in metadata_items]
        vectors = await embedding_model.embed([query, *corpus])
        query_vector = vectors[0]
        scored = [
            SkillSearchCandidate(
                metadata=item,
                score=self._cosine_similarity(query_vector, vector),
            )
            for item, vector in zip(metadata_items, vectors[1:], strict=False)
        ]
        scored.sort(key=lambda candidate: candidate.score, reverse=True)
        return scored[: self._top_k]

    async def _judge_candidates(
        self,
        *,
        query: str,
        candidates: list[SkillSearchCandidate],
    ) -> SkillSearchRecommendation:
        model = self._judge_model or self._build_default_judge_model()
        content_parts: list[str] = []
        messages = [
            {
                "role": "system",
                "content": (
                    "You recommend at most one skill. Return JSON only with "
                    '"decision", optional "skill_name", and "reason". Use '
                    '"no_recommendation" if no skill should be used.'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query,
                        "candidates": [
                            {
                                "name": candidate.metadata.name,
                                "description": candidate.metadata.description,
                                "metadata": candidate.metadata.metadata,
                            }
                            for candidate in candidates
                        ],
                    }
                ),
            },
        ]
        async for chunk in model.arun_stream(messages=messages, tools=None):
            if chunk.content:
                content_parts.append(chunk.content)

        payload = json.loads("".join(content_parts))
        decision = payload.get("decision")
        reason = str(payload.get("reason", DEFAULT_NO_RECOMMENDATION_REASON))
        if decision != "recommend":
            return SkillSearchRecommendation(
                decision="no_recommendation",
                reason=reason,
            )

        skill_name = payload.get("skill_name")
        if not isinstance(skill_name, str):
            return SkillSearchRecommendation(
                decision="no_recommendation", reason=reason
            )
        if any(candidate.metadata.name == skill_name for candidate in candidates):
            return SkillSearchRecommendation(
                decision="recommend",
                skill_name=skill_name,
                reason=reason,
            )
        return SkillSearchRecommendation(decision="no_recommendation", reason=reason)

    def _build_default_judge_model(self) -> Model:
        settings = get_settings()
        return create_model(
            ModelSpec(
                provider=settings.tool_default_model_provider,
                model_name=settings.get_tool_model_name(),
                api_key_env_name=settings.get_tool_model_api_key_env_name(),
                base_url=settings.tool_default_model_base_url,
                temperature=0.0,
                top_p=1.0,
                max_output_tokens=256,
            )
        )

    def _build_search_text(self, metadata: SkillMetadata) -> str:
        parts = [metadata.name, metadata.description]
        parts.extend(
            f"{key}:{value}" for key, value in sorted(metadata.metadata.items())
        )
        return "\n".join(parts)

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)


__all__ = [
    "DEFAULT_NO_RECOMMENDATION_REASON",
    "SkillSearchCandidate",
    "SkillSearchRecommendation",
    "SkillSearchService",
]
