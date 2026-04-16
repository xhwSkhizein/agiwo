from pathlib import Path

import pytest

from agiwo.embedding.base import EmbeddingError, EmbeddingModel
from agiwo.llm.base import Model, StreamChunk
from agiwo.skill.registry import SkillMetadata
from agiwo.skill.search import SkillSearchRecommendation, SkillSearchService


def _meta(name: str, description: str) -> SkillMetadata:
    path = Path(f"/tmp/{name}/SKILL.md")
    return SkillMetadata(
        name=name,
        description=description,
        path=path,
        base_dir=path.parent,
    )


class StubEmbedding(EmbeddingModel):
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        super().__init__(id="stub", name="stub", dimensions=3)
        self.mapping = mapping

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.mapping[text] for text in texts]


class FailingEmbedding(EmbeddingModel):
    def __init__(self) -> None:
        super().__init__(id="stub", name="stub", dimensions=3)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("boom")


class StubJudgeModel(Model):
    def __init__(self, payload: str) -> None:
        super().__init__(id="judge", name="judge", provider="openai")
        self.payload = payload

    async def arun_stream(self, messages, tools=None):
        del messages, tools
        yield StreamChunk(content=self.payload)


@pytest.mark.asyncio
async def test_search_returns_specific_skill_from_top_k() -> None:
    service = SkillSearchService(
        embedding_model=StubEmbedding(
            {
                "help me design this change": [1.0, 0.0, 0.0],
                "brainstorming\nExplore design before implementation.": [0.9, 0.0, 0.0],
                "writing-plans\nWrite implementation plans.": [0.5, 0.0, 0.0],
            }
        ),
        judge_model=StubJudgeModel(
            '{"decision":"recommend","skill_name":"brainstorming","reason":"design task"}'
        ),
        top_k=2,
    )

    result = await service.search(
        query="help me design this change",
        metadata_items=[
            _meta("brainstorming", "Explore design before implementation."),
            _meta("writing-plans", "Write implementation plans."),
        ],
    )

    assert result == SkillSearchRecommendation(
        decision="recommend",
        skill_name="brainstorming",
        reason="design task",
    )


@pytest.mark.asyncio
async def test_search_downgrades_to_no_recommendation_when_embedding_fails() -> None:
    service = SkillSearchService(
        embedding_model=FailingEmbedding(),
        judge_model=StubJudgeModel(
            '{"decision":"recommend","skill_name":"brainstorming","reason":"design task"}'
        ),
        top_k=2,
    )

    result = await service.search(
        query="help me design this change",
        metadata_items=[
            _meta("brainstorming", "Explore design before implementation.")
        ],
    )

    assert result.decision == "no_recommendation"
    assert result.skill_name is None
