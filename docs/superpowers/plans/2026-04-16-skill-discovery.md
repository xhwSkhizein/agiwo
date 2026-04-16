# Skill Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement prompt-safe skill discovery with a small configured prompt skill set and a dual-mode `skill` tool that can search before activation.

**Architecture:** Keep prompt-time skill visibility small by rendering only configured default skills through `SkillManager`, then move long-tail skill discovery into a new in-memory `agiwo.skill.search` service that uses embedding TopK recall plus LLM judgment. Reuse the existing tool-model and embedding infrastructure, keep allowlist enforcement in `SkillManager`, and preserve explicit `skill.activate` behavior for full `SKILL.md` loading.

**Tech Stack:** Python 3.11, `pydantic-settings`, `agiwo.llm` model factory, `agiwo.embedding`, pytest

---

## File Structure

- Modify: `agiwo/config/settings.py`
  Add `default_prompt_skills`, `skill_search_enabled`, and `skill_search_top_k`.
- Modify: `tests/config/test_settings_env.py`
  Cover the new settings fields and defaults.
- Modify: `agiwo/skill/prompt_catalog.py`
  Render the smaller prompt-time skill section with `skill.search` guidance and without `<location>`.
- Modify: `agiwo/skill/manager.py`
  Validate configured prompt skills, filter them by `allowed_skills`, and expose `search_skills(query, allowed_skills)`.
- Create: `agiwo/skill/search.py`
  Hold the in-memory search DTOs and the embedding + judge search service.
- Modify: `agiwo/skill/skill_tool.py`
  Support `mode=search` and `mode=activate`.
- Modify: `agiwo/skill/__init__.py`
  Re-export search DTOs if they are used by tests or downstream callers.
- Create: `tests/skill/test_manager.py`
  Cover default prompt skill filtering and manager-level search allowlist behavior.
- Create: `tests/skill/test_search.py`
  Cover TopK recall, invalid judge output, and failure downgrade.
- Create: `tests/skill/test_skill_tool.py`
  Cover `skill.search`, `skill.activate`, and validation branches.
- Create: `tests/agent/test_prompt.py`
  Cover reduced system prompt rendering and disabled-skill behavior.
- Modify: `docs/guides/skills.md`
  Update the guide to explain prompt-time defaults plus runtime `search` + `activate`.

### Task 1: Settings And Reduced Prompt Skill Rendering

**Files:**
- Modify: `agiwo/config/settings.py`
- Modify: `agiwo/skill/prompt_catalog.py`
- Modify: `agiwo/skill/manager.py`
- Modify: `tests/config/test_settings_env.py`
- Test: `tests/skill/test_manager.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/config/test_settings_env.py
def test_skill_search_settings_are_loaded(monkeypatch) -> None:
    monkeypatch.setenv('AGIWO_DEFAULT_PROMPT_SKILLS', '["brainstorming","writing-plans"]')
    monkeypatch.setenv("AGIWO_SKILL_SEARCH_ENABLED", "false")
    monkeypatch.setenv("AGIWO_SKILL_SEARCH_TOP_K", "4")

    settings = load_settings(include_env_file=False)

    assert settings.default_prompt_skills == ["brainstorming", "writing-plans"]
    assert settings.skill_search_enabled is False
    assert settings.skill_search_top_k == 4


# tests/skill/test_manager.py
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agiwo.skill.config import SkillDiscoveryConfig
from agiwo.skill.manager import SkillManager
from agiwo.skill.registry import SkillMetadata


def _meta(name: str, description: str) -> SkillMetadata:
    path = Path(f"/tmp/{name}/SKILL.md")
    return SkillMetadata(
        name=name,
        description=description,
        path=path,
        base_dir=path.parent,
    )


def test_render_skills_section_uses_only_default_prompt_skills() -> None:
    manager = SkillManager(SkillDiscoveryConfig(skills_dirs=[], root_path="/tmp"))
    manager._metadata_cache = [
        _meta("brainstorming", "Explore design before implementation."),
        _meta("writing-plans", "Write implementation plans."),
        _meta("imagegen", "Generate raster images."),
    ]
    manager._initialized = True

    fake_settings = SimpleNamespace(
        default_prompt_skills=["brainstorming", "writing-plans"],
        skill_search_enabled=True,
        skill_search_top_k=6,
        root_path="/tmp",
        skills_dirs=[],
    )

    with patch("agiwo.skill.manager.get_settings", return_value=fake_settings):
        rendered = manager.render_skills_section(allowed_skills=["brainstorming"])

    assert "brainstorming" in rendered
    assert "writing-plans" not in rendered
    assert "imagegen" not in rendered
    assert "skill.search" in rendered
    assert "<location>" not in rendered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/config/test_settings_env.py tests/skill/test_manager.py -q`

Expected: FAIL with missing `default_prompt_skills` / `skill_search_*` fields and current prompt rendering still including all cached skills plus `<location>`.

- [ ] **Step 3: Write minimal implementation**

```python
# agiwo/config/settings.py
class AgiwoSettings(BaseSettings):
    # === Skills ===
    skills_dirs: list[str] = Field(
        default_factory=lambda: ["examples/skills", "skills"],
        description="Skill directories to scan (relative to root_path if not absolute)",
    )
    default_prompt_skills: list[str] = Field(
        default_factory=list,
        description="Explicit skill names rendered into the agent system prompt",
    )
    skill_search_enabled: bool = Field(
        default=True,
        description="Whether runtime skill recommendation is enabled",
    )
    skill_search_top_k: int = Field(
        default=6,
        ge=1,
        description="How many skill metadata candidates to send to the search judge",
    )
```

```python
# agiwo/skill/manager.py
from agiwo.config.settings import get_settings


class SkillManager:
    def _resolve_default_prompt_skill_names(self) -> list[str]:
        runtime_settings = get_settings()
        return self.validate_explicit_allowed_skills(
            runtime_settings.default_prompt_skills,
            available_skill_names=self.list_available_skill_names(),
        ) or []

    def _select_prompt_metadata(
        self,
        allowed_skills: list[str] | None = None,
    ) -> list[SkillMetadata]:
        default_names = set(self._resolve_default_prompt_skill_names())
        items = [item for item in self._metadata_cache if item.name in default_names]
        if allowed_skills is None:
            return items
        allowed = set(allowed_skills)
        return [item for item in items if item.name in allowed]

    def render_skills_section(
        self,
        allowed_skills: list[str] | None = None,
    ) -> str:
        return self._prompt_catalog.render_section(
            self._select_prompt_metadata(allowed_skills)
        )
```

```python
# agiwo/skill/prompt_catalog.py
class SkillPromptCatalog:
    def render_section(self, metadata_items: list[SkillMetadata]) -> str:
        if not metadata_items:
            return ""

        lines = ["## Skills", ""]
        lines.append("Skills are optional. Do not use one unless it is clearly helpful.")
        lines.append("If you are unsure, call `skill.search` with the user's original request before activating any skill.")
        lines.append("")
        lines.append("<available_skills>")
        for metadata in metadata_items:
            lines.append("  <skill>")
            lines.append(f"    <name>{metadata.name}</name>")
            lines.append(f"    <description>{metadata.description}</description>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/config/test_settings_env.py tests/skill/test_manager.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agiwo/config/settings.py agiwo/skill/prompt_catalog.py agiwo/skill/manager.py tests/config/test_settings_env.py tests/skill/test_manager.py
git commit -m "feat: reduce prompt-time skill rendering"
```

### Task 2: Add In-Memory Skill Search Service

**Files:**
- Create: `agiwo/skill/search.py`
- Modify: `agiwo/skill/manager.py`
- Modify: `agiwo/skill/__init__.py`
- Test: `tests/skill/test_search.py`
- Test: `tests/skill/test_manager.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/skill/test_search.py
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
        metadata_items=[_meta("brainstorming", "Explore design before implementation.")],
    )

    assert result.decision == "no_recommendation"
    assert result.skill_name is None
```

```python
# tests/skill/test_manager.py
import pytest


@pytest.mark.asyncio
async def test_search_skills_applies_allowed_skill_filter() -> None:
    manager = SkillManager(SkillDiscoveryConfig(skills_dirs=[], root_path="/tmp"))
    manager._metadata_cache = [
        _meta("brainstorming", "Explore design before implementation."),
        _meta("writing-plans", "Write implementation plans."),
    ]
    manager._initialized = True

    class StubSearchService:
        async def search(self, *, query: str, metadata_items: list[SkillMetadata]):
            assert query == "help me plan"
            assert [item.name for item in metadata_items] == ["brainstorming"]
            return SimpleNamespace(
                decision="recommend",
                skill_name="brainstorming",
                reason="design task",
            )

    manager._search_service = StubSearchService()

    result = await manager.search_skills(
        query="help me plan",
        allowed_skills=["brainstorming"],
    )

    assert result.skill_name == "brainstorming"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/skill/test_search.py tests/skill/test_manager.py -q`

Expected: FAIL because `agiwo.skill.search` and `SkillManager.search_skills(query, allowed_skills)` do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# agiwo/skill/search.py
import json
import math
from dataclasses import dataclass
from typing import Literal

from agiwo.config.settings import get_settings
from agiwo.embedding import EmbeddingFactory
from agiwo.embedding.base import EmbeddingModel
from agiwo.llm import Model, ModelSpec, create_model
from agiwo.skill.registry import SkillMetadata


@dataclass(frozen=True)
class SkillSearchRecommendation:
    decision: Literal["recommend", "no_recommendation"]
    skill_name: str | None = None
    reason: str = ""


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
            return SkillSearchRecommendation(
                decision="no_recommendation",
                reason="No available skill is necessary or clearly matched for this request.",
            )

        try:
            candidates = await self._recall_top_k(query=query, metadata_items=metadata_items)
        except Exception:
            return SkillSearchRecommendation(
                decision="no_recommendation",
                reason="No available skill is necessary or clearly matched for this request.",
            )

        if not candidates:
            return SkillSearchRecommendation(
                decision="no_recommendation",
                reason="No available skill is necessary or clearly matched for this request.",
            )

        try:
            return await self._judge_candidates(query=query, candidates=candidates)
        except Exception:
            return SkillSearchRecommendation(
                decision="no_recommendation",
                reason="No available skill is necessary or clearly matched for this request.",
            )

    async def _recall_top_k(
        self,
        *,
        query: str,
        metadata_items: list[SkillMetadata],
    ) -> list[SkillSearchCandidate]:
        embedding_model = self._embedding_model or EmbeddingFactory.create()
        if embedding_model is None:
            raise RuntimeError("embedding unavailable")

        texts = [self._build_search_text(item) for item in metadata_items]
        vectors = await embedding_model.embed([query, *texts])
        query_vector = vectors[0]
        scored = [
            SkillSearchCandidate(metadata=item, score=self._cosine_similarity(query_vector, vector))
            for item, vector in zip(metadata_items, vectors[1:])
        ]
        scored.sort(key=lambda candidate: candidate.score, reverse=True)
        return scored[: self._top_k]

    async def _judge_candidates(
        self,
        *,
        query: str,
        candidates: list[SkillSearchCandidate],
    ) -> SkillSearchRecommendation:
        model = self._judge_model or create_model(
            ModelSpec(
                provider=get_settings().tool_default_model_provider,
                model_name=get_settings().get_tool_model_name(),
                api_key_env_name=get_settings().get_tool_model_api_key_env_name(),
                base_url=get_settings().tool_default_model_base_url,
                temperature=0.0,
                top_p=1.0,
                max_output_tokens=256,
            )
        )
        prompt = json.dumps(
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
                "instructions": "Return JSON with decision=recommend or no_recommendation.",
            }
        )
        content_parts: list[str] = []
        async for chunk in model.arun_stream(
            messages=[
                {"role": "system", "content": "You recommend one skill or no skill. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            tools=None,
        ):
            if chunk.content:
                content_parts.append(chunk.content)

        payload = json.loads("".join(content_parts))
        decision = payload.get("decision")
        skill_name = payload.get("skill_name")
        if decision == "recommend" and any(
            candidate.metadata.name == skill_name for candidate in candidates
        ):
            return SkillSearchRecommendation(
                decision="recommend",
                skill_name=skill_name,
                reason=str(payload.get("reason", "")),
            )
        return SkillSearchRecommendation(
            decision="no_recommendation",
            reason=str(payload.get("reason", "No available skill is necessary or clearly matched for this request.")),
        )

    def _build_search_text(self, metadata: SkillMetadata) -> str:
        parts = [metadata.name, metadata.description]
        parts.extend(f"{key}:{value}" for key, value in sorted(metadata.metadata.items()))
        return "\n".join(parts)

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)
```

```python
# agiwo/skill/manager.py
from agiwo.skill.search import SkillSearchRecommendation, SkillSearchService


class SkillManager:
    def __init__(self, config: SkillDiscoveryConfig) -> None:
        self._config = config
        self.registry = SkillRegistry()
        self.loader = SkillLoader(self.registry)
        self._prompt_catalog = SkillPromptCatalog()
        self._search_service: SkillSearchService | None = None
        self._metadata_cache: list[SkillMetadata] = []
        self._change_token = ""
        self._initialized = False

    def _get_search_service(self) -> SkillSearchService:
        if self._search_service is None:
            self._search_service = SkillSearchService()
        return self._search_service

    async def search_skills(
        self,
        *,
        query: str,
        allowed_skills: list[str] | None,
    ) -> SkillSearchRecommendation:
        self.initialize_sync()
        settings = get_settings()
        if not settings.skill_search_enabled:
            return SkillSearchRecommendation(
                decision="no_recommendation",
                reason="No available skill is necessary or clearly matched for this request.",
            )

        metadata_items = self._metadata_cache
        if allowed_skills is not None:
            allowed = set(allowed_skills)
            metadata_items = [item for item in metadata_items if item.name in allowed]

        return await self._get_search_service().search(
            query=query,
            metadata_items=metadata_items,
        )
```

```python
# agiwo/skill/__init__.py
from agiwo.skill.search import (
    SkillSearchCandidate,
    SkillSearchRecommendation,
    SkillSearchService,
)

__all__ += [
    "SkillSearchCandidate",
    "SkillSearchRecommendation",
    "SkillSearchService",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/skill/test_search.py tests/skill/test_manager.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agiwo/skill/search.py agiwo/skill/manager.py agiwo/skill/__init__.py tests/skill/test_search.py tests/skill/test_manager.py
git commit -m "feat: add in-memory skill search service"
```

### Task 3: Extend `skill` Tool To Search Before Activation

**Files:**
- Modify: `agiwo/skill/skill_tool.py`
- Modify: `agiwo/skill/manager.py`
- Test: `tests/skill/test_skill_tool.py`
- Test: `tests/tool/test_tool_manager.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/skill/test_skill_tool.py
import json
from pathlib import Path

import pytest

from agiwo.skill.loader import SkillLoader
from agiwo.skill.registry import SkillMetadata, SkillRegistry
from agiwo.skill.search import SkillSearchRecommendation
from agiwo.skill.skill_tool import SkillTool
from tests.utils.agent_context import build_tool_context


class StubSearchService:
    def __init__(self, result: SkillSearchRecommendation) -> None:
        self.result = result

    async def search(self, *, query: str, metadata_items):
        assert query == "help me explore this change"
        return self.result


@pytest.mark.asyncio
async def test_skill_tool_search_returns_structured_recommendation(tmp_path: Path) -> None:
    registry = SkillRegistry()
    loader = SkillLoader(registry)
    tool = SkillTool(
        registry=registry,
        loader=loader,
        allowed_skills=["brainstorming"],
        search_service=StubSearchService(
            SkillSearchRecommendation(
                decision="recommend",
                skill_name="brainstorming",
                reason="design task",
            )
        ),
    )

    result = await tool.execute(
        {"mode": "search", "query": "help me explore this change"},
        build_tool_context(),
    )

    payload = json.loads(result.content)
    assert result.is_success is True
    assert payload["decision"] == "recommend"
    assert payload["skill_name"] == "brainstorming"


@pytest.mark.asyncio
async def test_skill_tool_activate_keeps_existing_behavior(tmp_path: Path) -> None:
    skill_dir = tmp_path / "brainstorming"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\nname: brainstorming\ndescription: Explore design first.\n---\n\nUse this skill.",
        encoding="utf-8",
    )

    registry = SkillRegistry()
    registry.discover_skills_sync([tmp_path])
    loader = SkillLoader(registry)
    tool = SkillTool(registry=registry, loader=loader, allowed_skills=["brainstorming"])

    result = await tool.execute(
        {"mode": "activate", "skill_name": "brainstorming"},
        build_tool_context(),
    )

    assert result.is_success is True
    assert "Use this skill." in result.content
```

```python
# tests/tool/test_tool_manager.py
@patch("agiwo.tool.manager.get_global_skill_manager")
def test_build_skill_tool_still_passes_allowlist_to_skill_tool(
    mock_get_sm,
    tool_manager,
):
    mock_skill_manager = Mock()
    mock_skill_tool = Mock()
    mock_skill_manager.create_skill_tool.return_value = mock_skill_tool
    mock_get_sm.return_value = mock_skill_manager

    result = tool_manager._build_skill_tool(["brainstorming"])

    assert result is mock_skill_tool
    mock_skill_manager.create_skill_tool.assert_called_once_with(["brainstorming"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/skill/test_skill_tool.py tests/tool/test_tool_manager.py -q`

Expected: FAIL because the tool schema still requires only `skill_name` and `mode=search` is unsupported.

- [ ] **Step 3: Write minimal implementation**

```python
# agiwo/skill/manager.py
def create_skill_tool(
    self,
    allowed_skills: list[str] | None = None,
) -> SkillTool:
    return SkillTool(
        registry=self.registry,
        loader=self.loader,
        allowed_skills=allowed_skills,
        search_service=self._get_search_service(),
    )
```

```python
# agiwo/skill/skill_tool.py
import json
import time
from typing import Any

from agiwo.skill.search import SkillSearchRecommendation, SkillSearchService


class SkillTool(BaseTool):
    def __init__(
        self,
        registry: SkillRegistry,
        loader: SkillLoader,
        allowed_skills: list[str] | None = None,
        search_service: SkillSearchService | None = None,
    ) -> None:
        super().__init__()
        self.registry = registry
        self.loader = loader
        self._search_service = search_service
        self._allowed_skills = (
            frozenset(allowed_skills) if allowed_skills is not None else None
        )

    description = (
        "Search for a matching skill or activate a specific skill. "
        "Use mode=search with the user's original request when you are unsure whether a skill is needed. "
        "Use mode=activate only after you have chosen a specific skill."
    )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["search", "activate"],
                    "description": "Search for a recommended skill or activate one by name",
                },
                "query": {
                    "type": "string",
                    "description": "Original user request when mode=search",
                },
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to activate when mode=activate",
                },
            },
            "required": ["mode"],
        }

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        del abort_signal
        start_time = time.time()
        mode = parameters.get("mode")
        if mode == "search":
            return await self._execute_search(parameters, context, start_time)
        if mode == "activate":
            return await self._execute_activate(parameters, context, start_time)
        return ToolResult.failed(
            tool_name=self.name,
            error=f"Invalid mode: {mode}",
            tool_call_id=context.tool_call_id,
            input_args=parameters,
            start_time=start_time,
        )

    async def _execute_search(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        start_time: float,
    ) -> ToolResult:
        query = parameters.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult.failed(
                tool_name=self.name,
                error="Missing required parameter: query",
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                start_time=start_time,
            )

        recommendation = (
            await self._search_service.search(query=query, metadata_items=self._searchable_metadata())
            if self._search_service is not None
            else SkillSearchRecommendation(
                decision="no_recommendation",
                reason="No available skill is necessary or clearly matched for this request.",
            )
        )
        payload = {
            "decision": recommendation.decision,
            "skill_name": recommendation.skill_name,
            "reason": recommendation.reason,
        }
        return ToolResult.success(
            tool_name=self.name,
            tool_call_id=context.tool_call_id,
            input_args=parameters,
            content=json.dumps(payload),
            output=payload,
            content_for_user="Skill search completed.",
            start_time=start_time,
        )

    async def _execute_activate(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        start_time: float,
    ) -> ToolResult:
        skill_name = parameters.get("skill_name")
        if not isinstance(skill_name, str) or not skill_name.strip():
            return ToolResult.failed(
                tool_name=self.name,
                error="Missing required parameter: skill_name",
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                start_time=start_time,
            )
        if self._allowed_skills is not None and skill_name not in self._allowed_skills:
            return ToolResult.failed(
                tool_name=self.name,
                error=f"Skill '{skill_name}' is not allowed by the configured allowlist.",
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                start_time=start_time,
            )

        metadata = self.registry.get_metadata(skill_name)
        if metadata is None:
            return ToolResult.failed(
                tool_name=self.name,
                error=f"Skill '{skill_name}' not found. Available skills: {', '.join(self.registry.list_available())}",
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                start_time=start_time,
            )

        skill_content = await self.loader.load_skill(skill_name)
        return ToolResult.success(
            tool_name=self.name,
            tool_call_id=context.tool_call_id,
            input_args=parameters,
            content=skill_content.body,
            content_for_user=f'The skill "{skill_name}" has been activated.',
            output={"skill_name": skill_name, "metadata": metadata.model_dump()},
            start_time=start_time,
        )

    def _searchable_metadata(self) -> list[SkillMetadata]:
        metadata_items = list(self.registry._metadata_cache.values())
        if self._allowed_skills is None:
            return metadata_items
        return [item for item in metadata_items if item.name in self._allowed_skills]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/skill/test_skill_tool.py tests/tool/test_tool_manager.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agiwo/skill/skill_tool.py agiwo/skill/manager.py tests/skill/test_skill_tool.py tests/tool/test_tool_manager.py
git commit -m "feat: add search mode to skill tool"
```

### Task 4: Prompt Integration Regressions And User-Facing Docs

**Files:**
- Create: `tests/agent/test_prompt.py`
- Modify: `docs/guides/skills.md`
- Modify: `agiwo/agent/prompt.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/agent/test_prompt.py
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agiwo.agent.prompt import build_system_prompt
from agiwo.workspace.documents import WorkspaceDocumentStore
from agiwo.workspace.layout import build_agent_workspace


class NoopBootstrapper:
    async def ensure_prompt_ready(self, workspace):
        workspace.workspace.mkdir(parents=True, exist_ok=True)
        workspace.work_dir.mkdir(parents=True, exist_ok=True)
        workspace.memory_dir.mkdir(parents=True, exist_ok=True)


class StubSkillManager:
    async def initialize(self) -> None:
        return None

    async def refresh_if_changed(self) -> None:
        return None

    def render_skills_section(self, allowed_skills=None) -> str:
        assert allowed_skills == ["brainstorming"]
        return "## Skills\n\n<available_skills>\n  <skill>\n    <name>brainstorming</name>\n    <description>Explore design before implementation.</description>\n  </skill>\n</available_skills>"


@pytest.mark.asyncio
async def test_build_system_prompt_uses_reduced_skill_section(tmp_path: Path) -> None:
    workspace = build_agent_workspace(root_path=tmp_path, agent_name="planner")
    fake_settings = SimpleNamespace(
        default_prompt_skills=["brainstorming"],
        skill_search_enabled=True,
        skill_search_top_k=6,
        root_path=str(tmp_path),
        skills_dirs=[],
    )

    with patch("agiwo.agent.prompt.get_global_skill_manager", return_value=StubSkillManager()):
        with patch("agiwo.skill.manager.get_settings", return_value=fake_settings):
            prompt = await build_system_prompt(
                base_prompt="Base system prompt",
                workspace=workspace,
                tools=[],
                allowed_skills=["brainstorming"],
                bootstrapper=NoopBootstrapper(),
                document_store=WorkspaceDocumentStore(),
            )

    assert "brainstorming" in prompt
    assert "<location>" not in prompt
    assert "skill.search" in prompt


@pytest.mark.asyncio
async def test_build_system_prompt_skips_skill_section_when_disabled(tmp_path: Path) -> None:
    workspace = build_agent_workspace(root_path=tmp_path, agent_name="planner")
    prompt = await build_system_prompt(
        base_prompt="Base system prompt",
        workspace=workspace,
        tools=[],
        allowed_skills=[],
        bootstrapper=NoopBootstrapper(),
        document_store=WorkspaceDocumentStore(),
    )

    assert "## Skills" not in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agent/test_prompt.py -q`

Expected: FAIL because there is no dedicated prompt regression file yet and the current guide still documents skill loading as a prompt-time catalog.

- [ ] **Step 3: Write minimal implementation**

```python
# agiwo/agent/prompt.py
async def build_system_prompt(
    *,
    base_prompt: str,
    workspace: AgentWorkspace,
    tools: list[BaseTool] | None = None,
    allowed_skills: list[str] | None = None,
    bootstrapper: WorkspaceBootstrapper,
    document_store: WorkspaceDocumentStore,
) -> str:
    await bootstrapper.ensure_prompt_ready(workspace)
    skill_manager = (
        get_global_skill_manager() if skills_enabled(allowed_skills) else None
    )
    if skill_manager is not None:
        await skill_manager.initialize()
        await skill_manager.refresh_if_changed()

    current_dt = datetime.now().astimezone()
    skills_section = (
        skill_manager.render_skills_section(allowed_skills)
        if skill_manager is not None
        else ""
    )
    documents = document_store.read(workspace)
    sections = [
        _render_identity(documents),
        _render_soul(workspace, documents),
        base_prompt.strip() if base_prompt else "",
        _render_environment(
            workspace,
            os_info=_get_os_info(),
            language_info=_get_language_info(),
            timezone=str(current_dt.tzinfo),
            current_date=current_dt.strftime("%Y-%m-%d"),
        ),
        _render_tools(
            tuple((tool.name, tool.get_short_description()) for tool in (tools or []))
        ),
        f"---\n\n{skills_section}".strip() if skills_section else "",
        _render_user(documents),
    ]
    return "\n\n".join(filter(None, sections))
```

```markdown
<!-- docs/guides/skills.md -->
## Skill Discovery

Skills are still discovered from configured directories at startup, but the agent system prompt no longer receives the full catalog.

Instead:

1. `default_prompt_skills` controls the small subset rendered into the prompt
2. the `skill` tool can run `mode="search"` with the user's original request
3. the `skill` tool only loads full `SKILL.md` content during `mode="activate"`

This keeps prompt size stable even when `skills_dirs` contains many skills.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agent/test_prompt.py tests/skill/test_manager.py tests/skill/test_search.py tests/skill/test_skill_tool.py tests/config/test_settings_env.py tests/tool/test_tool_manager.py -q`

Expected: PASS

Run: `uv run python scripts/lint.py changed`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/prompt.py docs/guides/skills.md tests/agent/test_prompt.py
git commit -m "docs: document dynamic skill discovery"
```

## Self-Review Checklist

- Spec coverage:
  Prompt shrinkage is covered by Task 1 and Task 4.
  Dual-mode `skill` tool is covered by Task 3.
  Embedding TopK plus LLM judgment is covered by Task 2.
  `allowed_skills` enforcement is covered by Task 1 and Task 2 manager tests.
  No migration / in-memory-only scope is enforced by Task 2 architecture and file list.
- Placeholder scan:
  No `TODO`, `TBD`, or "similar to previous task" shortcuts remain in the plan.
- Type consistency:
  The plan uses the same names throughout: `default_prompt_skills`, `skill_search_enabled`, `skill_search_top_k`, `SkillSearchRecommendation`, `SkillSearchService`, `mode=search`, and `mode=activate`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-16-skill-discovery.md`. Two execution options:

1. Subagent-Driven (recommended) - I dispatch a fresh subagent per task, review between tasks, fast iteration

2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
