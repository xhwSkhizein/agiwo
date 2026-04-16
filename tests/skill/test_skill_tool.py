import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agiwo.skill.loader import SkillLoader
from agiwo.skill.registry import SkillRegistry
from agiwo.skill.search import SkillSearchRecommendation
from agiwo.skill.skill_tool import SkillTool
from tests.utils.agent_context import build_tool_context


class StubSearchService:
    def __init__(self, result: SkillSearchRecommendation) -> None:
        self.result = result

    async def search(self, *, query: str, metadata_items):
        assert query == "help me explore this change"
        assert [item.name for item in metadata_items] == ["brainstorming"]
        return self.result


@pytest.mark.asyncio
async def test_skill_tool_search_returns_structured_recommendation(
    tmp_path: Path,
) -> None:
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
async def test_skill_tool_search_respects_disabled_runtime_setting(
    tmp_path: Path,
) -> None:
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

    fake_settings = SimpleNamespace(skill_search_enabled=False)
    with patch("agiwo.skill.skill_tool.get_settings", return_value=fake_settings):
        result = await tool.execute(
            {"mode": "search", "query": "help me explore this change"},
            build_tool_context(),
        )

    payload = json.loads(result.content)
    assert payload["decision"] == "no_recommendation"
    assert payload["skill_name"] is None


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
    tool = SkillTool(
        registry=registry,
        loader=loader,
        allowed_skills=["brainstorming"],
    )

    result = await tool.execute(
        {"mode": "activate", "skill_name": "brainstorming"},
        build_tool_context(),
    )

    assert result.is_success is True
    assert "Use this skill." in result.content
