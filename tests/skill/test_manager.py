from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

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
