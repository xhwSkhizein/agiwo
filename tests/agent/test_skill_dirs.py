from pathlib import Path

from agiwo.skill.config import (
    SkillDiscoveryConfig,
    resolve_skill_dirs,
)


def test_resolve_skill_dirs_resolves_against_root(tmp_path: Path) -> None:
    root = tmp_path / "workspace-root"
    root.mkdir()

    config = SkillDiscoveryConfig(
        skills_dirs=["skills", "nested/custom-skills"],
        root_path=str(root),
    )

    resolved = resolve_skill_dirs(config)

    assert resolved == [
        (root / "skills").resolve(),
        (root / "nested" / "custom-skills").resolve(),
    ]


def test_resolve_skill_dirs_skips_blank_entries(tmp_path: Path) -> None:
    root = tmp_path / "workspace-root"
    root.mkdir()

    config = SkillDiscoveryConfig(
        skills_dirs=["skills", "", "  "],
        root_path=str(root),
    )

    resolved = resolve_skill_dirs(config)

    assert resolved == [(root / "skills").resolve()]


def test_resolve_skill_dirs_empty_when_no_dirs() -> None:
    config = SkillDiscoveryConfig(skills_dirs=[], root_path="/tmp")

    resolved = resolve_skill_dirs(config)

    assert resolved == []
