from pathlib import Path

from agiwo.skill.config import (
    SkillDiscoveryConfig,
    normalize_skill_dirs,
    resolve_skill_dirs,
)


def test_normalize_skills_dirs_accepts_string_and_list() -> None:
    assert normalize_skill_dirs("skills") == ["skills"]
    assert normalize_skill_dirs(["skills", "  ~/.agent/skills  "]) == [
        "skills",
        "~/.agent/skills",
    ]
    assert normalize_skill_dirs(["", "   "]) is None


def test_resolve_skill_dirs_resolves_against_config_root(tmp_path: Path) -> None:
    root = tmp_path / "workspace-root"
    root.mkdir()

    config = SkillDiscoveryConfig(
        root_path=str(root),
        configured_dirs=["skills", "nested/custom-skills"],
        env_dirs=[],
    )

    resolved = resolve_skill_dirs(config)

    assert resolved == [
        (root / "skills").resolve(),
        (root / "nested" / "custom-skills").resolve(),
    ]


def test_resolve_skill_dirs_uses_default_root_skills(tmp_path: Path) -> None:
    root = tmp_path / "workspace-root"
    default_skills = root / "skills"
    default_skills.mkdir(parents=True)

    config = SkillDiscoveryConfig(
        root_path=str(root),
        configured_dirs=None,
        env_dirs=[],
    )

    resolved = resolve_skill_dirs(config)

    assert resolved == [default_skills.resolve()]
