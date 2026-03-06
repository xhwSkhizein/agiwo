from pathlib import Path

from agiwo.agent.options import AgentOptions, normalize_skills_dirs


def test_normalize_skills_dirs_accepts_string_and_list() -> None:
    assert normalize_skills_dirs("skills") == ["skills"]
    assert normalize_skills_dirs(["skills", "  ~/.agent/skills  "]) == [
        "skills",
        "~/.agent/skills",
    ]
    assert normalize_skills_dirs(["", "   "]) is None


def test_get_configured_skills_dirs_resolves_against_config_root(tmp_path: Path) -> None:
    root = tmp_path / "workspace-root"
    root.mkdir()

    options = AgentOptions(
        config_root=str(root),
        skills_dirs=["skills", "nested/custom-skills"],
    )

    resolved = options.get_configured_skills_dirs()

    assert resolved == [
        (root / "skills").resolve(),
        (root / "nested" / "custom-skills").resolve(),
    ]


def test_get_configured_skills_dirs_uses_default_root_skills(tmp_path: Path) -> None:
    root = tmp_path / "workspace-root"
    default_skills = root / "skills"
    default_skills.mkdir(parents=True)

    options = AgentOptions(config_root=str(root))

    resolved = options.get_configured_skills_dirs()

    assert resolved == [default_skills.resolve()]
