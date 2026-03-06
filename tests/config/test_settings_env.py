from agiwo.config.settings import load_settings


def test_settings_reads_uppercase_agiwo_env(monkeypatch) -> None:
    monkeypatch.setenv("AGIWO_SKILLS_DIRS", '["skills","~/.agent/skills"]')
    monkeypatch.setenv("AGIWO_IS_SKILLS_ENABLED", "false")

    settings = load_settings(include_env_file=False)

    assert settings.skills_dirs == ["skills", "~/.agent/skills"]
    assert settings.is_skills_enabled is False
    assert "skills_dirs" in settings.model_fields_set
    assert "is_skills_enabled" in settings.model_fields_set
