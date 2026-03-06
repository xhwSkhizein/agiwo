import pytest
from pydantic import ValidationError

from agiwo.config.settings import load_settings


def test_settings_reads_uppercase_agiwo_env(monkeypatch) -> None:
    monkeypatch.setenv("AGIWO_SKILLS_DIRS", '["skills","~/.agent/skills"]')
    monkeypatch.setenv("AGIWO_IS_SKILLS_ENABLED", "false")

    settings = load_settings(include_env_file=False)

    assert settings.skills_dirs == ["skills", "~/.agent/skills"]
    assert settings.is_skills_enabled is False
    assert "skills_dirs" in settings.model_fields_set
    assert "is_skills_enabled" in settings.model_fields_set


def test_tool_model_settings_support_global_defaults_and_tool_override(monkeypatch) -> None:
    monkeypatch.setenv("AGIWO_TOOL_DEFAULT_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("AGIWO_TOOL_DEFAULT_MODEL_NAME", "gpt-4o-mini")
    monkeypatch.setenv(
        "AGIWO_TOOL_DEFAULT_MODEL_API_KEY_ENV_NAME", "TOOL_DEFAULT_API_KEY"
    )
    monkeypatch.setenv("AGIWO_TOOL_DEFAULT_MODEL_TEMPERATURE", "0.15")
    monkeypatch.setenv("AGIWO_TOOL_WEB_READER_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("AGIWO_TOOL_WEB_READER_MODEL_NAME", "claude-3-5-haiku-latest")
    monkeypatch.setenv(
        "AGIWO_TOOL_WEB_READER_MODEL_API_KEY_ENV_NAME", "WEB_READER_API_KEY"
    )
    monkeypatch.setenv("AGIWO_TOOL_WEB_READER_MODEL_MAX_TOKENS", "1024")

    settings = load_settings(include_env_file=False)

    assert settings.get_tool_model_provider("web_reader") == "anthropic"
    assert settings.get_tool_model_name("web_reader") == "claude-3-5-haiku-latest"
    assert (
        settings.get_tool_model_api_key_env_name("web_reader")
        == "WEB_READER_API_KEY"
    )
    assert settings.get_tool_model_max_tokens("web_reader") == 1024
    assert settings.get_tool_model_temperature("web_reader") == 0.15


def test_tool_model_name_falls_back_to_provider_default(monkeypatch) -> None:
    monkeypatch.setenv("AGIWO_TOOL_DEFAULT_MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("AGIWO_DEEPSEEK_MODEL_NAME", "deepseek-reasoner")

    settings = load_settings(include_env_file=False)

    assert settings.get_tool_model_provider("web_reader") == "deepseek"
    assert settings.get_tool_model_name("web_reader") == "deepseek-reasoner"


def test_tool_model_provider_legacy_values_are_rejected(monkeypatch) -> None:
    monkeypatch.setenv("AGIWO_TOOL_DEFAULT_MODEL_PROVIDER", "generic")

    with pytest.raises(ValidationError):
        load_settings(include_env_file=False)
