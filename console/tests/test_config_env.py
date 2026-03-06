import pytest
from pydantic import ValidationError

from agiwo.llm.openai import OpenAIModel
from agiwo.tool.storage.citation import CitationStoreConfig
from server.config import ConsoleConfig
from server.schemas import AgentConfigCreate, AgentOptionsPayload
from server.services.agent_builder import build_agent_options, build_model
from server.services.agent_registry import AgentConfigRecord
from server.services import agent_builder
from server.tools import create_tools


def test_console_config_reads_uppercase_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGIWO_CONSOLE_FEISHU_ENABLED", "true")

    config = ConsoleConfig()

    assert config.feishu_enabled is True


def test_console_config_rejects_legacy_default_model_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGIWO_CONSOLE_DEFAULT_AGENT_MODEL_PROVIDER", "generic")

    with pytest.raises(ValidationError):
        ConsoleConfig()


def test_console_config_rejects_plain_api_key_in_default_model_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AGIWO_CONSOLE_DEFAULT_AGENT_MODEL_PARAMS",
        '{"api_key":"sk-plain-text","base_url":"https://api.example.com/v1"}',
    )

    with pytest.raises(ValidationError, match="api_key"):
        ConsoleConfig()


def test_build_agent_options_uses_global_skills_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_builder.settings, "is_skills_enabled", False)

    config = AgentConfigRecord(
        name="tester",
        model_provider="openai",
        model_name="gpt-4o-mini",
        options={},
    )
    console_config = ConsoleConfig(
        run_step_storage_type="memory",
        trace_storage_type="memory",
        metadata_storage_type="memory",
    )

    options = build_agent_options(config, console_config)

    assert options.enable_skill is False
    assert options.skills_dirs is None


def test_agent_options_payload_normalizes_single_and_legacy_skills_dir() -> None:
    payload = AgentOptionsPayload.model_validate({"skills_dirs": "skills"})
    legacy_payload = AgentOptionsPayload.model_validate(
        {"skills_dir": ["skills", "~/.agent/skills"]}
    )

    assert payload.skills_dirs == ["skills"]
    assert legacy_payload.skills_dirs == ["skills", "~/.agent/skills"]


def test_build_agent_options_normalizes_skills_dirs_and_maps_all_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_builder.settings, "is_skills_enabled", True)

    config = AgentConfigRecord(
        name="tester",
        model_provider="openai",
        model_name="gpt-4o-mini",
        options={
            "config_root": "/tmp/agent-root",
            "max_steps": 42,
            "run_timeout": 120,
            "max_context_window_tokens": 64000,
            "max_tokens_per_run": 256000,
            "max_run_token_cost": 1.25,
            "enable_termination_summary": False,
            "termination_summary_prompt": "Summarize before exit",
            "enable_skill": True,
            "skills_dirs": "skills",
            "relevant_memory_max_token": 1024,
            "stream_cleanup_timeout": 90.5,
            "compact_prompt": "Compact the context",
        },
    )
    console_config = ConsoleConfig(
        run_step_storage_type="memory",
        trace_storage_type="memory",
        metadata_storage_type="memory",
    )

    options = build_agent_options(config, console_config)

    assert options.config_root == "/tmp/agent-root"
    assert options.max_steps == 42
    assert options.run_timeout == 120
    assert options.max_context_window_tokens == 64000
    assert options.max_tokens_per_run == 256000
    assert options.max_run_token_cost == 1.25
    assert options.enable_termination_summary is False
    assert options.termination_summary_prompt == "Summarize before exit"
    assert options.enable_skill is True
    assert options.skills_dirs == ["skills"]
    assert options.relevant_memory_max_token == 1024
    assert options.stream_cleanup_timeout == 90.5
    assert options.compact_prompt == "Compact the context"


def test_create_tools_uses_config_overrides_for_web_tools() -> None:
    citation_config = CitationStoreConfig(
        storage_type="memory",
        collection_name="console-test-citations",
    )

    tools = create_tools(
        ["web_search", "web_reader"],
        tool_config_overrides={
            "web_search": {"citation_store_config": citation_config},
            "web_reader": {"citation_store_config": citation_config},
        },
    )

    assert [tool.get_name() for tool in tools] == ["web_search", "web_reader"]
    assert tools[0]._citation_source_store is tools[1]._citation_source_store


def test_build_model_uses_shared_model_factory_for_compatible_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")

    config = AgentConfigRecord(
        name="tester",
        model_provider="openai-compatible",
        model_name="MiniMax-M2.5",
        model_params={
            "base_url": "https://api.minimax.chat/v1",
            "api_key_env_name": "MINIMAX_API_KEY",
            "max_output_tokens_per_call": 123,
            "temperature": 0.25,
        },
    )

    model = build_model(config)

    assert isinstance(model, OpenAIModel)
    assert model.provider == "openai-compatible"
    assert model.base_url == "https://api.minimax.chat/v1"
    assert model.api_key == "test-minimax-key"
    assert model.max_tokens == 123
    assert model.temperature == 0.25


def test_build_model_does_not_fallback_to_openai_credentials_for_compatible_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "shared-openai-key")

    config = AgentConfigRecord(
        name="tester",
        model_provider="openai-compatible",
        model_name="MiniMax-M2.5",
        model_params={"base_url": "https://api.minimax.chat/v1"},
    )

    with pytest.raises(ValueError, match="api_key"):
        build_model(config)


def test_agent_config_create_requires_explicit_connection_for_compatible_provider() -> None:
    with pytest.raises(ValidationError, match="base_url"):
        AgentConfigCreate.model_validate(
            {
                "name": "tester",
                "model_provider": "openai-compatible",
                "model_name": "MiniMax-M2.5",
                "model_params": {"api_key_env_name": "MINIMAX_API_KEY"},
            }
        )


def test_agent_config_create_rejects_plain_api_key_in_model_params() -> None:
    with pytest.raises(ValidationError, match="api_key is not supported"):
        AgentConfigCreate.model_validate(
            {
                "name": "tester",
                "model_provider": "openai-compatible",
                "model_name": "MiniMax-M2.5",
                "model_params": {
                    "base_url": "https://api.minimax.chat/v1",
                    "api_key_env_name": "MINIMAX_API_KEY",
                    "api_key": "sk-plain-text",
                },
            }
        )


def test_agent_config_record_sanitizes_model_params_and_strips_plain_api_key() -> None:
    record = AgentConfigRecord.model_validate(
        {
            "name": "tester",
            "model_provider": "openai-compatible",
            "model_name": "MiniMax-M2.5",
            "model_params": {
                "base_url": " https://api.minimax.chat/v1 ",
                "api_key_env_name": " MINIMAX_API_KEY ",
                "api_key": "sk-plain-text",
            },
        }
    )

    assert record.model_provider == "openai-compatible"
    assert record.model_params.get("base_url") == "https://api.minimax.chat/v1"
    assert record.model_params.get("api_key_env_name") == "MINIMAX_API_KEY"
    assert "api_key" not in record.model_params
