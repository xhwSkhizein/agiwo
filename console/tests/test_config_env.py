import pytest
from pydantic import ValidationError

from agiwo.agent import AgentOptions
from agiwo.config.settings import settings
from agiwo.llm.openai import OpenAIModel
from server.app import _build_default_agent_config
from server.config import ConsoleConfig
from server.domain.agent_configs import AgentOptionsInput
from server.schemas import AgentConfigPayload, AgentConfigReplace
from server.services.agent_lifecycle import (
    build_agent_options,
    build_default_agent_options,
    build_model,
)
from server.services.agent_registry import AgentConfigRecord, AgentRegistry
from server.domain.tool_references import (
    InvalidToolReferenceError,
    parse_tool_references,
)
from server.tools import AgentToolRef, BuiltinToolRef, build_tools


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


def test_build_agent_options_uses_global_skills_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "is_skills_enabled", False)

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


def test_default_agent_config_uses_shared_option_defaults() -> None:
    config = ConsoleConfig()

    record = _build_default_agent_config(config)

    assert record.options == build_default_agent_options()
    assert record.options["max_steps"] == AgentOptions().max_steps


def test_agent_options_payload_normalizes_single_skills_dirs() -> None:
    payload = AgentOptionsInput.model_validate({"skills_dirs": "skills"})

    assert payload.skills_dirs == ["skills"]


def test_build_agent_options_normalizes_skills_dirs_and_maps_all_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "is_skills_enabled", True)

    config = AgentConfigRecord(
        name="tester",
        model_provider="openai",
        model_name="gpt-4o-mini",
        options={
            "config_root": "/tmp/agent-root",
            "max_steps": 42,
            "run_timeout": 120,
            "max_input_tokens_per_call": 64000,
            "max_run_cost": 1.25,
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
    assert options.max_input_tokens_per_call == 64000
    assert options.max_run_cost == 1.25
    assert options.enable_termination_summary is False
    assert options.termination_summary_prompt == "Summarize before exit"
    assert options.enable_skill is True
    assert options.skills_dirs == ["skills"]
    assert options.relevant_memory_max_token == 1024
    assert options.stream_cleanup_timeout == 90.5
    assert options.compact_prompt == "Compact the context"


def test_console_tool_catalog_parses_builtin_and_agent_refs() -> None:
    refs = parse_tool_references(["web_search", "agent:child-1"])

    assert refs == [
        BuiltinToolRef(name="web_search"),
        AgentToolRef(agent_id="child-1"),
    ]

    with pytest.raises(InvalidToolReferenceError):
        parse_tool_references(["missing"])

    with pytest.raises(InvalidToolReferenceError):
        parse_tool_references(["agent:"])


@pytest.mark.asyncio
async def test_console_tool_catalog_builds_shared_web_tool_overrides() -> None:
    console_config = ConsoleConfig(
        run_step_storage_type="memory",
        trace_storage_type="memory",
        metadata_storage_type="memory",
    )

    tools = await build_tools(
        ["web_search", "web_reader"],
        console_config=console_config,
        build_agent_tool=pytest.fail,
    )

    assert [tool.name for tool in tools] == ["web_search", "web_reader"]
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
            "max_output_tokens": 123,
            "temperature": 0.25,
        },
    )

    model = build_model(config)

    assert isinstance(model, OpenAIModel)
    assert model.provider == "openai-compatible"
    assert model.base_url == "https://api.minimax.chat/v1"
    assert model.api_key == "test-minimax-key"
    assert model.max_output_tokens == 123
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


def test_agent_config_create_requires_explicit_connection_for_compatible_provider() -> (
    None
):
    with pytest.raises(ValidationError, match="base_url"):
        AgentConfigPayload.model_validate(
            {
                "name": "tester",
                "model_provider": "openai-compatible",
                "model_name": "MiniMax-M2.5",
                "model_params": {"api_key_env_name": "MINIMAX_API_KEY"},
            }
        )


def test_agent_config_create_rejects_plain_api_key_in_model_params() -> None:
    with pytest.raises(ValidationError, match="api_key is not supported"):
        AgentConfigPayload.model_validate(
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


def test_agent_config_replace_requires_full_nested_payloads() -> None:
    with pytest.raises(ValidationError, match="options"):
        AgentConfigReplace.model_validate(
            {
                "name": "tester",
                "description": "",
                "model_provider": "openai-compatible",
                "model_name": "MiniMax-M2.5",
                "system_prompt": "",
                "tools": [],
                "model_params": {
                    "base_url": "https://api.minimax.chat/v1",
                    "api_key_env_name": "MINIMAX_API_KEY",
                },
            }
        )


@pytest.mark.asyncio
async def test_agent_registry_replace_overwrites_nested_config_without_merge() -> None:
    registry = AgentRegistry(
        ConsoleConfig(
            run_step_storage_type="memory",
            trace_storage_type="memory",
            metadata_storage_type="memory",
        )
    )
    await registry.initialize()

    try:
        created = await registry.create_agent(
            AgentConfigRecord(
                name="tester",
                model_provider="openai-compatible",
                model_name="MiniMax-M2.5",
                options={"max_steps": 10, "max_run_cost": 1.5},
                model_params={
                    "base_url": "https://api.minimax.chat/v1",
                    "api_key_env_name": "MINIMAX_API_KEY",
                    "temperature": 0.7,
                },
            )
        )

        updated = await registry.replace_agent(
            created.id,
            AgentConfigRecord(
                name="tester",
                description="replacement",
                model_provider="openai-compatible",
                model_name="MiniMax-M2.5",
                tools=["web_search"],
                options={"max_steps": 5},
                model_params={
                    "base_url": "https://api.other.example/v1",
                    "api_key_env_name": "OTHER_API_KEY",
                    "temperature": 0.2,
                },
            ),
        )

        assert updated is not None
        assert updated.description == "replacement"
        assert updated.tools == ["web_search"]
        assert updated.options["max_steps"] == 5
        assert "max_run_cost" not in updated.options
        assert updated.model_params["base_url"] == "https://api.other.example/v1"
        assert updated.model_params["api_key_env_name"] == "OTHER_API_KEY"
        assert updated.model_params["temperature"] == 0.2
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_agent_registry_replace_rejects_invalid_full_compatible_config() -> None:
    registry = AgentRegistry(
        ConsoleConfig(
            run_step_storage_type="memory",
            trace_storage_type="memory",
            metadata_storage_type="memory",
        )
    )
    await registry.initialize()

    try:
        created = await registry.create_agent(
            AgentConfigRecord(
                name="tester",
                model_provider="openai-compatible",
                model_name="MiniMax-M2.5",
                model_params={
                    "base_url": "https://api.minimax.chat/v1",
                    "api_key_env_name": "MINIMAX_API_KEY",
                },
            )
        )

        with pytest.raises(ValidationError, match="api_key_env_name"):
            await registry.replace_agent(
                created.id,
                AgentConfigRecord(
                    name="tester",
                    model_provider="openai-compatible",
                    model_name="MiniMax-M2.5",
                    model_params={"base_url": "https://api.minimax.chat/v1"},
                ),
            )
    finally:
        await registry.close()
