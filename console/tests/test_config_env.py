import pytest
from pydantic import ValidationError

import agiwo.agent.agent as agent_module
import agiwo.skill as skill_module
import server.services.agent_registry.defaults as registry_defaults_module
import server.services.agent_registry.models as registry_models_module
from agiwo.agent.models.config import AgentOptions as SDKAgentOptions
from agiwo.agent import AgentOptions
from agiwo.llm.base import Model
from agiwo.llm.openai import OpenAIModel
from agiwo.skill.loader import SkillLoader
from agiwo.skill.registry import SkillRegistry
from agiwo.skill.skill_tool import SkillTool
import server.services.runtime.agent_factory as agent_factory_module
from server.config import ConsoleConfig, DefaultAgentConfig
from server.models.agent_config import AgentOptionsInput
from server.models.view import AgentConfigPayload
from server.services.agent_registry import (
    AgentConfigRecord,
    AgentRegistry,
    build_default_agent_record as build_registry_default_agent_record,
)
from server.services.runtime.agent_factory import (
    agent_options_input_to_agent_options,
    build_agent,
    build_default_agent_record,
    build_model,
)
from server.services.storage_wiring import (
    build_run_step_storage_config,
    build_trace_storage_config,
)
from server.services.tool_catalog.tool_builder import build_tools
from agiwo.tool.reference import (
    InvalidToolReferenceError,
    BuiltinToolReference,
    AgentToolReference,
    parse_tool_reference,
)


class MockModel(Model):
    async def arun_stream(self, messages, tools=None):
        if False:
            yield None


class DummyRegistry:
    def __init__(self, records: dict[str, AgentConfigRecord] | None = None) -> None:
        self._records = records or {}

    async def get_agent(self, agent_id: str) -> AgentConfigRecord | None:
        return self._records.get(agent_id)


class DummySkillManager:
    def expand_allowed_skills(
        self,
        allowed_skills,
        *,
        available_skill_names=None,
    ):
        del available_skill_names
        if allowed_skills is None:
            return None
        return list(allowed_skills)

    def create_skill_tool(self, allowed_skills=None) -> SkillTool:
        registry = SkillRegistry()
        loader = SkillLoader(registry)
        return SkillTool(registry, loader, allowed_skills)

    def validate_explicit_allowed_skills(
        self,
        allowed_skills,
        *,
        available_skill_names=None,
    ):
        del available_skill_names
        if allowed_skills is None:
            return None
        return list(allowed_skills)


@pytest.fixture
def stub_skill_manager(monkeypatch: pytest.MonkeyPatch) -> DummySkillManager:
    manager = DummySkillManager()
    monkeypatch.setattr(agent_module, "get_global_skill_manager", lambda: manager)
    monkeypatch.setattr(
        skill_module,
        "get_global_skill_manager",
        lambda: manager,
    )
    monkeypatch.setattr(
        registry_defaults_module,
        "get_global_skill_manager",
        lambda: manager,
    )
    monkeypatch.setattr(
        registry_models_module,
        "get_global_skill_manager",
        lambda: manager,
    )
    return manager


def test_console_config_reads_uppercase_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGIWO_CONSOLE_FEISHU_ENABLED", "true")

    config = ConsoleConfig()

    assert config.channels.feishu.enabled is True


def test_console_config_rejects_legacy_default_agent_group_env_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGIWO_CONSOLE_DEFAULT_AGENT_MODEL__PROVIDER", "deepseek")
    monkeypatch.setenv("AGIWO_CONSOLE_DEFAULT_AGENT_MODEL__NAME", "deepseek-chat")
    monkeypatch.setenv(
        "AGIWO_CONSOLE_DEFAULT_AGENT_SYSTEM__PROMPT",
        "You are Walaha.",
    )

    with pytest.raises(
        ValidationError,
        match="AGIWO_CONSOLE_DEFAULT_AGENT__MODEL_PROVIDER",
    ):
        ConsoleConfig()


def test_console_config_rejects_partial_default_agent_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGIWO_CONSOLE_DEFAULT_AGENT__ID", "Walaha000")
    monkeypatch.setenv("AGIWO_CONSOLE_DEFAULT_AGENT__NAME", "Walaha")
    monkeypatch.setenv(
        "AGIWO_CONSOLE_DEFAULT_AGENT__MODEL_PARAMS__BASE_URL",
        "https://api.deepseek.com",
    )
    monkeypatch.setenv(
        "AGIWO_CONSOLE_DEFAULT_AGENT__MODEL_PARAMS__API_KEY_ENV_NAME",
        "DEEPSEEK_API_KEY",
    )

    with pytest.raises(
        ValidationError,
        match="AGIWO_CONSOLE_DEFAULT_AGENT__MODEL_PROVIDER",
    ):
        ConsoleConfig()


def test_console_config_rejects_legacy_default_model_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGIWO_CONSOLE_DEFAULT_AGENT__MODEL_PROVIDER", "generic")

    with pytest.raises(ValidationError):
        ConsoleConfig()


def test_console_config_rejects_plain_api_key_in_default_model_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AGIWO_CONSOLE_DEFAULT_AGENT__MODEL_PROVIDER",
        "openai-compatible",
    )
    monkeypatch.setenv(
        "AGIWO_CONSOLE_DEFAULT_AGENT__MODEL_NAME",
        "MiniMax-M2.5",
    )
    monkeypatch.setenv(
        "AGIWO_CONSOLE_DEFAULT_AGENT__MODEL_PARAMS",
        '{"api_key":"sk-plain-text","base_url":"https://api.example.com/v1"}',
    )

    with pytest.raises(ValidationError, match="api_key"):
        ConsoleConfig()


def test_agent_options_input_defaults() -> None:
    console_config = ConsoleConfig(
        run_step_storage_type="memory",
        trace_storage_type="memory",
        metadata_storage_type="memory",
    )
    opts = AgentOptionsInput.model_validate({})
    options = agent_options_input_to_agent_options(
        opts,
        run_step_storage=build_run_step_storage_config(console_config),
        trace_storage=build_trace_storage_config(console_config),
    )

    assert options.max_steps == 50
    assert options.enable_termination_summary is True


def test_default_agent_record_uses_shared_option_defaults() -> None:
    template = DefaultAgentConfig()

    record = build_default_agent_record(template)

    expected = AgentOptionsInput.model_validate({}).model_dump(exclude_none=True)
    assert record.options == expected
    assert record.options["max_steps"] == AgentOptions().max_steps
    assert record.allowed_skills is None


def test_default_agent_record_preserves_empty_allowed_tools() -> None:
    template = DefaultAgentConfig(
        model_provider="openai",
        model_name="gpt-4o-mini",
        allowed_tools=[],
    )

    record = build_default_agent_record(template)

    assert record.allowed_tools == []


def test_console_default_agent_config_normalizes_allowed_skills() -> None:
    config = ConsoleConfig(
        default_agent={
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "allowed_skills": [" skill*", "", "*review ", "skill*"],
        }
    )

    assert config.default_agent.allowed_skills == ["skill*", "*review"]


def test_agent_options_input_maps_all_fields() -> None:
    console_config = ConsoleConfig(
        run_step_storage_type="memory",
        trace_storage_type="memory",
        metadata_storage_type="memory",
    )
    opts = AgentOptionsInput.model_validate(
        {
            "config_root": "/tmp/agent-root",
            "max_steps": 42,
            "run_timeout": 120,
            "max_input_tokens_per_call": 64000,
            "max_run_cost": 1.25,
            "enable_termination_summary": False,
            "termination_summary_prompt": "Summarize before exit",
            "relevant_memory_max_token": 1024,
            "stream_cleanup_timeout": 90.5,
            "compact_prompt": "Compact the context",
            "enable_context_rollback": False,
            "enable_tool_retrospect": False,
            "retrospect_token_threshold": 2049,
            "retrospect_round_interval": 7,
            "retrospect_accumulated_token_threshold": 16384,
        }
    )
    options = agent_options_input_to_agent_options(
        opts,
        run_step_storage=build_run_step_storage_config(console_config),
        trace_storage=build_trace_storage_config(console_config),
    )

    assert options.config_root == "/tmp/agent-root"
    assert options.max_steps == 42
    assert options.run_timeout == 120
    assert options.max_input_tokens_per_call == 64000
    assert options.max_run_cost == 1.25
    assert options.enable_termination_summary is False
    assert options.termination_summary_prompt == "Summarize before exit"
    assert options.relevant_memory_max_token == 1024
    assert options.stream_cleanup_timeout == 90.5
    assert options.compact_prompt == "Compact the context"
    assert options.enable_context_rollback is False
    assert options.enable_tool_retrospect is False
    assert options.retrospect_token_threshold == 2049
    assert options.retrospect_round_interval == 7
    assert options.retrospect_accumulated_token_threshold == 16384


def test_console_agent_options_input_matches_sdk_agent_options() -> None:
    expected = set(SDKAgentOptions.model_fields) - {"storage"}
    actual = set(AgentOptionsInput.model_fields)
    assert actual == expected


@pytest.mark.asyncio
async def test_default_agent_record_is_consistent_across_entrypoints(
    stub_skill_manager: DummySkillManager,
) -> None:
    del stub_skill_manager
    config = ConsoleConfig(
        run_step_storage_type="memory",
        trace_storage_type="memory",
        metadata_storage_type="memory",
        default_agent={
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "allowed_tools": None,
            "allowed_skills": ["alpha", "beta"],
            "model_params": {"max_output_tokens": 128},
        },
    )
    registry = AgentRegistry(config)
    await registry.initialize()
    try:
        shared = build_registry_default_agent_record(config.default_agent)
        runtime = build_default_agent_record(config.default_agent)
        by_id = await registry.get_agent(config.default_agent.id)
        by_name = await registry.get_agent_by_name(config.default_agent.name)
    finally:
        await registry.close()

    exclude_timestamps = {"created_at", "updated_at"}
    shared_payload = shared.model_dump(exclude=exclude_timestamps)
    assert runtime.model_dump(exclude=exclude_timestamps) == shared_payload
    assert by_id is not None
    assert by_name is not None
    assert by_id.model_dump(exclude=exclude_timestamps) == shared_payload
    assert by_name.model_dump(exclude=exclude_timestamps) == shared_payload


def test_console_tool_catalog_parses_builtin_and_agent_refs() -> None:
    ref1 = parse_tool_reference("web_search")
    assert isinstance(ref1, BuiltinToolReference)
    assert ref1.name == "web_search"

    ref2 = parse_tool_reference("agent:child-1")
    assert isinstance(ref2, AgentToolReference)
    assert ref2.agent_id == "child-1"

    with pytest.raises(InvalidToolReferenceError):
        parse_tool_reference("missing")

    with pytest.raises(InvalidToolReferenceError):
        parse_tool_reference("agent:")


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


@pytest.mark.asyncio
async def test_build_agent_preserves_empty_allowed_skills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_factory_module,
        "build_model",
        lambda _config: MockModel(id="mock", name="mock", provider="openai"),
    )
    console_config = ConsoleConfig(
        run_step_storage_type="memory",
        trace_storage_type="memory",
        metadata_storage_type="memory",
    )

    agent = await build_agent(
        AgentConfigRecord(
            name="tester",
            model_provider="openai",
            model_name="gpt-4o-mini",
            allowed_skills=[],
        ),
        console_config,
        DummyRegistry(),
    )

    assert agent.config.allowed_skills == []
    assert all(tool.name != "skill" for tool in agent.tools)


@pytest.mark.asyncio
async def test_build_agent_tool_preserves_delegate_skill_access(
    monkeypatch: pytest.MonkeyPatch,
    stub_skill_manager: DummySkillManager,
) -> None:
    del stub_skill_manager
    monkeypatch.setattr(
        agent_factory_module,
        "build_model",
        lambda _config: MockModel(id="mock", name="mock", provider="openai"),
    )
    console_config = ConsoleConfig(
        run_step_storage_type="memory",
        trace_storage_type="memory",
        metadata_storage_type="memory",
    )
    child = AgentConfigRecord(
        id="child",
        name="child",
        model_provider="openai",
        model_name="gpt-4o-mini",
        allowed_skills=["alpha"],
    )
    parent = AgentConfigRecord(
        id="parent",
        name="parent",
        model_provider="openai",
        model_name="gpt-4o-mini",
        allowed_tools=["agent:child"],
    )

    agent = await build_agent(
        parent,
        console_config,
        DummyRegistry({"child": child}),
    )

    agent_tool = next(tool for tool in agent.tools if tool.name == "child")
    wrapped_agent = agent_tool._agent
    skill_tool = next(tool for tool in wrapped_agent.tools if tool.name == "skill")

    assert wrapped_agent.config.allowed_skills == ["alpha"]
    assert skill_tool._allowed_skills == frozenset({"alpha"})


def test_agent_config_record_rejects_unknown_allowed_skills() -> None:
    with pytest.raises(ValidationError, match="Unknown allowed skill"):
        AgentConfigRecord.model_validate(
            {
                "name": "tester",
                "model_provider": "openai",
                "model_name": "gpt-4o-mini",
                "allowed_skills": ["definitely-not-a-real-skill"],
            }
        )


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


def test_agent_config_record_rejects_wildcard_allowed_skills() -> None:
    with pytest.raises(ValidationError, match="explicit skill names"):
        AgentConfigRecord.model_validate(
            {
                "name": "tester",
                "model_provider": "openai",
                "model_name": "gpt-4o-mini",
                "allowed_skills": ["skill*"],
            }
        )


def test_agent_config_payload_uses_defaults_when_options_omitted() -> None:
    payload = AgentConfigPayload.model_validate(
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
    assert payload.options.max_steps == 50
    assert payload.model_params.max_output_tokens == 4096


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
                allowed_tools=["web_search"],
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
        assert updated.allowed_tools == ["web_search"]
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
