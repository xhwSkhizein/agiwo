import pytest

from server.config import ConsoleConfig
from server.schemas import AgentOptionsPayload
from server.services.agent_builder import build_agent_options
from server.services.agent_registry import AgentConfigRecord
from server.services import agent_builder


def test_console_config_reads_uppercase_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGIWO_CONSOLE_FEISHU_ENABLED", "true")

    config = ConsoleConfig()

    assert config.feishu_enabled is True


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
