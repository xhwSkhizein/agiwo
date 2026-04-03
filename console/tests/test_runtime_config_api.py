"""Integration tests for runtime config inspection and update APIs."""

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

import server.services.runtime_config as runtime_config_module
from agiwo.config.settings import get_settings
from server.app import create_app
from server.config import ConsoleConfig
from server.dependencies import (
    ConsoleRuntime,
    bind_console_runtime,
    clear_console_runtime,
)
from server.services.agent_registry import AgentRegistry
from server.services.runtime_config import RuntimeConfigService
from server.services.storage_wiring import create_run_step_storage, create_trace_storage


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch):
    class DummySkillManager:
        async def initialize(self) -> None:
            return None

        def expand_allowed_skills(self, allowed_skills):
            if allowed_skills == ["skill*"]:
                return ["skill-alpha", "skill-beta"]
            return list(allowed_skills or [])

        def get_resolved_skills_dirs(self) -> list[Path]:
            return [Path("/tmp/runtime-skills"), Path("/tmp/runtime-extra")]

    app = create_app()
    config = ConsoleConfig(
        run_step_storage_type="memory",
        trace_storage_type="memory",
        metadata_storage_type="memory",
    )
    run_step_storage = create_run_step_storage(config)
    trace_storage = create_trace_storage(config)
    registry = AgentRegistry(config)
    await registry.initialize()
    runtime_config_service = RuntimeConfigService(config)

    monkeypatch.setattr(
        runtime_config_module,
        "get_global_skill_manager",
        lambda: DummySkillManager(),
    )
    monkeypatch.setattr(
        RuntimeConfigService,
        "_build_skill_manager_for_settings",
        lambda self, runtime_settings: DummySkillManager(),
    )
    settings = get_settings()
    original_skills_dirs = list(settings.skills_dirs)
    settings.skills_dirs = ["examples/skills", "skills"]

    bind_console_runtime(
        app,
        ConsoleRuntime(
            config=config,
            run_step_storage=run_step_storage,
            trace_storage=trace_storage,
            agent_registry=registry,
            runtime_config_service=runtime_config_service,
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    settings.skills_dirs = original_skills_dirs
    clear_console_runtime(app)
    await registry.close()
    await trace_storage.close()
    await run_step_storage.close()


@pytest.mark.asyncio
async def test_get_runtime_config_returns_editable_and_readonly_sections(
    client,
) -> None:
    response = await client.get("/api/config/runtime")

    assert response.status_code == 200
    payload = response.json()
    assert payload["runtime_only"] is True
    assert payload["editable"]["skills_dirs"] == ["examples/skills", "skills"]
    assert payload["effective"]["resolved_skills_dirs"] == [
        "/tmp/runtime-skills",
        "/tmp/runtime-extra",
    ]
    assert "console" in payload["readonly"]
    assert "sdk" in payload["readonly"]


@pytest.mark.asyncio
async def test_update_runtime_config_applies_skills_dirs_and_default_agent(
    client,
) -> None:
    response = await client.put(
        "/api/config/runtime",
        json={
            "skills_dirs": ["runtime/skills", "workspace/skills"],
            "default_agent": {
                "id": "default-console-agent",
                "name": "Runtime Default",
                "description": "Updated at runtime",
                "model_provider": "openai-compatible",
                "model_name": "qwen3-coder-plus",
                "system_prompt": "Follow the runtime config",
                "tools": ["bash", "web_search"],
                "allowed_skills": ["skill*"],
                "model_params": {
                    "base_url": "https://example.com/v1",
                    "api_key_env_name": "RUNTIME_API_KEY",
                    "temperature": 0.1,
                },
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["editable"]["skills_dirs"] == ["runtime/skills", "workspace/skills"]
    assert payload["editable"]["default_agent"]["name"] == "Runtime Default"
    assert payload["editable"]["default_agent"]["allowed_skills"] == [
        "skill-alpha",
        "skill-beta",
    ]
    assert payload["editable"]["default_agent"]["tools"] == ["bash", "web_search"]

    settings = get_settings()
    assert settings.skills_dirs == ["runtime/skills", "workspace/skills"]
