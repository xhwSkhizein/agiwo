"""
Integration tests for the Agents API endpoints.
"""

import pytest

from httpx import ASGITransport, AsyncClient

import server.routers.agents as agents_router
import server.services.agent_registry.models as registry_models_module
from server.app import create_app
from server.config import ConsoleConfig
from server.dependencies import (
    ConsoleRuntime,
    bind_console_runtime,
    clear_console_runtime,
    get_console_runtime_from_app,
)
from server.services.agent_registry import AgentConfigRecord, AgentRegistry
from server.services.storage_wiring import create_run_step_storage, create_trace_storage


def _runtime(client: AsyncClient) -> ConsoleRuntime:
    return get_console_runtime_from_app(client._transport.app)  # type: ignore[attr-defined]


@pytest.fixture
async def client():
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
    bind_console_runtime(
        app,
        ConsoleRuntime(
            config=config,
            run_step_storage=run_step_storage,
            trace_storage=trace_storage,
            agent_registry=registry,
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    clear_console_runtime(app)
    await registry.close()
    await run_step_storage.close()


@pytest.mark.asyncio
async def test_list_agents_includes_default_env_agent_when_registry_store_is_empty(
    client,
) -> None:
    response = await client.get("/api/agents")

    assert response.status_code == 200
    payload = response.json()
    assert [item["id"] for item in payload] == ["default-console-agent"]
    assert payload[0]["name"] == "Console Agent"
    assert payload[0]["is_default"] is True


@pytest.mark.asyncio
async def test_update_agent_put_replaces_full_agent_config(client) -> None:
    registry = _runtime(client).agent_registry
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

    response = await client.put(
        f"/api/agents/{created.id}",
        json={
            "name": "tester-v2",
            "description": "replacement",
            "model_provider": "openai-compatible",
            "model_name": "MiniMax-M2.5",
            "system_prompt": "Use the new instructions",
            "allowed_tools": ["web_search"],
            "options": {"max_steps": 5, "max_run_cost": None},
            "model_params": {
                "base_url": "https://api.other.example/v1",
                "api_key_env_name": "OTHER_API_KEY",
                "temperature": 0.2,
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "tester-v2"
    assert payload["description"] == "replacement"
    assert payload["system_prompt"] == "Use the new instructions"
    assert payload["allowed_tools"] == ["web_search"]
    assert payload["options"]["max_steps"] == 5
    assert payload["options"]["max_run_cost"] is None
    assert payload["model_params"]["base_url"] == "https://api.other.example/v1"
    assert payload["model_params"]["api_key_env_name"] == "OTHER_API_KEY"
    assert payload["model_params"]["temperature"] == 0.2


@pytest.mark.asyncio
async def test_list_available_tools_uses_catalog_for_builtin_and_agent_entries(
    client,
) -> None:
    registry = _runtime(client).agent_registry
    created = await registry.create_agent(
        AgentConfigRecord(
            name="delegate",
            description="Delegate work",
            model_provider="openai",
            model_name="gpt-4o-mini",
        )
    )

    response = await client.get("/api/agents/tools/available")

    assert response.status_code == 200
    payload = response.json()
    builtin_names = {item["name"] for item in payload if item["type"] == "builtin"}
    agent_names = {item["name"] for item in payload if item["type"] == "agent"}
    assert "web_search" in builtin_names
    assert f"agent:{created.id}" in agent_names


@pytest.mark.asyncio
async def test_update_agent_put_rejects_partial_payloads(client) -> None:
    registry = _runtime(client).agent_registry
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

    response = await client.put(
        f"/api/agents/{created.id}",
        json={"model_params": {"api_key_env_name": None}},
    )

    assert response.status_code == 422
    assert "name" in response.text
    assert "model_name" in response.text


@pytest.mark.asyncio
async def test_create_agent_accepts_custom_tool_names(client) -> None:
    """Unknown tool names are accepted — they may refer to user-supplied custom tools."""
    response = await client.post(
        "/api/agents",
        json={
            "name": "tester",
            "description": "",
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "system_prompt": "",
            "allowed_tools": ["custom-tool"],
            "options": {},
            "model_params": {},
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert "custom-tool" in data["allowed_tools"]


@pytest.mark.asyncio
async def test_update_agent_rejects_invalid_agent_tool_reference(client) -> None:
    registry = _runtime(client).agent_registry
    created = await registry.create_agent(
        AgentConfigRecord(
            name="tester",
            model_provider="openai",
            model_name="gpt-4o-mini",
        )
    )

    response = await client.put(
        f"/api/agents/{created.id}",
        json={
            "name": "tester",
            "description": "",
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "system_prompt": "",
            "allowed_tools": ["agent:   "],
            "options": {},
            "model_params": {},
        },
    )

    assert response.status_code == 422
    assert "Empty agent id" in response.text


@pytest.mark.asyncio
async def test_create_agent_expands_allowed_skill_patterns_at_api_boundary(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummySkillManager:
        def expand_allowed_skills(self, allowed_skills):
            assert allowed_skills == ["skill*", "*review"]
            return ["skill-review", "skill-build", "code-review"]

        def validate_explicit_allowed_skills(
            self, allowed_skills, *, available_skill_names=None
        ):
            del available_skill_names
            return list(allowed_skills)

    monkeypatch.setattr(
        agents_router,
        "get_global_skill_manager",
        lambda: DummySkillManager(),
    )
    monkeypatch.setattr(
        registry_models_module,
        "get_global_skill_manager",
        lambda: DummySkillManager(),
    )

    response = await client.post(
        "/api/agents",
        json={
            "name": "tester",
            "description": "",
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "system_prompt": "",
            "allowed_tools": [],
            "allowed_skills": ["skill*", "*review"],
            "options": {},
            "model_params": {},
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["allowed_skills"] == [
        "skill-review",
        "skill-build",
        "code-review",
    ]


@pytest.mark.asyncio
async def test_create_agent_rejects_unknown_exact_allowed_skill(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummySkillManager:
        def expand_allowed_skills(self, allowed_skills):
            assert allowed_skills == ["missing-skill"]
            raise ValueError("Unknown allowed skill(s): missing-skill")

        def validate_explicit_allowed_skills(
            self, allowed_skills, *, available_skill_names=None
        ):
            del available_skill_names
            return list(allowed_skills)

    monkeypatch.setattr(
        agents_router,
        "get_global_skill_manager",
        lambda: DummySkillManager(),
    )

    response = await client.post(
        "/api/agents",
        json={
            "name": "tester",
            "description": "",
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "system_prompt": "",
            "allowed_tools": [],
            "allowed_skills": ["missing-skill"],
            "options": {},
            "model_params": {},
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Unknown allowed skill(s): missing-skill"


@pytest.mark.asyncio
async def test_get_agent_capabilities_returns_provider_schema(client) -> None:
    response = await client.get("/api/agents/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert "providers" in payload
    providers = {item["value"]: item for item in payload["providers"]}
    assert "openai" in providers
    assert "openai-compatible" in providers
    assert providers["openai-compatible"]["requires_base_url"] is True
    assert providers["openai-compatible"]["requires_api_key_env_name"] is True
