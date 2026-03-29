from datetime import datetime, timedelta

import pytest

from agiwo.config.settings import settings as sdk_settings
from server.config import ConsoleConfig
from server.services.agent_registry import AgentConfigRecord
from server.services.agent_registry.store import create_agent_registry_store
from server.services.agent_registry.store.memory import InMemoryAgentRegistryStore
from server.services.agent_registry.store.sqlite import SqliteAgentRegistryStore


def _make_record(
    *,
    record_id: str,
    name: str = "agent",
    updated_at: datetime,
) -> AgentConfigRecord:
    return AgentConfigRecord(
        id=record_id,
        name=name,
        model_provider="openai-compatible",
        model_name="gpt-test",
        model_params={
            "base_url": "https://api.example.com/v1",
            "api_key_env_name": "TEST_API_KEY",
        },
        created_at=updated_at - timedelta(minutes=1),
        updated_at=updated_at,
    )


def _make_store(
    kind: str,
    *,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    if kind == "sqlite":
        monkeypatch.setattr(
            sdk_settings,
            "sqlite_db_path",
            str(tmp_path / "agent-registry.sqlite3"),
        )
    return create_agent_registry_store(
        ConsoleConfig(
            run_step_storage_type="memory",
            trace_storage_type="memory",
            metadata_storage_type=kind,
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["memory", "sqlite"])
async def test_agent_registry_store_round_trips_and_orders_records(
    kind: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _make_store(kind, tmp_path=tmp_path, monkeypatch=monkeypatch)
    await store.connect()
    try:
        now = datetime.now()
        older = _make_record(record_id="agent-1", name="shared", updated_at=now)
        newer = _make_record(
            record_id="agent-2",
            name="shared",
            updated_at=now + timedelta(minutes=5),
        )

        await store.upsert_agent(older)
        await store.upsert_agent(newer)

        listed = await store.list_agents()
        fetched = await store.get_agent("agent-1")
        fetched_by_name = await store.get_agent_by_name("shared")
        deleted = await store.delete_agent("agent-1")

        assert [record.id for record in listed] == ["agent-2", "agent-1"]
        assert fetched == older
        assert fetched_by_name == newer
        assert deleted is True
        assert await store.get_agent("agent-1") is None
    finally:
        await store.close()


def test_agent_registry_store_factory_selects_backend(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sdk_settings,
        "sqlite_db_path",
        str(tmp_path / "agent-registry.sqlite3"),
    )

    memory_store = create_agent_registry_store(
        ConsoleConfig(metadata_storage_type="memory")
    )
    sqlite_store = create_agent_registry_store(
        ConsoleConfig(metadata_storage_type="sqlite")
    )

    assert isinstance(memory_store, InMemoryAgentRegistryStore)
    assert isinstance(sqlite_store, SqliteAgentRegistryStore)
