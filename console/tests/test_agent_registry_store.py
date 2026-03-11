from datetime import datetime, timedelta
from typing import Any

import pytest

from agiwo.config.settings import settings as sdk_settings
from server.config import ConsoleConfig
from server.services.agent_registry import AgentConfigRecord
from server.services.agent_registry_memory_store import InMemoryAgentRegistryStore
from server.services.agent_registry_mongo_store import MongoAgentRegistryStore
from server.services.agent_registry_sqlite_store import SqliteAgentRegistryStore
from server.services.agent_registry_store import create_agent_registry_store


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
    mongo_store = create_agent_registry_store(
        ConsoleConfig(metadata_storage_type="mongodb")
    )

    assert isinstance(memory_store, InMemoryAgentRegistryStore)
    assert isinstance(sqlite_store, SqliteAgentRegistryStore)
    assert isinstance(mongo_store, MongoAgentRegistryStore)


class FakeDeleteResult:
    def __init__(self, deleted_count: int) -> None:
        self.deleted_count = deleted_count


class FakeMongoCursor:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self._documents = list(documents)
        self._offset = 0
        self._limit: int | None = None
        self._iterator: iter[dict[str, Any]] | None = None

    def sort(self, field: str, direction: int) -> "FakeMongoCursor":
        reverse = direction < 0
        self._documents.sort(key=lambda doc: doc.get(field), reverse=reverse)
        return self

    def skip(self, offset: int) -> "FakeMongoCursor":
        self._offset = offset
        return self

    def limit(self, limit: int) -> "FakeMongoCursor":
        self._limit = limit
        return self

    async def to_list(self, length: int) -> list[dict[str, Any]]:
        return [dict(doc) for doc in self._sliced_documents()[:length]]

    def __aiter__(self) -> "FakeMongoCursor":
        self._iterator = iter(self._sliced_documents())
        return self

    async def __anext__(self) -> dict[str, Any]:
        assert self._iterator is not None
        try:
            return dict(next(self._iterator))
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    def _sliced_documents(self) -> list[dict[str, Any]]:
        documents = self._documents[self._offset:]
        if self._limit is not None:
            documents = documents[:self._limit]
        return documents


class FakeMongoCollection:
    def __init__(self) -> None:
        self._documents: dict[str, dict[str, Any]] = {}
        self.indexes: list[tuple[str, bool]] = []

    async def create_index(self, field: str, unique: bool = False) -> None:
        self.indexes.append((field, unique))

    def find(self, query: dict[str, Any] | None = None) -> FakeMongoCursor:
        query = query or {}
        documents = [
            dict(document)
            for document in self._documents.values()
            if all(document.get(key) == value for key, value in query.items())
        ]
        return FakeMongoCursor(documents)

    async def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        for document in self._documents.values():
            if all(document.get(key) == value for key, value in query.items()):
                return dict(document)
        return None

    async def replace_one(
        self,
        query: dict[str, Any],
        document: dict[str, Any],
        *,
        upsert: bool,
    ) -> None:
        del query, upsert
        self._documents[document["id"]] = dict(document)

    async def delete_one(self, query: dict[str, Any]) -> FakeDeleteResult:
        deleted = self._documents.pop(query["id"], None)
        return FakeDeleteResult(1 if deleted is not None else 0)


class FakeMongoDatabase:
    def __init__(self) -> None:
        self._collections: dict[str, FakeMongoCollection] = {}

    def __getitem__(self, name: str) -> FakeMongoCollection:
        if name not in self._collections:
            self._collections[name] = FakeMongoCollection()
        return self._collections[name]


class FakeMongoClient:
    def __init__(self) -> None:
        self._databases: dict[str, FakeMongoDatabase] = {}

    def __getitem__(self, name: str) -> FakeMongoDatabase:
        if name not in self._databases:
            self._databases[name] = FakeMongoDatabase()
        return self._databases[name]


@pytest.mark.asyncio
async def test_mongo_agent_registry_store_round_trips_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeMongoClient()
    released: list[str] = []

    async def fake_get_shared_mongo_client(mongo_uri: str) -> FakeMongoClient:
        assert mongo_uri == "mongodb://test"
        return fake_client

    async def fake_release_shared_mongo_client(mongo_uri: str) -> None:
        released.append(mongo_uri)

    monkeypatch.setattr(
        "server.services.agent_registry_mongo_store.get_shared_mongo_client",
        fake_get_shared_mongo_client,
    )
    monkeypatch.setattr(
        "server.services.agent_registry_mongo_store.release_shared_mongo_client",
        fake_release_shared_mongo_client,
    )

    store = MongoAgentRegistryStore(
        mongo_uri="mongodb://test",
        mongo_db_name="agiwo-test",
    )
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
    finally:
        await store.close()

    assert released == ["mongodb://test"]
