"""MongoDB backend for agent registry."""

from agiwo.utils.mongo_pool import (
    get_shared_mongo_client,
    release_shared_mongo_client,
)
from server.services.agent_registry_models import AgentConfigRecord


class MongoAgentRegistryStore:
    def __init__(self, mongo_uri: str, mongo_db_name: str) -> None:
        self._mongo_uri = mongo_uri
        self._mongo_db_name = mongo_db_name
        self._collection: object | None = None

    async def connect(self) -> None:
        if self._collection is not None:
            return
        client = await get_shared_mongo_client(self._mongo_uri)
        database = client[self._mongo_db_name]
        self._collection = database["agent_configs"]
        await self._collection.create_index("id", unique=True)

    async def close(self) -> None:
        if self._collection is None:
            return
        await release_shared_mongo_client(self._mongo_uri)
        self._collection = None

    async def list_agents(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AgentConfigRecord]:
        collection = await self._require_collection()
        cursor = collection.find().sort("updated_at", -1).skip(offset).limit(limit)
        records: list[AgentConfigRecord] = []
        async for document in cursor:
            records.append(self._deserialize_document(document))
        return records

    async def get_agent(self, agent_id: str) -> AgentConfigRecord | None:
        collection = await self._require_collection()
        document = await collection.find_one({"id": agent_id})
        if document is None:
            return None
        return self._deserialize_document(document)

    async def get_agent_by_name(self, agent_name: str) -> AgentConfigRecord | None:
        collection = await self._require_collection()
        results = (
            await collection.find({"name": agent_name})
            .sort("updated_at", -1)
            .limit(1)
            .to_list(length=1)
        )
        if not results:
            return None
        return self._deserialize_document(results[0])

    async def upsert_agent(self, record: AgentConfigRecord) -> None:
        collection = await self._require_collection()
        await collection.replace_one(
            {"id": record.id},
            record.model_dump(mode="json"),
            upsert=True,
        )

    async def delete_agent(self, agent_id: str) -> bool:
        collection = await self._require_collection()
        result = await collection.delete_one({"id": agent_id})
        return result.deleted_count > 0

    async def _require_collection(self) -> object:
        if self._collection is None:
            await self.connect()
        assert self._collection is not None
        return self._collection

    def _deserialize_document(self, document: dict[str, object]) -> AgentConfigRecord:
        normalized = dict(document)
        normalized.pop("_id", None)
        return AgentConfigRecord.model_validate(normalized)


__all__ = ["MongoAgentRegistryStore"]
