"""Shared Mongo runtime helpers for storage implementations."""

from collections.abc import Sequence
from dataclasses import dataclass

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)

from agiwo.utils.logging import FilteringBoundLogger
from agiwo.utils.mongo_pool import get_shared_mongo_client, release_shared_mongo_client


@dataclass(frozen=True)
class MongoIndexSpec:
    """Declarative description of one MongoDB index."""

    keys: str | list[tuple[str, int]]
    unique: bool = False


class MongoCollectionRuntime:
    """Own the shared Mongo client lifecycle for one store instance."""

    def __init__(
        self,
        uri: str,
        db_name: str,
        *,
        logger: FilteringBoundLogger,
        connect_event: str,
        disconnect_event: str | None = None,
    ) -> None:
        self.uri = uri
        self.db_name = db_name
        self._logger = logger
        self._connect_event = connect_event
        self._disconnect_event = disconnect_event
        self._client: AsyncIOMotorClient | None = None
        self._db: AsyncIOMotorDatabase | None = None
        self._collections: dict[str, AsyncIOMotorCollection] = {}

    @property
    def client(self) -> AsyncIOMotorClient | None:
        return self._client

    @property
    def db(self) -> AsyncIOMotorDatabase | None:
        return self._db

    async def ensure_collection(
        self,
        collection_name: str,
        *,
        indexes: Sequence[MongoIndexSpec] = (),
    ) -> AsyncIOMotorCollection:
        if collection_name in self._collections:
            return self._collections[collection_name]

        if self._client is None:
            self._client = await get_shared_mongo_client(self.uri)
            self._db = self._client[self.db_name]
            self._logger.info(
                self._connect_event,
                uri=self.uri,
                db_name=self.db_name,
            )

        if self._db is None:
            raise RuntimeError("Database connection not established")

        collection = self._db[collection_name]
        if collection is None:
            raise RuntimeError(f"Failed to get MongoDB collection: {collection_name}")

        for index in indexes:
            await collection.create_index(index.keys, unique=index.unique)

        self._collections[collection_name] = collection
        return collection

    async def disconnect(self) -> None:
        if self._client is None:
            return

        await release_shared_mongo_client(self.uri)
        self._client = None
        self._db = None
        self._collections.clear()
        if self._disconnect_event is not None:
            self._logger.info(
                self._disconnect_event,
                uri=self.uri,
                db_name=self.db_name,
            )


__all__ = ["MongoCollectionRuntime", "MongoIndexSpec"]
