"""Shared MongoDB client pool."""

import asyncio

from motor.motor_asyncio import AsyncIOMotorClient

from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class MongoClientPool:
    """Process-wide shared AsyncIOMotorClient pool keyed by URI."""

    def __init__(self) -> None:
        self._clients: dict[str, AsyncIOMotorClient] = {}
        self._ref_counts: dict[str, int] = {}
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get_client(self, mongo_uri: str) -> AsyncIOMotorClient:
        lock = self._get_lock()
        async with lock:
            if mongo_uri not in self._clients:
                client = AsyncIOMotorClient(mongo_uri)
                self._clients[mongo_uri] = client
                self._ref_counts[mongo_uri] = 0
                logger.info("mongo_pool_client_created", mongo_uri=mongo_uri)

            self._ref_counts[mongo_uri] += 1
            return self._clients[mongo_uri]

    async def release_client(self, mongo_uri: str) -> None:
        lock = self._get_lock()
        async with lock:
            client = self._clients.get(mongo_uri)
            if client is None:
                return

            self._ref_counts[mongo_uri] -= 1
            if self._ref_counts[mongo_uri] <= 0:
                client.close()
                self._clients.pop(mongo_uri, None)
                self._ref_counts.pop(mongo_uri, None)
                logger.info("mongo_pool_client_closed", mongo_uri=mongo_uri)

    async def close_all(self) -> None:
        lock = self._get_lock()
        async with lock:
            for mongo_uri, client in list(self._clients.items()):
                client.close()
                logger.info("mongo_pool_client_closed", mongo_uri=mongo_uri)
            self._clients.clear()
            self._ref_counts.clear()

    def get_client_count(self) -> int:
        return len(self._clients)


_pool: MongoClientPool | None = None


def get_mongo_pool() -> MongoClientPool:
    global _pool
    if _pool is None:
        _pool = MongoClientPool()
    return _pool


async def get_shared_mongo_client(mongo_uri: str) -> AsyncIOMotorClient:
    return await get_mongo_pool().get_client(mongo_uri)


async def release_shared_mongo_client(mongo_uri: str) -> None:
    await get_mongo_pool().release_client(mongo_uri)


async def close_all_mongo_clients() -> None:
    await get_mongo_pool().close_all()


def reset_mongo_pool() -> None:
    """Reset the global pool instance (useful for testing)."""
    global _pool
    _pool = None


__all__ = [
    "MongoClientPool",
    "get_mongo_pool",
    "get_shared_mongo_client",
    "release_shared_mongo_client",
    "close_all_mongo_clients",
    "reset_mongo_pool",
]
