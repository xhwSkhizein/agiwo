from pathlib import Path

import pytest

from agiwo.agent.storage.session import SQLiteSessionStorage
from agiwo.tool.storage.citation.sqlite_store import SQLiteCitationStore
from agiwo.utils.mongo_pool import (
    get_mongo_pool,
    get_shared_mongo_client,
    release_shared_mongo_client,
)
from agiwo.utils.sqlite_pool import get_sqlite_pool


@pytest.mark.asyncio
async def test_sqlite_stores_share_same_connection(tmp_path: Path) -> None:
    db_path = str(tmp_path / "shared.db")
    citation_store = SQLiteCitationStore(db_path=db_path)
    session_store = SQLiteSessionStorage(db_path=db_path)

    await citation_store.connect()
    await session_store._ensure_connection()

    assert citation_store._connection is not None
    assert session_store._conn is not None
    assert citation_store._connection is session_store._conn
    assert get_sqlite_pool().get_connection_count() == 1

    await citation_store.disconnect()
    assert get_sqlite_pool().get_connection_count() == 1

    await session_store.close()
    assert get_sqlite_pool().get_connection_count() == 0


@pytest.mark.asyncio
async def test_mongo_pool_reuses_client_for_same_uri() -> None:
    mongo_uri = "mongodb://localhost:27017"
    client1 = await get_shared_mongo_client(mongo_uri)
    client2 = await get_shared_mongo_client(mongo_uri)

    assert client1 is client2
    assert get_mongo_pool().get_client_count() == 1

    await release_shared_mongo_client(mongo_uri)
    assert get_mongo_pool().get_client_count() == 1

    await release_shared_mongo_client(mongo_uri)
    assert get_mongo_pool().get_client_count() == 0
