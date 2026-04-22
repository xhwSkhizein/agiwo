from pathlib import Path

import pytest

from agiwo.agent.storage.sqlite import SQLiteRunLogStorage
from agiwo.tool.storage.citation.sqlite_store import SQLiteCitationStore
from agiwo.utils.sqlite_pool import get_sqlite_pool


@pytest.mark.asyncio
async def test_sqlite_stores_share_same_connection(tmp_path: Path) -> None:
    db_path = str(tmp_path / "shared.db")
    baseline = get_sqlite_pool().get_connection_count()
    citation_store = SQLiteCitationStore(db_path=db_path)
    run_step_store = SQLiteRunLogStorage(db_path=db_path)

    await citation_store.connect()
    await run_step_store.connect()

    assert citation_store._connection is not None
    assert run_step_store._connection is not None
    assert citation_store._connection is run_step_store._connection
    assert get_sqlite_pool().get_connection_count() == baseline + 1

    await citation_store.disconnect()
    assert get_sqlite_pool().get_connection_count() == baseline + 1

    await run_step_store.close()
    assert get_sqlite_pool().get_connection_count() == baseline
