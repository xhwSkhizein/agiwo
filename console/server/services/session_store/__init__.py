"""Console session store implementations and factory."""

from server.services.session_store.base import SessionStore
from server.services.session_store.memory import InMemorySessionStore
from server.services.session_store.sqlite import SqliteSessionStore


def create_session_store(
    *,
    db_path: str,
    use_persistent_store: bool,
) -> SessionStore:
    """Create a session store instance.

    Args:
        db_path: Path to SQLite database (used when use_persistent_store=True)
        use_persistent_store: If True, use SQLite; otherwise use in-memory store
    """
    if use_persistent_store:
        return SqliteSessionStore(db_path=db_path)
    return InMemorySessionStore()


__all__ = [
    "SessionStore",
    "InMemorySessionStore",
    "SqliteSessionStore",
    "create_session_store",
]
