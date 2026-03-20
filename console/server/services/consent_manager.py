"""Shared consent waiter and store for Console tool authorization."""

from agiwo.agent.tool_auth.state import ConsentWaiter
from agiwo.tool.authz import ConsentStore, InMemoryConsentStore


class ConsentManager:
    """Singleton manager for consent waiter and store across Console."""

    def __init__(self) -> None:
        self._waiter = ConsentWaiter(default_timeout=300.0)
        self._store: ConsentStore = InMemoryConsentStore()

    @property
    def waiter(self) -> ConsentWaiter:
        return self._waiter

    @property
    def store(self) -> ConsentStore:
        return self._store


_consent_manager: ConsentManager | None = None


def get_consent_manager() -> ConsentManager:
    global _consent_manager
    if _consent_manager is None:
        _consent_manager = ConsentManager()
    return _consent_manager


__all__ = ["ConsentManager", "get_consent_manager"]
