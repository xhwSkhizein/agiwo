"""
Global PermissionManager factory for dependency injection.

This module provides a singleton PermissionManager instance that is shared
across all Agents in the system. It is created lazily on first access.
"""

from agiwo.tool.permission.manager import PermissionManager
from agiwo.tool.permission import (
    ConsentWaiter,
    InMemoryConsentStore,
    PermissionService,
)
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

_permission_manager: PermissionManager | None = None


def get_permission_manager() -> PermissionManager:
    """
    Get global PermissionManager singleton.

    The PermissionManager is shared across all Agents in the system.

    Returns:
        PermissionManager: Global singleton instance
    """
    global _permission_manager

    if _permission_manager is None:

        logger.info("initializing_global_permission_manager")

        _permission_manager = PermissionManager(
            consent_store=InMemoryConsentStore(),
            consent_waiter=ConsentWaiter(default_timeout=300.0),
            permission_service=PermissionService(),
            cache_ttl=300,
            cache_size=1000,
        )

    return _permission_manager


def reset_permission_manager() -> None:
    """
    Reset global PermissionManager singleton.

    This is primarily used for testing to ensure a clean state.
    """
    global _permission_manager
    _permission_manager = None
    logger.debug("permission_manager_reset")


__all__ = ["get_permission_manager", "reset_permission_manager"]
