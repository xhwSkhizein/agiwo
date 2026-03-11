"""
Permission system for tool execution.

This module provides a unified permission management system for controlling
tool execution based on user consent and authorization policies.
"""

from agiwo.tool.permission.store import (
    ConsentDecision,
    ConsentRecord,
    ConsentStore,
    ConsentWaiter,
    InMemoryConsentStore,
    MongoConsentStore,
)
from agiwo.tool.permission.manager import (
    ConsentResult,
    PermissionDecision,
    PermissionManager,
    PermissionService,
    get_permission_manager,
    reset_permission_manager,
)

__all__ = [
    "ConsentDecision",
    "ConsentRecord",
    "ConsentStore",
    "ConsentWaiter",
    "InMemoryConsentStore",
    "MongoConsentStore",
    "ConsentResult",
    "PermissionDecision",
    "PermissionManager",
    "PermissionService",
    "get_permission_manager",
    "reset_permission_manager",
]
