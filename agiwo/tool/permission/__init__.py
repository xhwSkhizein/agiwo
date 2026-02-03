"""
Permission system for tool execution.

This module provides a unified permission management system for controlling
tool execution based on user consent and authorization policies.
"""

from agiwo.tool.permission.consent_store import (
    ConsentStore,
    InMemoryConsentStore,
    MongoConsentStore,
)
from agiwo.tool.permission.consent_waiter import ConsentDecision, ConsentWaiter
from agiwo.tool.permission.manager import ConsentResult, PermissionManager
from agiwo.tool.permission.service import PermissionDecision, PermissionService

__all__ = [
    "PermissionManager",
    "ConsentResult",
    "ConsentStore",
    "InMemoryConsentStore",
    "MongoConsentStore",
    "ConsentWaiter",
    "ConsentDecision",
    "PermissionService",
    "PermissionDecision",
]
