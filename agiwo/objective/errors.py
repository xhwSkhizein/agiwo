"""Typed domain errors for the Objective package."""

from datetime import datetime
from typing import Any


class ObjectiveError(Exception):
    """Base typed error. Control flow uses ``code`` and fields, not message text."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.details: dict[str, Any] = details or {}
        super().__init__(f"{code}: {message}")


class ValidationError(ObjectiveError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("validation_error", message, details=details)


class InvalidStateTransition(ObjectiveError):
    def __init__(
        self,
        *,
        entity: str,
        from_status: str,
        to_status: str,
        **details: Any,
    ) -> None:
        super().__init__(
            "invalid_state_transition",
            f"{entity} cannot transition from {from_status!r} to {to_status!r}",
            details={
                "entity": entity,
                "from_status": from_status,
                "to_status": to_status,
                **details,
            },
        )


class InvariantViolation(ObjectiveError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("invariant_violation", message, details=details)


class IdempotencyConflict(ObjectiveError):
    def __init__(
        self,
        *,
        scope: str,
        idempotency_key: str,
        existing_hash: str,
        request_hash: str,
    ) -> None:
        super().__init__(
            "idempotency_conflict",
            (
                "idempotency key reused with a different request hash "
                f"(scope={scope!r}, key={idempotency_key!r})"
            ),
            details={
                "scope": scope,
                "idempotency_key": idempotency_key,
                "existing_hash": existing_hash,
                "request_hash": request_hash,
            },
        )


class CommandUnavailable(ObjectiveError):
    def __init__(self, command: str, reason: str = "not_wired") -> None:
        super().__init__(
            "command_unavailable",
            f"command {command!r} is not available: {reason}",
            details={"command": command, "reason": reason},
        )


class ProjectionError(ObjectiveError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("projection_error", message, details=details)


class StoreError(ObjectiveError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("store_error", message, details=details)


class UnsupportedStorageBackend(ObjectiveError):
    def __init__(self, storage_type: str) -> None:
        super().__init__(
            "unsupported_storage_backend",
            (
                f"ObjectiveStore does not support storage_type={storage_type!r}; "
                "MVP only supports memory and sqlite "
                "(fail-closed, no silent fallback)"
            ),
            details={"storage_type": storage_type},
        )


class BudgetBoundaryHit(ObjectiveError):
    """Quota insufficient for the pending Decision follow-on action.

    P3-01 raises this instead of creating the next Assignment/outbox.
    P3-05 consumes it to enter DRAINING / BUDGET_PAUSED.
    """

    def __init__(
        self,
        *,
        dimension: str,
        checked_at: datetime,
        used: float,
        limit: float,
        pending_action: str,
        required: float = 1.0,
        **extra: Any,
    ) -> None:
        super().__init__(
            "budget_boundary_hit",
            (
                f"ObjectiveBudget.{dimension} insufficient for {pending_action} "
                f"(used={used}, limit={limit}, required={required})"
            ),
            details={
                "dimension": dimension,
                "checked_at": checked_at.isoformat(),
                "used": used,
                "limit": limit,
                "remaining": max(0.0, limit - used),
                "required": required,
                "pending_action": pending_action,
                **extra,
            },
        )
        self.dimension = dimension
        self.checked_at = checked_at
        self.used = used
        self.limit = limit
        self.pending_action = pending_action


__all__ = [
    "BudgetBoundaryHit",
    "CommandUnavailable",
    "IdempotencyConflict",
    "InvalidStateTransition",
    "InvariantViolation",
    "ObjectiveError",
    "ProjectionError",
    "StoreError",
    "UnsupportedStorageBackend",
    "ValidationError",
]
