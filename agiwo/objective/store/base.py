"""ObjectiveStore contract: atomic command transaction boundary."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from agiwo.objective.errors import ValidationError
from agiwo.objective.log import ObjectiveLogEntry
from agiwo.objective.models import CommandReceiptStatus, new_id, utc_now
from agiwo.objective.outbox import DispatchRequested
from agiwo.objective.store.notify import CommitNotifier
from agiwo.utils.serialization import (
    parse_datetime,
    parse_optional_datetime,
    parse_optional_datetime_or_default,
    serialize_optional_datetime,
)


@dataclass(frozen=True, slots=True)
class SessionSlot:
    """Session-level unique occupancy of a non-terminal Objective."""

    session_id: str
    objective_id: str
    acquired_at: datetime
    objective_revision: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "objective_id": self.objective_id,
            "acquired_at": self.acquired_at.isoformat(),
            "objective_revision": self.objective_revision,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionSlot":
        return cls(
            session_id=data["session_id"],
            objective_id=data["objective_id"],
            acquired_at=parse_datetime(data["acquired_at"]),
            objective_revision=int(data.get("objective_revision", 0)),
        )


@dataclass(frozen=True, slots=True)
class SlotMutation:
    """Acquire or release a session activity slot inside commit_command."""

    action: Literal["acquire", "release"]
    session_id: str
    objective_id: str
    objective_revision: int = 0
    acquired_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.action not in {"acquire", "release"}:
            raise ValidationError("slot action must be acquire or release")
        if not self.session_id or not self.objective_id:
            raise ValidationError("slot mutation requires session_id and objective_id")


@dataclass(frozen=True, slots=True)
class CommandReceipt:
    """Durable idempotency receipt for Objective write commands."""

    scope: str
    idempotency_key: str
    request_hash: str
    status: CommandReceiptStatus
    response_payload: dict[str, Any]
    created_at: datetime = field(default_factory=utc_now)
    completed_at: datetime | None = None
    receipt_id: str = field(default_factory=lambda: new_id("rcpt_"))

    def __post_init__(self) -> None:
        if not self.scope:
            raise ValidationError("receipt scope is required")
        if not self.idempotency_key:
            raise ValidationError("receipt idempotency_key is required")
        if not self.request_hash:
            raise ValidationError("receipt request_hash is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "scope": self.scope,
            "idempotency_key": self.idempotency_key,
            "request_hash": self.request_hash,
            "status": self.status.value,
            "response_payload": dict(self.response_payload),
            "created_at": self.created_at.isoformat(),
            "completed_at": serialize_optional_datetime(self.completed_at),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommandReceipt":
        return cls(
            receipt_id=data.get("receipt_id") or new_id("rcpt_"),
            scope=data["scope"],
            idempotency_key=data["idempotency_key"],
            request_hash=data["request_hash"],
            status=CommandReceiptStatus(data["status"]),
            response_payload=dict(data.get("response_payload") or {}),
            created_at=parse_optional_datetime_or_default(
                data.get("created_at"), utc_now()
            ),
            completed_at=parse_optional_datetime(data.get("completed_at")),
        )


def create_scope(session_id: str) -> str:
    return f"session:{session_id}:objective:create"


def command_scope(objective_id: str, command_kind: str) -> str:
    return f"objective:{objective_id}:{command_kind}"


class ObjectiveStore(ABC):
    """Narrow persistence boundary for ObjectiveLog + outbox + slot + receipt."""

    def __init__(self) -> None:
        self._commit_notifier = CommitNotifier()

    @property
    def commit_notifier(self) -> CommitNotifier:
        return self._commit_notifier

    async def close(self) -> None:
        """Release resources (optional)."""

    @abstractmethod
    async def commit_command(
        self,
        *,
        receipt: CommandReceipt,
        facts: list[ObjectiveLogEntry],
        slot_mutation: SlotMutation | None = None,
        outbox_records: list[DispatchRequested] | None = None,
    ) -> CommandReceipt:
        """Atomically commit receipt, facts, optional slot and outbox records.

        Same scope/key + same hash returns the first persisted receipt.
        Same scope/key + different hash raises IdempotencyConflict.
        """

    @abstractmethod
    async def get_receipt(
        self, *, scope: str, idempotency_key: str
    ) -> CommandReceipt | None: ...

    @abstractmethod
    async def list_facts(
        self,
        *,
        objective_id: str,
        after_sequence: int | None = None,
        limit: int = 10000,
    ) -> list[ObjectiveLogEntry]: ...

    @abstractmethod
    async def get_max_sequence(self, objective_id: str) -> int: ...

    @abstractmethod
    async def get_session_slot(self, session_id: str) -> SessionSlot | None: ...

    @abstractmethod
    async def list_objective_ids_for_session(self, session_id: str) -> list[str]: ...

    @abstractmethod
    async def list_objective_ids(self) -> list[str]:
        """Return every known objective_id (from ObjectiveLog), sorted."""

    @abstractmethod
    async def claim_dispatch(
        self, *, owner: str, lease_seconds: float = 30.0, now: datetime | None = None
    ) -> DispatchRequested | None: ...

    @abstractmethod
    async def renew_dispatch_lease(
        self,
        *,
        dispatch_id: str,
        owner: str,
        lease_seconds: float = 30.0,
        now: datetime | None = None,
    ) -> DispatchRequested: ...

    @abstractmethod
    async def complete_dispatch(
        self,
        *,
        dispatch_id: str,
        owner: str,
        status: Literal["dispatched", "completed", "failed"] = "dispatched",
        last_error: str | None = None,
        now: datetime | None = None,
    ) -> DispatchRequested: ...

    @abstractmethod
    async def release_dispatch(
        self,
        *,
        dispatch_id: str,
        owner: str,
        last_error: str | None = None,
        now: datetime | None = None,
    ) -> DispatchRequested: ...

    @abstractmethod
    async def get_dispatch(self, dispatch_id: str) -> DispatchRequested | None: ...

    @abstractmethod
    async def list_pending_dispatches(
        self, *, objective_id: str | None = None, limit: int = 100
    ) -> list[DispatchRequested]: ...

    async def list_dispatches(
        self, *, objective_id: str | None = None, limit: int = 100
    ) -> list[DispatchRequested]:
        """Optional: all outbox rows (default falls back to pending/claimed)."""
        return await self.list_pending_dispatches(
            objective_id=objective_id, limit=limit
        )


__all__ = [
    "CommandReceipt",
    "ObjectiveStore",
    "SessionSlot",
    "SlotMutation",
    "command_scope",
    "create_scope",
]
