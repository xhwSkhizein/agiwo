"""Dispatch outbox records for reliable root Run execution."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from agiwo.objective.errors import ValidationError
from agiwo.objective.models import DispatchStatus, RunRole, new_id, utc_now
from agiwo.utils.serialization import (
    parse_optional_datetime,
    parse_optional_datetime_or_default,
    serialize_optional_datetime,
)


@dataclass(frozen=True, slots=True)
class DispatchRequested:
    """Transactional outbox record. Not part of ObjectiveStatus projection."""

    dispatch_id: str
    objective_id: str
    run_id: str
    role: RunRole
    status: DispatchStatus = DispatchStatus.PENDING
    attempt: int = 0
    created_at: datetime = field(default_factory=utc_now)
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    last_error: str | None = None
    # Serialized system UserMessage for the Run Input snapshot.
    run_input: dict | None = None
    template_hash: str | None = None
    session_id: str | None = None
    state_id: str | None = None

    def __post_init__(self) -> None:
        if not self.dispatch_id:
            raise ValidationError("dispatch_id is required")
        if not self.objective_id:
            raise ValidationError("objective_id is required")
        if not self.run_id:
            raise ValidationError("run_id is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dispatch_id": self.dispatch_id,
            "objective_id": self.objective_id,
            "run_id": self.run_id,
            "role": self.role.value,
            "status": self.status.value,
            "attempt": self.attempt,
            "created_at": self.created_at.isoformat(),
            "lease_owner": self.lease_owner,
            "lease_expires_at": serialize_optional_datetime(self.lease_expires_at),
            "last_error": self.last_error,
            "run_input": self.run_input,
            "template_hash": self.template_hash,
            "session_id": self.session_id,
            "state_id": self.state_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DispatchRequested":
        run_input = data.get("run_input")
        if run_input is None:
            run_input = data.get("assignment_input")
        return cls(
            dispatch_id=data["dispatch_id"],
            objective_id=data["objective_id"],
            run_id=data["run_id"],
            role=RunRole(data.get("role") or data.get("kind")),
            status=DispatchStatus(data.get("status", DispatchStatus.PENDING.value)),
            attempt=int(data.get("attempt", 0)),
            created_at=parse_optional_datetime_or_default(
                data.get("created_at"), utc_now()
            ),
            lease_owner=data.get("lease_owner"),
            lease_expires_at=parse_optional_datetime(data.get("lease_expires_at")),
            last_error=data.get("last_error"),
            run_input=run_input,
            template_hash=data.get("template_hash"),
            session_id=data.get("session_id"),
            state_id=data.get("state_id"),
        )

    @classmethod
    def create(
        cls,
        *,
        objective_id: str,
        run_id: str,
        role: RunRole,
        dispatch_id: str | None = None,
        run_input: dict | None = None,
        template_hash: str | None = None,
        session_id: str | None = None,
        state_id: str | None = None,
    ) -> "DispatchRequested":
        return cls(
            dispatch_id=dispatch_id or new_id("dsp_"),
            objective_id=objective_id,
            run_id=run_id,
            role=role,
            run_input=run_input,
            template_hash=template_hash,
            session_id=session_id,
            state_id=state_id,
        )

    def with_updates(self, **kwargs: Any) -> "DispatchRequested":
        payload = self.to_dict()
        for key, value in kwargs.items():
            if key in {"status"} and hasattr(value, "value"):
                payload[key] = value.value
            elif key in {"role"} and hasattr(value, "value"):
                payload[key] = value.value
            elif key.endswith("_at") and hasattr(value, "isoformat"):
                payload[key] = value.isoformat()
            else:
                payload[key] = value
        return DispatchRequested.from_dict(payload)


__all__ = ["DispatchRequested"]
