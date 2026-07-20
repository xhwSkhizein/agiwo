"""Objective domain models: pure types, value objects, and status machines.

These types do not import Scheduler internals or Console session models.
Session identity is an opaque ``session_id`` string.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4
from agiwo.agent.models.input import UserMessage
from agiwo.agent.models.run import RUN_TERMINAL_STATUSES, RunStatus
from agiwo.objective.errors import (
    InvalidStateTransition,
    InvariantViolation,
    ValidationError,
)
from agiwo.utils.serialization import parse_optional_datetime_or_default


def new_id(prefix: str = "") -> str:
    """Allocate a stable opaque id. Prefix is optional for readability."""
    token = uuid4().hex
    return f"{prefix}{token}" if prefix else token


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ObjectiveStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    DRAINING = "DRAINING"
    WAITING_USER = "WAITING_USER"
    BUDGET_PAUSED = "BUDGET_PAUSED"
    USER_PAUSED = "USER_PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RunRole(str, Enum):
    WORK = "work"
    VERIFICATION = "verification"


class HandoffTarget(str, Enum):
    AGENT = "agent"
    VERIFIER = "verifier"
    USER = "user"


class DispatchStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    DISPATCHED = "dispatched"
    COMPLETED = "completed"
    FAILED = "failed"


class CommandReceiptStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


OBJECTIVE_TERMINAL: frozenset[ObjectiveStatus] = frozenset(
    {ObjectiveStatus.COMPLETED, ObjectiveStatus.FAILED}
)
OBJECTIVE_PAUSED: frozenset[ObjectiveStatus] = frozenset(
    {
        ObjectiveStatus.WAITING_USER,
        ObjectiveStatus.BUDGET_PAUSED,
        ObjectiveStatus.USER_PAUSED,
    }
)
OBJECTIVE_ACTIVE_NON_TERMINAL: frozenset[ObjectiveStatus] = frozenset(
    (s for s in ObjectiveStatus if s not in OBJECTIVE_TERMINAL)
)
OBJECTIVE_TRANSITIONS: dict[ObjectiveStatus, frozenset[ObjectiveStatus]] = {
    ObjectiveStatus.CREATED: frozenset(
        {
            ObjectiveStatus.RUNNING,
            ObjectiveStatus.WAITING_USER,
            ObjectiveStatus.DRAINING,
            ObjectiveStatus.USER_PAUSED,
            ObjectiveStatus.BUDGET_PAUSED,
            ObjectiveStatus.FAILED,
        }
    ),
    ObjectiveStatus.RUNNING: frozenset(
        {
            ObjectiveStatus.DRAINING,
            ObjectiveStatus.WAITING_USER,
            ObjectiveStatus.BUDGET_PAUSED,
            ObjectiveStatus.USER_PAUSED,
            ObjectiveStatus.COMPLETED,
            ObjectiveStatus.FAILED,
        }
    ),
    ObjectiveStatus.DRAINING: frozenset(
        {
            ObjectiveStatus.BUDGET_PAUSED,
            ObjectiveStatus.USER_PAUSED,
            ObjectiveStatus.WAITING_USER,
            ObjectiveStatus.RUNNING,
            ObjectiveStatus.COMPLETED,
            ObjectiveStatus.FAILED,
        }
    ),
    ObjectiveStatus.WAITING_USER: frozenset(
        {
            ObjectiveStatus.RUNNING,
            ObjectiveStatus.DRAINING,
            ObjectiveStatus.USER_PAUSED,
            ObjectiveStatus.FAILED,
        }
    ),
    ObjectiveStatus.BUDGET_PAUSED: frozenset(
        {ObjectiveStatus.RUNNING, ObjectiveStatus.DRAINING, ObjectiveStatus.FAILED}
    ),
    ObjectiveStatus.USER_PAUSED: frozenset(
        {ObjectiveStatus.RUNNING, ObjectiveStatus.DRAINING, ObjectiveStatus.FAILED}
    ),
    ObjectiveStatus.COMPLETED: frozenset(),
    ObjectiveStatus.FAILED: frozenset(),
}


def validate_objective_transition(
    from_status: ObjectiveStatus,
    to_status: ObjectiveStatus,
    *,
    objective_id: str | None = None,
) -> None:
    if from_status == to_status:
        return
    allowed = OBJECTIVE_TRANSITIONS.get(from_status, frozenset())
    if to_status not in allowed:
        raise InvalidStateTransition(
            entity="Objective",
            from_status=from_status.value,
            to_status=to_status.value,
            objective_id=objective_id,
        )
    if from_status in OBJECTIVE_TERMINAL:
        raise InvalidStateTransition(
            entity="Objective",
            from_status=from_status.value,
            to_status=to_status.value,
            objective_id=objective_id,
            reason="terminal_reopen_forbidden",
        )


def is_objective_resumable(status: ObjectiveStatus) -> bool:
    return status in {
        ObjectiveStatus.WAITING_USER,
        ObjectiveStatus.BUDGET_PAUSED,
        ObjectiveStatus.USER_PAUSED,
    }


def is_root_run_resumable(status: RunStatus | None) -> bool:
    return status is RunStatus.PAUSED


@dataclass(frozen=True, slots=True)
class BudgetDimension:
    """Single budget axis: only limit/used, no reservation."""

    limit: float
    used: float = 0.0

    def __post_init__(self) -> None:
        if self.limit <= 0:
            raise ValidationError(
                "budget dimension limit must be finite and > 0", limit=self.limit
            )
        if self.used < 0:
            raise ValidationError("budget dimension used must be >= 0", used=self.used)

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self.used)

    def with_used(self, used: float) -> "BudgetDimension":
        return BudgetDimension(limit=self.limit, used=used)

    def with_limit(self, limit: float) -> "BudgetDimension":
        return BudgetDimension(limit=limit, used=self.used)

    def to_dict(self) -> dict[str, float]:
        return {"limit": self.limit, "used": self.used}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BudgetDimension":
        return cls(limit=float(data["limit"]), used=float(data.get("used", 0)))


@dataclass(frozen=True, slots=True)
class ObjectiveBudget:
    """Four hard dimensions. Read-only view; mutations are new facts."""

    handoffs: BudgetDimension
    verification_attempts: BudgetDimension
    llm_cost_usd: BudgetDimension
    active_seconds: BudgetDimension

    @classmethod
    def create(
        cls,
        *,
        handoffs: float,
        verification_attempts: float,
        llm_cost_usd: float,
        active_seconds: float,
    ) -> "ObjectiveBudget":
        return cls(
            handoffs=BudgetDimension(limit=handoffs),
            verification_attempts=BudgetDimension(limit=verification_attempts),
            llm_cost_usd=BudgetDimension(limit=llm_cost_usd),
            active_seconds=BudgetDimension(limit=active_seconds),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "handoffs": self.handoffs.to_dict(),
            "verification_attempts": self.verification_attempts.to_dict(),
            "llm_cost_usd": self.llm_cost_usd.to_dict(),
            "active_seconds": self.active_seconds.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectiveBudget":
        return cls(
            handoffs=BudgetDimension.from_dict(data["handoffs"]),
            verification_attempts=BudgetDimension.from_dict(
                data["verification_attempts"]
            ),
            llm_cost_usd=BudgetDimension.from_dict(data["llm_cost_usd"]),
            active_seconds=BudgetDimension.from_dict(data["active_seconds"]),
        )


ARTIFACT_INLINE_CONTENT_MAX_BYTES = 4096


@dataclass(frozen=True, slots=True)
class Artifact:
    """File index under sessions/<session_id>/artifacts/ (ADR 0045)."""

    artifact_id: str
    path: str
    summary: str
    inline_content: str | None = None
    source_input_id: str | None = None
    content_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValidationError("artifact_id is required")
        if not self.path:
            raise ValidationError("artifact path is required")
        if "/artifacts/" not in self.path and (not self.path.startswith("artifacts/")):
            if "artifacts" not in self.path:
                raise ValidationError(
                    "artifact path must reference the session artifacts directory",
                    path=self.path,
                )
        if self.inline_content is not None:
            size = len(self.inline_content.encode("utf-8"))
            if size > ARTIFACT_INLINE_CONTENT_MAX_BYTES:
                raise ValidationError(
                    "artifact inline_content exceeds size threshold; store path only",
                    size=size,
                    max_bytes=ARTIFACT_INLINE_CONTENT_MAX_BYTES,
                )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "path": self.path,
            "summary": self.summary,
        }
        if self.inline_content is not None:
            payload["inline_content"] = self.inline_content
        if self.source_input_id is not None:
            payload["source_input_id"] = self.source_input_id
        if self.content_hash is not None:
            payload["content_hash"] = self.content_hash
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Artifact":
        return cls(
            artifact_id=data["artifact_id"],
            path=data["path"],
            summary=data.get("summary") or "",
            inline_content=data.get("inline_content"),
            source_input_id=data.get("source_input_id"),
            content_hash=data.get("content_hash"),
        )


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Reference to a registered file Artifact (no routing fields)."""

    artifact_id: str | None = None
    path: str | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id and (not self.path):
            raise ValidationError("ArtifactRef requires artifact_id or path")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.artifact_id is not None:
            payload["artifact_id"] = self.artifact_id
        if self.path is not None:
            payload["path"] = self.path
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRef":
        return cls(artifact_id=data.get("artifact_id"), path=data.get("path"))


@dataclass(frozen=True, slots=True)
class HandoffDecision:
    """Closed handoff target. No named agent/config/pattern/executor fields."""

    target: HandoffTarget
    expects_reply: bool | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.target == HandoffTarget.USER:
            if self.expects_reply is None:
                raise ValidationError(
                    "HandoffDecision target=user requires explicit expects_reply"
                )
        elif self.expects_reply is not None:
            raise ValidationError(
                "expects_reply is only allowed when target=user",
                target=self.target.value,
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"target": self.target.value}
        if self.expects_reply is not None:
            payload["expects_reply"] = self.expects_reply
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HandoffDecision":
        return cls(
            target=HandoffTarget(data["target"]),
            expects_reply=data.get("expects_reply"),
            reason=data.get("reason"),
        )


@dataclass(frozen=True, slots=True)
class ObjectiveUserInput:
    """Authoritative user-side fact. Full UserMessage, never rewritten."""

    input_id: str
    message: UserMessage
    in_reply_to_message_id: str | None = None
    related_outcome_id: str | None = None
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.input_id:
            raise ValidationError("input_id is required")
        if not self.message.is_user_provided:
            raise ValidationError(
                "ObjectiveUserInput requires is_user_provided=true UserMessage",
                input_id=self.input_id,
            )
        if not self.message.has_content():
            raise ValidationError(
                "ObjectiveUserInput requires non-empty user content",
                input_id=self.input_id,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_id": self.input_id,
            "message": self.message.to_dict(),
            "in_reply_to_message_id": self.in_reply_to_message_id,
            "related_outcome_id": self.related_outcome_id,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectiveUserInput":
        message = UserMessage.from_storage_value(data["message"])
        if not isinstance(message, UserMessage):
            raise ValidationError(
                "ObjectiveUserInput.message must decode to UserMessage",
                input_id=data.get("input_id"),
            )
        return cls(
            input_id=data["input_id"],
            message=message,
            in_reply_to_message_id=data.get("in_reply_to_message_id"),
            related_outcome_id=data.get("related_outcome_id"),
            created_at=parse_optional_datetime_or_default(
                data.get("created_at"), utc_now()
            ),
        )


@dataclass(frozen=True, slots=True)
class ContributionAnnotation:
    """Append-only note on an existing contribution. from/time filled by system."""

    contribution_id: str
    annotation: str
    from_run_id: str
    time: datetime
    deactivate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "contribution_id": self.contribution_id,
            "annotation": self.annotation,
            "from": self.from_run_id,
            "time": self.time.isoformat(),
            "deactivate": self.deactivate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContributionAnnotation":
        return cls(
            contribution_id=data["contribution_id"],
            annotation=data["annotation"],
            from_run_id=data.get("from") or data.get("from_run_id") or "",
            time=parse_optional_datetime_or_default(data.get("time"), utc_now()),
            deactivate=bool(data.get("deactivate", False)),
        )


@dataclass(frozen=True, slots=True)
class ObjectiveContribution:
    """Agent-derived discovery. content is immutable after creation."""

    contribution_id: str
    content: str
    summary: str | None = None
    annotations: tuple[ContributionAnnotation, ...] = ()
    active: bool = True

    def __post_init__(self) -> None:
        if not self.contribution_id:
            raise ValidationError("contribution_id is required")
        if not self.content:
            raise ValidationError("contribution content is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contribution_id": self.contribution_id,
            "content": self.content,
            "summary": self.summary,
            "annotations": [a.to_dict() for a in self.annotations],
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectiveContribution":
        annotations = tuple(
            (
                ContributionAnnotation.from_dict(item)
                for item in data.get("annotations") or []
            )
        )
        return cls(
            contribution_id=data["contribution_id"],
            content=data["content"],
            summary=data.get("summary"),
            annotations=annotations,
            active=bool(data.get("active", True)),
        )


@dataclass(frozen=True, slots=True)
class NewContributionSpec:
    """Wire shape for creating a contribution (system assigns id)."""

    content: str
    summary: str | None = None

    def __post_init__(self) -> None:
        if not self.content:
            raise ValidationError("new contribution content is required")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"content": self.content}
        if self.summary is not None:
            payload["summary"] = self.summary
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NewContributionSpec":
        return cls(content=data["content"], summary=data.get("summary"))


@dataclass(frozen=True, slots=True)
class ContributionAnnotationSpec:
    """Wire shape for annotating an existing contribution."""

    contribution_id: str
    annotation: str
    deactivate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "contribution_id": self.contribution_id,
            "annotation": self.annotation,
            "deactivate": self.deactivate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContributionAnnotationSpec":
        return cls(
            contribution_id=data["contribution_id"],
            annotation=data["annotation"],
            deactivate=bool(data.get("deactivate", False)),
        )


SourceKind = Literal[
    "user_input", "contribution", "artifact", "outcome", "objective_fact"
]


@dataclass(frozen=True, slots=True)
class GoalSourceRef:
    kind: SourceKind
    id: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "id": self.id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoalSourceRef":
        return cls(kind=data["kind"], id=data["id"])


@dataclass(frozen=True, slots=True)
class CurrentGoalAnalysis:
    """Mutable analysis projection. Never covers user facts or plan/budget."""

    revision: int
    intent: str | None = None
    scope: str | None = None
    success_criteria: str | None = None
    assumptions: tuple[str, ...] = ()
    sources: tuple[GoalSourceRef, ...] = ()

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValidationError("current_goal revision must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "intent": self.intent,
            "scope": self.scope,
            "success_criteria": self.success_criteria,
            "assumptions": list(self.assumptions),
            "sources": [s.to_dict() for s in self.sources],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurrentGoalAnalysis":
        forbidden = {"run_plan", "plan", "budget", "handoff", "status", "milestones"}
        extra = forbidden.intersection(data)
        if extra:
            raise ValidationError(
                "current_goal analysis cannot carry plan/budget/status fields",
                forbidden_fields=sorted(extra),
            )
        return cls(
            revision=int(data.get("revision", 0)),
            intent=data.get("intent"),
            scope=data.get("scope"),
            success_criteria=data.get("success_criteria"),
            assumptions=tuple(data.get("assumptions") or ()),
            sources=tuple(
                (GoalSourceRef.from_dict(item) for item in data.get("sources") or [])
            ),
        )


@dataclass(frozen=True, slots=True)
class ObjectiveUpdateSpec:
    """Wire shape for assignment finalization objective_update."""

    expected_revision: int
    intent: str | None = None
    scope: str | None = None
    success_criteria: str | None = None
    assumptions: tuple[str, ...] | None = None
    sources: tuple[GoalSourceRef, ...] = ()

    def __post_init__(self) -> None:
        if self.expected_revision < 0:
            raise ValidationError("expected_revision must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"expected_revision": self.expected_revision}
        if self.intent is not None:
            payload["intent"] = self.intent
        if self.scope is not None:
            payload["scope"] = self.scope
        if self.success_criteria is not None:
            payload["success_criteria"] = self.success_criteria
        if self.assumptions is not None:
            payload["assumptions"] = list(self.assumptions)
        if self.sources:
            payload["sources"] = [s.to_dict() for s in self.sources]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ObjectiveUpdateSpec | None":
        if data is None:
            return None
        forbidden = {"run_plan", "plan", "budget", "handoff", "status", "milestones"}
        extra = forbidden.intersection(data)
        if extra:
            raise ValidationError(
                "objective_update cannot carry plan/budget/status fields",
                forbidden_fields=sorted(extra),
            )
        assumptions_raw = data.get("assumptions")
        return cls(
            expected_revision=int(data["expected_revision"]),
            intent=data.get("intent"),
            scope=data.get("scope"),
            success_criteria=data.get("success_criteria"),
            assumptions=None if assumptions_raw is None else tuple(assumptions_raw),
            sources=tuple(
                (GoalSourceRef.from_dict(item) for item in data.get("sources") or [])
            ),
        )


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Unique terminal record for one root Run (ADR 0046)."""

    run_id: str
    role: RunRole
    terminal_status: RunStatus
    reason: str
    report: str
    outcome_id: str = field(default_factory=lambda: new_id("out_"))
    artifact_refs: tuple[ArtifactRef, ...] = ()
    new_contributions: tuple[NewContributionSpec, ...] = ()
    contribution_annotations: tuple[ContributionAnnotationSpec, ...] = ()
    decision: HandoffDecision | None = None
    objective_update: ObjectiveUpdateSpec | None = None
    carry_forward: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValidationError("RunOutcome.run_id is required")
        if self.terminal_status not in RUN_TERMINAL_STATUSES:
            raise ValidationError(
                "RunOutcome.terminal_status must be a terminal RunStatus",
                terminal_status=self.terminal_status.value,
            )
        if not self.report and self.terminal_status == RunStatus.COMPLETED:
            raise ValidationError("COMPLETED RunOutcome requires a plain-text report")

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome_id": self.outcome_id,
            "run_id": self.run_id,
            "role": self.role.value,
            "terminal_status": self.terminal_status.value,
            "reason": self.reason,
            "report": self.report,
            "artifact_refs": [r.to_dict() for r in self.artifact_refs],
            "new_contributions": [c.to_dict() for c in self.new_contributions],
            "contribution_annotations": [
                a.to_dict() for a in self.contribution_annotations
            ],
            "decision": self.decision.to_dict() if self.decision else None,
            "objective_update": self.objective_update.to_dict()
            if self.objective_update
            else None,
            "carry_forward": list(self.carry_forward),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunOutcome":
        decision_raw = data.get("decision")
        update_raw = data.get("objective_update")
        return cls(
            outcome_id=data.get("outcome_id") or new_id("out_"),
            run_id=data["run_id"],
            role=RunRole(data["role"]),
            terminal_status=RunStatus(data["terminal_status"]),
            reason=data.get("reason") or "",
            report=data.get("report") or "",
            artifact_refs=tuple(
                (
                    ArtifactRef.from_dict(item)
                    for item in data.get("artifact_refs") or []
                )
            ),
            new_contributions=tuple(
                (
                    NewContributionSpec.from_dict(item)
                    for item in data.get("new_contributions") or []
                )
            ),
            contribution_annotations=tuple(
                (
                    ContributionAnnotationSpec.from_dict(item)
                    for item in data.get("contribution_annotations") or []
                )
            ),
            decision=HandoffDecision.from_dict(decision_raw) if decision_raw else None,
            objective_update=ObjectiveUpdateSpec.from_dict(update_raw),
            carry_forward=tuple(data.get("carry_forward") or ()),
            provenance=dict(data.get("provenance") or {}),
        )


def assert_single_active_objective(
    *, session_id: str, active_objective_ids: list[str]
) -> None:
    if len(active_objective_ids) > 1:
        raise InvariantViolation(
            "a Session may have at most one non-terminal Objective",
            session_id=session_id,
            active_objective_ids=active_objective_ids,
        )


def assert_single_active_root_run(
    *, objective_id: str, active_run_ids: list[str]
) -> None:
    if len(active_run_ids) > 1:
        raise InvariantViolation(
            "an Objective may have at most one non-terminal root Run",
            objective_id=objective_id,
            active_run_ids=active_run_ids,
        )


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    """Create-time finite budget limits (used starts at 0)."""

    handoffs: float
    verification_attempts: float
    llm_cost_usd: float
    active_seconds: float

    def to_budget(self) -> ObjectiveBudget:
        return ObjectiveBudget.create(
            handoffs=self.handoffs,
            verification_attempts=self.verification_attempts,
            llm_cost_usd=self.llm_cost_usd,
            active_seconds=self.active_seconds,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "handoffs": self.handoffs,
            "verification_attempts": self.verification_attempts,
            "llm_cost_usd": self.llm_cost_usd,
            "active_seconds": self.active_seconds,
        }


@dataclass(frozen=True, slots=True)
class CreateObjectiveRequest:
    session_id: str
    user_message: UserMessage
    budget: BudgetLimits
    idempotency_key: str
    objective_id: str | None = None
    in_reply_to_message_id: str | None = None
    related_outcome_id: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValidationError("session_id is required")
        if not self.idempotency_key:
            raise ValidationError("idempotency_key is required")
        if not self.user_message.is_user_provided:
            raise ValidationError(
                "create objective requires is_user_provided=true user message"
            )


@dataclass(frozen=True, slots=True)
class SubmitUserInputRequest:
    objective_id: str
    user_message: UserMessage
    idempotency_key: str
    in_reply_to_message_id: str | None = None
    related_outcome_id: str | None = None

    def __post_init__(self) -> None:
        if not self.objective_id:
            raise ValidationError("objective_id is required")
        if not self.idempotency_key:
            raise ValidationError("idempotency_key is required")
        if not self.user_message.is_user_provided:
            raise ValidationError(
                "submit user input requires is_user_provided=true user message"
            )


@dataclass(frozen=True, slots=True)
class ExternalizeUserInputRequest:
    objective_id: str
    input_id: str
    summary: str
    idempotency_key: str
    content_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.objective_id:
            raise ValidationError("objective_id is required")
        if not self.input_id:
            raise ValidationError("input_id is required")
        if not self.idempotency_key:
            raise ValidationError("idempotency_key is required")
        if not self.summary:
            raise ValidationError("externalize requires a summary for model context")


@dataclass(frozen=True, slots=True)
class PauseObjectiveRequest:
    objective_id: str
    idempotency_key: str
    reason: str = "user_pause"

    def __post_init__(self) -> None:
        if not self.objective_id:
            raise ValidationError("objective_id is required")
        if not self.idempotency_key:
            raise ValidationError("idempotency_key is required")


@dataclass(frozen=True, slots=True)
class ResumeObjectiveRequest:
    objective_id: str
    idempotency_key: str
    reason: str = "user_resume"

    def __post_init__(self) -> None:
        if not self.objective_id:
            raise ValidationError("objective_id is required")
        if not self.idempotency_key:
            raise ValidationError("idempotency_key is required")


@dataclass(frozen=True, slots=True)
class AdjustBudgetRequest:
    objective_id: str
    idempotency_key: str
    handoffs: float | None = None
    verification_attempts: float | None = None
    llm_cost_usd: float | None = None
    active_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.objective_id:
            raise ValidationError("objective_id is required")
        if not self.idempotency_key:
            raise ValidationError("idempotency_key is required")
        if all(
            (
                v is None
                for v in (
                    self.handoffs,
                    self.verification_attempts,
                    self.llm_cost_usd,
                    self.active_seconds,
                )
            )
        ):
            raise ValidationError("adjust budget requires at least one dimension")


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Stable command response returned by ObjectiveService."""

    objective_id: str
    status: str
    replayed: bool = False
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "status": self.status,
            "replayed": self.replayed,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommandResult":
        return cls(
            objective_id=data["objective_id"],
            status=data["status"],
            replayed=bool(data.get("replayed", False)),
            payload=dict(data.get("payload") or {}),
        )


__all__ = [
    "ARTIFACT_INLINE_CONTENT_MAX_BYTES",
    "AdjustBudgetRequest",
    "Artifact",
    "ArtifactRef",
    "BudgetDimension",
    "BudgetLimits",
    "CommandReceiptStatus",
    "CommandResult",
    "ContributionAnnotation",
    "ContributionAnnotationSpec",
    "CreateObjectiveRequest",
    "CurrentGoalAnalysis",
    "DispatchStatus",
    "ExternalizeUserInputRequest",
    "GoalSourceRef",
    "HandoffDecision",
    "HandoffTarget",
    "NewContributionSpec",
    "OBJECTIVE_ACTIVE_NON_TERMINAL",
    "OBJECTIVE_PAUSED",
    "OBJECTIVE_TERMINAL",
    "OBJECTIVE_TRANSITIONS",
    "ObjectiveBudget",
    "ObjectiveContribution",
    "ObjectiveStatus",
    "ObjectiveUpdateSpec",
    "ObjectiveUserInput",
    "PauseObjectiveRequest",
    "ResumeObjectiveRequest",
    "RunOutcome",
    "RunRole",
    "SourceKind",
    "SubmitUserInputRequest",
    "assert_single_active_objective",
    "assert_single_active_root_run",
    "is_objective_resumable",
    "is_root_run_resumable",
    "new_id",
    "utc_now",
    "validate_objective_transition",
]
