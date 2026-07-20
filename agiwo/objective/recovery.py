"""Startup reconciliation for Objective outbox / RunLog / pause state."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from agiwo.agent import RunStatus
from agiwo.objective.log import ObjectiveFactKind
from agiwo.objective.models import CommandReceiptStatus, DispatchStatus, utc_now
from agiwo.objective.outbox import DispatchRequested
from agiwo.objective.projection import project_objective
from agiwo.objective.store.base import CommandReceipt, ObjectiveStore, command_scope
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class RecoveryKind(str, Enum):
    REDISPATCH = "redispatch"
    ATTACH_RUNTIME = "attach_runtime"
    WAIT_PAUSED = "wait_paused"
    REWRITE_OUTCOME = "rewrite_outcome"
    COMPLETE_OUTBOX = "complete_outbox"
    RECOVERY_FAULT = "recovery_fault"
    NOOP = "noop"


@dataclass(frozen=True, slots=True)
class RecoveryAction:
    kind: RecoveryKind
    dispatch_id: str
    run_id: str
    objective_id: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    actions: tuple[RecoveryAction, ...] = ()
    faults: tuple[RecoveryAction, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.faults


class RunQuery(Protocol):
    async def get_run_status(self, run_id: str) -> RunStatus | None: ...

    async def get_run_view(self, run_id: str) -> object | None: ...


async def classify_dispatch(
    store: ObjectiveStore,
    record: DispatchRequested,
    runs: RunQuery | None,
) -> RecoveryAction:
    """First-layer matrix: outbox record × RunView (public query only)."""
    facts = await store.list_facts(objective_id=record.objective_id)
    view = project_objective(facts, objective_id=record.objective_id)
    has_outcome = any(
        f.kind == ObjectiveFactKind.RUN_OUTCOME
        and f.payload.get("run_id") == record.run_id
        for f in facts
    )
    execution_started = any(
        f.kind == ObjectiveFactKind.ROOT_RUN_STARTED
        and f.payload.get("run_id") == record.run_id
        for f in facts
    )

    if has_outcome:
        return RecoveryAction(
            kind=RecoveryKind.COMPLETE_OUTBOX,
            dispatch_id=record.dispatch_id,
            run_id=record.run_id,
            objective_id=record.objective_id,
            details={"reason": "outcome_present"},
        )

    run_status: RunStatus | None = None
    if runs is not None:
        run_status = await runs.get_run_status(record.run_id)

    if run_status is None:
        if record.status in {DispatchStatus.PENDING, DispatchStatus.CLAIMED}:
            return RecoveryAction(
                kind=RecoveryKind.REDISPATCH,
                dispatch_id=record.dispatch_id,
                run_id=record.run_id,
                objective_id=record.objective_id,
                details={"reason": "no_run_started"},
            )
        if record.status is DispatchStatus.DISPATCHED and not execution_started:
            return RecoveryAction(
                kind=RecoveryKind.REDISPATCH,
                dispatch_id=record.dispatch_id,
                run_id=record.run_id,
                objective_id=record.objective_id,
                details={"reason": "dispatched_without_execution_started"},
            )
        return RecoveryAction(
            kind=RecoveryKind.RECOVERY_FAULT,
            dispatch_id=record.dispatch_id,
            run_id=record.run_id,
            objective_id=record.objective_id,
            details={"reason": "dispatched_run_missing"},
        )

    if run_status is RunStatus.PAUSED:
        return RecoveryAction(
            kind=RecoveryKind.WAIT_PAUSED,
            dispatch_id=record.dispatch_id,
            run_id=record.run_id,
            objective_id=record.objective_id,
            details={
                "objective_status": view.status.value if view else None,
            },
        )

    if run_status is RunStatus.RUNNING:
        return RecoveryAction(
            kind=RecoveryKind.ATTACH_RUNTIME,
            dispatch_id=record.dispatch_id,
            run_id=record.run_id,
            objective_id=record.objective_id,
            details={"reason": "running_attach_or_orphan"},
        )

    if run_status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.INTERRUPTED}:
        return RecoveryAction(
            kind=RecoveryKind.REWRITE_OUTCOME,
            dispatch_id=record.dispatch_id,
            run_id=record.run_id,
            objective_id=record.objective_id,
            details={"run_status": run_status.value},
        )

    return RecoveryAction(
        kind=RecoveryKind.NOOP,
        dispatch_id=record.dispatch_id,
        run_id=record.run_id,
        objective_id=record.objective_id,
    )


async def list_open_dispatches(
    store: ObjectiveStore,
    *,
    objective_id: str | None = None,
    limit: int = 200,
) -> list[DispatchRequested]:
    """Pending/claimed plus unfinished dispatched records."""
    open_records = await store.list_pending_dispatches(
        objective_id=objective_id, limit=limit
    )
    # DISPATCHED unfinished: walk pending list is not enough; use objective-scoped
    # scan via get_dispatch when store exposes only pending — fall back to pending.
    # Memory/SQLite stores may hold more in outbox; prefer list_pending + optional
    # dispatched via duck-typed helper.
    list_all = getattr(store, "list_dispatches", None)
    if callable(list_all):
        all_records = await list_all(objective_id=objective_id, limit=limit)
        open_records = [
            r
            for r in all_records
            if r.status
            in {
                DispatchStatus.PENDING,
                DispatchStatus.CLAIMED,
                DispatchStatus.DISPATCHED,
            }
        ]
    return open_records


async def reconcile_startup(
    store: ObjectiveStore,
    runs: RunQuery | None = None,
    *,
    apply: bool = True,
    owner: str = "startup_reconciler",
) -> RecoveryReport:
    """Classify open outbox records; optionally apply safe repairs."""
    records = await list_open_dispatches(store)
    actions: list[RecoveryAction] = []
    faults: list[RecoveryAction] = []
    for record in records:
        action = await classify_dispatch(store, record, runs)
        if action.kind is RecoveryKind.RECOVERY_FAULT:
            faults.append(action)
        else:
            actions.append(action)
        if apply:
            await _apply_action(store, record, action, owner=owner)
    if apply and runs is not None:
        from agiwo.objective.llm_budget import reconcile_missing_llm_run_totals  # noqa: PLC0415

        objective_ids = await store.list_objective_ids()
        await reconcile_missing_llm_run_totals(
            store,
            runs,
            objective_ids=objective_ids,
        )
    report = RecoveryReport(actions=tuple(actions), faults=tuple(faults))
    logger.info(
        "objective_startup_reconcile",
        action_count=len(actions),
        fault_count=len(faults),
    )
    return report


async def _apply_action(
    store: ObjectiveStore,
    record: DispatchRequested,
    action: RecoveryAction,
    *,
    owner: str,
) -> None:
    if action.kind is RecoveryKind.COMPLETE_OUTBOX:
        if record.status is not DispatchStatus.COMPLETED:
            try:
                await store.complete_dispatch(
                    dispatch_id=record.dispatch_id,
                    owner=record.lease_owner or owner,
                    status="completed",
                )
            except Exception:  # noqa: BLE001
                # Lease owner mismatch: leave for next claim cycle.
                logger.warning(
                    "objective_reconcile_complete_failed",
                    dispatch_id=record.dispatch_id,
                )
        return

    if action.kind is RecoveryKind.REDISPATCH:
        if record.status is DispatchStatus.CLAIMED:
            try:
                await store.release_dispatch(
                    dispatch_id=record.dispatch_id,
                    owner=record.lease_owner or owner,
                    last_error="startup_redispatch",
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "objective_reconcile_release_failed",
                    dispatch_id=record.dispatch_id,
                )
        elif record.status is DispatchStatus.DISPATCHED:
            # Reset to pending via release semantics is store-specific; record fault
            # receipt so operators can inspect without inventing a new run_id.
            now = utc_now()
            receipt = CommandReceipt(
                scope=command_scope(record.objective_id, "recovery"),
                idempotency_key=f"recovery:{record.dispatch_id}:redispatch",
                request_hash=record.run_id,
                status=CommandReceiptStatus.COMPLETED,
                response_payload={
                    "kind": action.kind.value,
                    "details": action.details,
                },
                created_at=now,
                completed_at=now,
            )
            await store.commit_command(receipt=receipt, facts=[])
        return

    if action.kind is RecoveryKind.RECOVERY_FAULT:
        now = utc_now()
        receipt = CommandReceipt(
            scope=command_scope(record.objective_id, "recovery_fault"),
            idempotency_key=f"recovery_fault:{record.dispatch_id}",
            request_hash=str(action.details),
            status=CommandReceiptStatus.COMPLETED,
            response_payload={
                "kind": action.kind.value,
                "details": action.details,
            },
            created_at=now,
            completed_at=now,
        )
        await store.commit_command(receipt=receipt, facts=[])


__all__ = [
    "RecoveryAction",
    "RecoveryKind",
    "RecoveryReport",
    "classify_dispatch",
    "list_open_dispatches",
    "reconcile_startup",
]
