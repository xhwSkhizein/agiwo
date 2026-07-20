"""In-memory ObjectiveStore with session/objective lock ordering."""

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Literal

from agiwo.objective.errors import IdempotencyConflict, StoreError
from agiwo.objective.log import ObjectiveLogEntry
from agiwo.objective.models import DispatchStatus, utc_now
from agiwo.objective.outbox import DispatchRequested
from agiwo.objective.store.base import (
    CommandReceipt,
    ObjectiveStore,
    SessionSlot,
    SlotMutation,
)
from agiwo.objective.store.commit_rules import validate_commit_invariants


def _aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class InMemoryObjectiveStore(ObjectiveStore):
    def __init__(self) -> None:
        super().__init__()
        self._facts: dict[str, list[ObjectiveLogEntry]] = {}
        self._sequences: dict[str, int] = {}
        self._slots: dict[str, SessionSlot] = {}
        self._receipts: dict[tuple[str, str], CommandReceipt] = {}
        self._outbox: dict[str, DispatchRequested] = {}
        self._session_objective_index: dict[str, set[str]] = {}
        self._global_lock = asyncio.Lock()
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._objective_locks: dict[str, asyncio.Lock] = {}

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]

    def _objective_lock(self, objective_id: str) -> asyncio.Lock:
        if objective_id not in self._objective_locks:
            self._objective_locks[objective_id] = asyncio.Lock()
        return self._objective_locks[objective_id]

    async def commit_command(
        self,
        *,
        receipt: CommandReceipt,
        facts: list[ObjectiveLogEntry],
        slot_mutation: SlotMutation | None = None,
        outbox_records: list[DispatchRequested] | None = None,
    ) -> CommandReceipt:
        session_ids: set[str] = set()
        objective_ids: set[str] = set()
        if slot_mutation is not None:
            session_ids.add(slot_mutation.session_id)
            objective_ids.add(slot_mutation.objective_id)
        for fact in facts:
            objective_ids.add(fact.objective_id)
        for record in outbox_records or []:
            objective_ids.add(record.objective_id)

        # Fixed lock order: global -> session_id sorted -> objective_id sorted
        async with self._global_lock:
            session_locks = [self._session_lock(sid) for sid in sorted(session_ids)]
            objective_locks = [
                self._objective_lock(oid) for oid in sorted(objective_ids)
            ]
            for lock in session_locks:
                await lock.acquire()
            try:
                for lock in objective_locks:
                    await lock.acquire()
                try:
                    return self._commit_unlocked(
                        receipt=receipt,
                        facts=facts,
                        slot_mutation=slot_mutation,
                        outbox_records=outbox_records or [],
                    )
                finally:
                    for lock in reversed(objective_locks):
                        lock.release()
            finally:
                for lock in reversed(session_locks):
                    lock.release()

    def _commit_unlocked(
        self,
        *,
        receipt: CommandReceipt,
        facts: list[ObjectiveLogEntry],
        slot_mutation: SlotMutation | None,
        outbox_records: list[DispatchRequested],
    ) -> CommandReceipt:
        key = (receipt.scope, receipt.idempotency_key)
        existing = self._receipts.get(key)
        if existing is not None:
            if existing.request_hash != receipt.request_hash:
                raise IdempotencyConflict(
                    scope=receipt.scope,
                    idempotency_key=receipt.idempotency_key,
                    existing_hash=existing.request_hash,
                    request_hash=receipt.request_hash,
                )
            return deepcopy(existing)

        # Snapshot for rollback
        facts_snap = {k: list(v) for k, v in self._facts.items()}
        seq_snap = dict(self._sequences)
        slots_snap = dict(self._slots)
        outbox_snap = dict(self._outbox)
        index_snap = {k: set(v) for k, v in self._session_objective_index.items()}

        try:
            validate_commit_invariants(
                existing_facts_by_objective=self._facts,
                facts=facts,
                slot_mutation=slot_mutation,
                outbox_records=outbox_records,
            )

            if slot_mutation is not None:
                self._apply_slot(slot_mutation)

            for fact in facts:
                bucket = self._facts.setdefault(fact.objective_id, [])
                if any(e.fact_id == fact.fact_id for e in bucket):
                    raise StoreError(
                        "duplicate fact_id",
                        fact_id=fact.fact_id,
                        objective_id=fact.objective_id,
                    )
                if any(e.sequence == fact.sequence for e in bucket):
                    raise StoreError(
                        "duplicate sequence",
                        sequence=fact.sequence,
                        objective_id=fact.objective_id,
                    )
                current = self._sequences.get(fact.objective_id, 0)
                if fact.sequence != current + 1:
                    raise StoreError(
                        "sequence must be strictly incremental",
                        expected=current + 1,
                        actual=fact.sequence,
                        objective_id=fact.objective_id,
                    )
                bucket.append(fact)
                self._sequences[fact.objective_id] = fact.sequence

            for record in outbox_records:
                if record.dispatch_id in self._outbox:
                    raise StoreError(
                        "duplicate dispatch_id",
                        dispatch_id=record.dispatch_id,
                    )
                self._outbox[record.dispatch_id] = record

            # Index session -> objectives from create facts
            for fact in facts:
                session_id = fact.payload.get("session_id")
                if session_id and fact.kind.value == "ObjectiveCreated":
                    self._session_objective_index.setdefault(session_id, set()).add(
                        fact.objective_id
                    )

            completed = CommandReceipt(
                receipt_id=receipt.receipt_id,
                scope=receipt.scope,
                idempotency_key=receipt.idempotency_key,
                request_hash=receipt.request_hash,
                status=receipt.status,
                response_payload=dict(receipt.response_payload),
                created_at=receipt.created_at,
                completed_at=receipt.completed_at or utc_now(),
            )
            self._receipts[key] = completed
            for oid in {f.objective_id for f in facts}:
                self._commit_notifier.notify(oid)
            return deepcopy(completed)
        except Exception:
            self._facts = facts_snap
            self._sequences = seq_snap
            self._slots = slots_snap
            self._outbox = outbox_snap
            self._session_objective_index = index_snap
            raise

    def _apply_slot(self, mutation: SlotMutation) -> None:
        if mutation.action == "acquire":
            existing = self._slots.get(mutation.session_id)
            if existing is not None and existing.objective_id != mutation.objective_id:
                raise StoreError(
                    "session already has an active objective slot",
                    session_id=mutation.session_id,
                    existing_objective_id=existing.objective_id,
                    requested_objective_id=mutation.objective_id,
                )
            if existing is not None and existing.objective_id == mutation.objective_id:
                return
            self._slots[mutation.session_id] = SessionSlot(
                session_id=mutation.session_id,
                objective_id=mutation.objective_id,
                acquired_at=mutation.acquired_at or utc_now(),
                objective_revision=mutation.objective_revision,
            )
            self._session_objective_index.setdefault(mutation.session_id, set()).add(
                mutation.objective_id
            )
            return

        # release
        existing = self._slots.get(mutation.session_id)
        if existing is None:
            return
        if existing.objective_id != mutation.objective_id:
            raise StoreError(
                "cannot release slot owned by another objective",
                session_id=mutation.session_id,
                existing_objective_id=existing.objective_id,
                requested_objective_id=mutation.objective_id,
            )
        del self._slots[mutation.session_id]

    async def get_receipt(
        self,
        *,
        scope: str,
        idempotency_key: str,
    ) -> CommandReceipt | None:
        async with self._global_lock:
            receipt = self._receipts.get((scope, idempotency_key))
            return deepcopy(receipt) if receipt else None

    async def list_facts(
        self,
        *,
        objective_id: str,
        after_sequence: int | None = None,
        limit: int = 10_000,
    ) -> list[ObjectiveLogEntry]:
        async with self._global_lock:
            entries = list(self._facts.get(objective_id, []))
        if after_sequence is not None:
            entries = [e for e in entries if e.sequence > after_sequence]
        return entries[:limit]

    async def get_max_sequence(self, objective_id: str) -> int:
        async with self._global_lock:
            return self._sequences.get(objective_id, 0)

    async def get_session_slot(self, session_id: str) -> SessionSlot | None:
        async with self._global_lock:
            slot = self._slots.get(session_id)
            return deepcopy(slot) if slot else None

    async def list_objective_ids_for_session(self, session_id: str) -> list[str]:
        async with self._global_lock:
            ids = sorted(self._session_objective_index.get(session_id, set()))
            # Also scan create facts if index empty (rebuild path)
            if not ids:
                for oid, facts in self._facts.items():
                    for fact in facts:
                        if (
                            fact.kind.value == "ObjectiveCreated"
                            and fact.payload.get("session_id") == session_id
                        ):
                            ids.append(oid)
                            break
            return ids

    async def list_objective_ids(self) -> list[str]:
        async with self._global_lock:
            return sorted(self._facts.keys())

    async def claim_dispatch(
        self,
        *,
        owner: str,
        lease_seconds: float = 30.0,
        now: datetime | None = None,
    ) -> DispatchRequested | None:
        now = _aware(now or utc_now())
        async with self._global_lock:
            candidates = sorted(
                self._outbox.values(),
                key=lambda r: r.created_at,
            )
            for record in candidates:
                if record.status == DispatchStatus.PENDING:
                    claimed = record.with_updates(
                        status=DispatchStatus.CLAIMED,
                        attempt=record.attempt + 1,
                        lease_owner=owner,
                        lease_expires_at=now + timedelta(seconds=lease_seconds),
                    )
                    self._outbox[record.dispatch_id] = claimed
                    return deepcopy(claimed)
                if record.status == DispatchStatus.CLAIMED:
                    expires = record.lease_expires_at
                    if expires is None or _aware(expires) <= now:
                        claimed = record.with_updates(
                            status=DispatchStatus.CLAIMED,
                            attempt=record.attempt + 1,
                            lease_owner=owner,
                            lease_expires_at=now + timedelta(seconds=lease_seconds),
                        )
                        self._outbox[record.dispatch_id] = claimed
                        return deepcopy(claimed)
            return None

    async def renew_dispatch_lease(
        self,
        *,
        dispatch_id: str,
        owner: str,
        lease_seconds: float = 30.0,
        now: datetime | None = None,
    ) -> DispatchRequested:
        now = _aware(now or utc_now())
        async with self._global_lock:
            record = self._outbox.get(dispatch_id)
            if record is None:
                raise StoreError("dispatch not found", dispatch_id=dispatch_id)
            if record.lease_owner != owner:
                raise StoreError(
                    "lease owner mismatch",
                    dispatch_id=dispatch_id,
                    owner=owner,
                    lease_owner=record.lease_owner,
                )
            if record.status != DispatchStatus.CLAIMED:
                raise StoreError(
                    "dispatch is not claimed",
                    dispatch_id=dispatch_id,
                    status=record.status.value,
                )
            renewed = record.with_updates(
                lease_owner=owner,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
            )
            self._outbox[dispatch_id] = renewed
            return deepcopy(renewed)

    async def complete_dispatch(
        self,
        *,
        dispatch_id: str,
        owner: str,
        status: Literal["dispatched", "completed", "failed"] = "dispatched",
        last_error: str | None = None,
        now: datetime | None = None,
    ) -> DispatchRequested:
        del now  # unused; completion is immediate
        async with self._global_lock:
            record = self._outbox.get(dispatch_id)
            if record is None:
                raise StoreError("dispatch not found", dispatch_id=dispatch_id)
            if record.lease_owner is not None and record.lease_owner != owner:
                raise StoreError(
                    "lease owner mismatch",
                    dispatch_id=dispatch_id,
                    owner=owner,
                    lease_owner=record.lease_owner,
                )
            new_status = DispatchStatus(status)
            completed = DispatchRequested(
                dispatch_id=record.dispatch_id,
                objective_id=record.objective_id,
                run_id=record.run_id,
                role=record.role,
                status=new_status,
                attempt=record.attempt,
                created_at=record.created_at,
                lease_owner=None,
                lease_expires_at=None,
                last_error=last_error,
                run_input=record.run_input,
                template_hash=record.template_hash,
                session_id=record.session_id,
                state_id=record.state_id,
            )
            self._outbox[dispatch_id] = completed
            return deepcopy(completed)

    async def release_dispatch(
        self,
        *,
        dispatch_id: str,
        owner: str,
        last_error: str | None = None,
        now: datetime | None = None,
    ) -> DispatchRequested:
        del now
        async with self._global_lock:
            record = self._outbox.get(dispatch_id)
            if record is None:
                raise StoreError("dispatch not found", dispatch_id=dispatch_id)
            if record.lease_owner is not None and record.lease_owner != owner:
                raise StoreError(
                    "lease owner mismatch",
                    dispatch_id=dispatch_id,
                    owner=owner,
                    lease_owner=record.lease_owner,
                )
            released = DispatchRequested(
                dispatch_id=record.dispatch_id,
                objective_id=record.objective_id,
                run_id=record.run_id,
                role=record.role,
                status=DispatchStatus.PENDING,
                attempt=record.attempt,
                created_at=record.created_at,
                lease_owner=None,
                lease_expires_at=None,
                last_error=last_error if last_error is not None else record.last_error,
            )
            self._outbox[dispatch_id] = released
            return deepcopy(released)

    async def get_dispatch(self, dispatch_id: str) -> DispatchRequested | None:
        async with self._global_lock:
            record = self._outbox.get(dispatch_id)
            return deepcopy(record) if record else None

    async def list_pending_dispatches(
        self,
        *,
        objective_id: str | None = None,
        limit: int = 100,
    ) -> list[DispatchRequested]:
        async with self._global_lock:
            records = list(self._outbox.values())
        if objective_id is not None:
            records = [r for r in records if r.objective_id == objective_id]
        records = [
            r
            for r in records
            if r.status in {DispatchStatus.PENDING, DispatchStatus.CLAIMED}
        ]
        records.sort(key=lambda r: r.created_at)
        return [deepcopy(r) for r in records[:limit]]

    async def list_dispatches(
        self,
        *,
        objective_id: str | None = None,
        limit: int = 100,
    ) -> list[DispatchRequested]:
        async with self._global_lock:
            records = list(self._outbox.values())
        if objective_id is not None:
            records = [r for r in records if r.objective_id == objective_id]
        records.sort(key=lambda r: r.created_at)
        return [deepcopy(r) for r in records[:limit]]


__all__ = ["InMemoryObjectiveStore"]
