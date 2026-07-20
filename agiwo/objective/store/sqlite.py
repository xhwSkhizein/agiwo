"""SQLite-backed ObjectiveStore (independent tables, shared connection runtime)."""

from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import sqlite3

import aiosqlite

from agiwo.objective.errors import IdempotencyConflict, StoreError
from agiwo.objective.log import ObjectiveLogEntry
from agiwo.objective.models import CommandReceiptStatus, DispatchStatus, utc_now
from agiwo.objective.outbox import DispatchRequested
from agiwo.objective.store.base import (
    CommandReceipt,
    ObjectiveStore,
    SessionSlot,
    SlotMutation,
)
from agiwo.objective.store.commit_rules import validate_commit_invariants
from agiwo.utils.logging import get_logger
from agiwo.utils.storage_support.sqlite_runtime import (
    SQLiteConnectionRuntime,
    begin_immediate,
    dumps_json_object,
    exclusive_sqlite_access,
    execute_statements,
    loads_json_object,
    safe_rollback,
)

logger = get_logger(__name__)


def _aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _dumps(payload: dict[str, Any]) -> str:
    return dumps_json_object(payload)


def _loads(raw: str) -> dict[str, Any]:
    try:
        return loads_json_object(raw)
    except TypeError as exc:
        raise StoreError("expected JSON object payload") from exc


class SQLiteObjectiveStore(ObjectiveStore):
    def __init__(self, db_path: str = "agiwo.db") -> None:
        super().__init__()
        self.db_path = db_path
        self._connection: aiosqlite.Connection | None = None
        self._runtime = SQLiteConnectionRuntime(
            db_path=db_path,
            logger=logger,
            connect_event="sqlite_objective_store_connected",
        )

    async def connect(self) -> None:
        self._connection = await self._runtime.ensure_connection(
            self._initialize_schema
        )

    async def close(self) -> None:
        await self.disconnect()

    async def disconnect(self) -> None:
        if self._connection is not None:
            await self._runtime.disconnect()
            self._connection = None

    async def _initialize_schema(self, connection: aiosqlite.Connection) -> None:
        await execute_statements(
            connection,
            [
                """
                CREATE TABLE IF NOT EXISTS objective_log_entries (
                    objective_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    fact_id TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (objective_id, sequence)
                )
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_objective_log_kind_seq
                ON objective_log_entries(objective_id, kind, sequence)
                """,
                """
                CREATE TABLE IF NOT EXISTS objective_session_slots (
                    session_id TEXT PRIMARY KEY,
                    objective_id TEXT NOT NULL,
                    acquired_at TEXT NOT NULL,
                    objective_revision INTEGER NOT NULL DEFAULT 0
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS objective_command_receipts (
                    scope TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    receipt_id TEXT NOT NULL,
                    PRIMARY KEY (scope, idempotency_key)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS objective_dispatch_outbox (
                    dispatch_id TEXT PRIMARY KEY,
                    objective_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    last_error TEXT,
                    payload TEXT NOT NULL
                )
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_objective_outbox_status_created
                ON objective_dispatch_outbox(status, created_at)
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_objective_outbox_objective
                ON objective_dispatch_outbox(objective_id, status)
                """,
                """
                CREATE TABLE IF NOT EXISTS objective_session_index (
                    session_id TEXT NOT NULL,
                    objective_id TEXT NOT NULL,
                    PRIMARY KEY (session_id, objective_id)
                )
                """,
            ],
        )
        await connection.commit()

    async def _ensure_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            await self.connect()
        assert self._connection is not None
        return self._connection

    async def commit_command(
        self,
        *,
        receipt: CommandReceipt,
        facts: list[ObjectiveLogEntry],
        slot_mutation: SlotMutation | None = None,
        outbox_records: list[DispatchRequested] | None = None,
    ) -> CommandReceipt:
        conn = await self._ensure_connection()
        outbox_records = outbox_records or []
        async with exclusive_sqlite_access(self.db_path):
            await begin_immediate(conn)
            try:
                existing = await self._get_receipt_unlocked(
                    conn,
                    scope=receipt.scope,
                    idempotency_key=receipt.idempotency_key,
                )
                if existing is not None:
                    if existing.request_hash != receipt.request_hash:
                        raise IdempotencyConflict(
                            scope=receipt.scope,
                            idempotency_key=receipt.idempotency_key,
                            existing_hash=existing.request_hash,
                            request_hash=receipt.request_hash,
                        )
                    await safe_rollback(conn)
                    return existing

                objective_ids = {fact.objective_id for fact in facts}
                if slot_mutation is not None:
                    objective_ids.add(slot_mutation.objective_id)
                existing_by_oid: dict[str, list[ObjectiveLogEntry]] = {}
                for oid in objective_ids:
                    existing_by_oid[oid] = await self._list_facts_unlocked(
                        conn, objective_id=oid
                    )
                validate_commit_invariants(
                    existing_facts_by_objective=existing_by_oid,
                    facts=facts,
                    slot_mutation=slot_mutation,
                    outbox_records=outbox_records,
                )

                if slot_mutation is not None:
                    await self._apply_slot_unlocked(conn, slot_mutation)

                for fact in facts:
                    await conn.execute(
                        """
                        INSERT INTO objective_log_entries
                        (objective_id, sequence, fact_id, kind, occurred_at, payload)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            fact.objective_id,
                            fact.sequence,
                            fact.fact_id,
                            fact.kind.value,
                            fact.occurred_at.isoformat(),
                            _dumps(fact.to_dict()),
                        ),
                    )
                    if fact.kind.value == "ObjectiveCreated":
                        session_id = fact.payload.get("session_id")
                        if session_id:
                            await conn.execute(
                                """
                                INSERT OR IGNORE INTO objective_session_index
                                (session_id, objective_id)
                                VALUES (?, ?)
                                """,
                                (session_id, fact.objective_id),
                            )

                for record in outbox_records:
                    await conn.execute(
                        """
                        INSERT INTO objective_dispatch_outbox
                        (dispatch_id, objective_id, run_id, role,
                         status, attempt, created_at, lease_owner, lease_expires_at,
                         last_error, payload)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.dispatch_id,
                            record.objective_id,
                            record.run_id,
                            record.role.value,
                            record.status.value,
                            record.attempt,
                            record.created_at.isoformat(),
                            record.lease_owner,
                            (
                                record.lease_expires_at.isoformat()
                                if record.lease_expires_at
                                else None
                            ),
                            record.last_error,
                            _dumps(record.to_dict()),
                        ),
                    )

                completed_at = receipt.completed_at or utc_now()
                completed = CommandReceipt(
                    receipt_id=receipt.receipt_id,
                    scope=receipt.scope,
                    idempotency_key=receipt.idempotency_key,
                    request_hash=receipt.request_hash,
                    status=receipt.status,
                    response_payload=dict(receipt.response_payload),
                    created_at=receipt.created_at,
                    completed_at=completed_at,
                )
                await conn.execute(
                    """
                    INSERT INTO objective_command_receipts
                    (scope, idempotency_key, request_hash, status,
                     response_payload, created_at, completed_at, receipt_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        completed.scope,
                        completed.idempotency_key,
                        completed.request_hash,
                        completed.status.value,
                        _dumps(completed.response_payload),
                        completed.created_at.isoformat(),
                        completed_at.isoformat(),
                        completed.receipt_id,
                    ),
                )
                await conn.commit()
                for oid in {f.objective_id for f in facts}:
                    self._commit_notifier.notify(oid)
                return completed
            except sqlite3.IntegrityError as exc:
                await safe_rollback(conn)
                msg = str(exc).lower()
                if "objective_session_slots" in msg or "unique" in msg:
                    raise StoreError(
                        "session already has an active objective slot or unique constraint failed",
                        error=str(exc),
                    ) from exc
                raise StoreError("sqlite integrity error", error=str(exc)) from exc
            except Exception:
                await safe_rollback(conn)
                raise

    async def _apply_slot_unlocked(
        self,
        conn: aiosqlite.Connection,
        mutation: SlotMutation,
    ) -> None:
        if mutation.action == "acquire":
            async with conn.execute(
                "SELECT objective_id FROM objective_session_slots WHERE session_id = ?",
                (mutation.session_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is not None:
                existing_oid = row[0]
                if existing_oid != mutation.objective_id:
                    raise StoreError(
                        "session already has an active objective slot",
                        session_id=mutation.session_id,
                        existing_objective_id=existing_oid,
                        requested_objective_id=mutation.objective_id,
                    )
                return
            acquired_at = (mutation.acquired_at or utc_now()).isoformat()
            await conn.execute(
                """
                INSERT INTO objective_session_slots
                (session_id, objective_id, acquired_at, objective_revision)
                VALUES (?, ?, ?, ?)
                """,
                (
                    mutation.session_id,
                    mutation.objective_id,
                    acquired_at,
                    mutation.objective_revision,
                ),
            )
            await conn.execute(
                """
                INSERT OR IGNORE INTO objective_session_index
                (session_id, objective_id)
                VALUES (?, ?)
                """,
                (mutation.session_id, mutation.objective_id),
            )
            return

        async with conn.execute(
            "SELECT objective_id FROM objective_session_slots WHERE session_id = ?",
            (mutation.session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return
        if row[0] != mutation.objective_id:
            raise StoreError(
                "cannot release slot owned by another objective",
                session_id=mutation.session_id,
                existing_objective_id=row[0],
                requested_objective_id=mutation.objective_id,
            )
        await conn.execute(
            "DELETE FROM objective_session_slots WHERE session_id = ?",
            (mutation.session_id,),
        )

    async def _get_receipt_unlocked(
        self,
        conn: aiosqlite.Connection,
        *,
        scope: str,
        idempotency_key: str,
    ) -> CommandReceipt | None:
        async with conn.execute(
            """
            SELECT scope, idempotency_key, request_hash, status,
                   response_payload, created_at, completed_at, receipt_id
            FROM objective_command_receipts
            WHERE scope = ? AND idempotency_key = ?
            """,
            (scope, idempotency_key),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return CommandReceipt(
            scope=row[0],
            idempotency_key=row[1],
            request_hash=row[2],
            status=CommandReceiptStatus(row[3]),
            response_payload=_loads(row[4]),
            created_at=datetime.fromisoformat(row[5]),
            completed_at=datetime.fromisoformat(row[6]) if row[6] else None,
            receipt_id=row[7],
        )

    async def get_receipt(
        self,
        *,
        scope: str,
        idempotency_key: str,
    ) -> CommandReceipt | None:
        conn = await self._ensure_connection()
        async with exclusive_sqlite_access(self.db_path):
            return await self._get_receipt_unlocked(
                conn, scope=scope, idempotency_key=idempotency_key
            )

    async def _list_facts_unlocked(
        self,
        conn: aiosqlite.Connection,
        *,
        objective_id: str,
        after_sequence: int | None = None,
        limit: int = 10_000,
    ) -> list[ObjectiveLogEntry]:
        if after_sequence is None:
            sql = """
                SELECT payload FROM objective_log_entries
                WHERE objective_id = ?
                ORDER BY sequence ASC
                LIMIT ?
            """
            params: tuple[Any, ...] = (objective_id, limit)
        else:
            sql = """
                SELECT payload FROM objective_log_entries
                WHERE objective_id = ? AND sequence > ?
                ORDER BY sequence ASC
                LIMIT ?
            """
            params = (objective_id, after_sequence, limit)
        async with conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [ObjectiveLogEntry.from_dict(_loads(row[0])) for row in rows]

    async def list_facts(
        self,
        *,
        objective_id: str,
        after_sequence: int | None = None,
        limit: int = 10_000,
    ) -> list[ObjectiveLogEntry]:
        conn = await self._ensure_connection()
        async with exclusive_sqlite_access(self.db_path):
            return await self._list_facts_unlocked(
                conn,
                objective_id=objective_id,
                after_sequence=after_sequence,
                limit=limit,
            )

    async def get_max_sequence(self, objective_id: str) -> int:
        conn = await self._ensure_connection()
        async with exclusive_sqlite_access(self.db_path):
            async with conn.execute(
                """
                SELECT COALESCE(MAX(sequence), 0)
                FROM objective_log_entries
                WHERE objective_id = ?
                """,
                (objective_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def get_session_slot(self, session_id: str) -> SessionSlot | None:
        conn = await self._ensure_connection()
        async with exclusive_sqlite_access(self.db_path):
            async with conn.execute(
                """
                SELECT session_id, objective_id, acquired_at, objective_revision
                FROM objective_session_slots
                WHERE session_id = ?
                """,
                (session_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return SessionSlot(
            session_id=row[0],
            objective_id=row[1],
            acquired_at=datetime.fromisoformat(row[2]),
            objective_revision=int(row[3]),
        )

    async def list_objective_ids_for_session(self, session_id: str) -> list[str]:
        conn = await self._ensure_connection()
        async with exclusive_sqlite_access(self.db_path):
            async with conn.execute(
                """
                SELECT objective_id FROM objective_session_index
                WHERE session_id = ?
                ORDER BY objective_id
                """,
                (session_id,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def list_objective_ids(self) -> list[str]:
        conn = await self._ensure_connection()
        async with exclusive_sqlite_access(self.db_path):
            async with conn.execute(
                """
                SELECT DISTINCT objective_id
                FROM objective_log_entries
                ORDER BY objective_id
                """
            ) as cursor:
                rows = await cursor.fetchall()
        return [row[0] for row in rows]

    def _row_to_dispatch(
        self, row: aiosqlite.Row | tuple[Any, ...]
    ) -> DispatchRequested:
        if isinstance(row, dict):
            payload = _loads(row["payload"])
        else:
            payload = _loads(row[10])
        return DispatchRequested.from_dict(payload)

    async def _load_dispatch(
        self,
        conn: aiosqlite.Connection,
        dispatch_id: str,
    ) -> DispatchRequested | None:
        async with conn.execute(
            """
            SELECT dispatch_id, objective_id, run_id, role,
                   status, attempt, created_at, lease_owner, lease_expires_at,
                   last_error, payload
            FROM objective_dispatch_outbox
            WHERE dispatch_id = ?
            """,
            (dispatch_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_dispatch(row)

    async def _save_dispatch(
        self,
        conn: aiosqlite.Connection,
        record: DispatchRequested,
    ) -> None:
        await conn.execute(
            """
            UPDATE objective_dispatch_outbox
            SET status = ?, attempt = ?, lease_owner = ?, lease_expires_at = ?,
                last_error = ?, payload = ?
            WHERE dispatch_id = ?
            """,
            (
                record.status.value,
                record.attempt,
                record.lease_owner,
                (
                    record.lease_expires_at.isoformat()
                    if record.lease_expires_at
                    else None
                ),
                record.last_error,
                _dumps(record.to_dict()),
                record.dispatch_id,
            ),
        )

    async def claim_dispatch(
        self,
        *,
        owner: str,
        lease_seconds: float = 30.0,
        now: datetime | None = None,
    ) -> DispatchRequested | None:
        now = _aware(now or utc_now())
        conn = await self._ensure_connection()
        async with exclusive_sqlite_access(self.db_path):
            await begin_immediate(conn)
            try:
                async with conn.execute(
                    """
                    SELECT dispatch_id, payload, status, lease_expires_at
                    FROM objective_dispatch_outbox
                    WHERE status IN ('pending', 'claimed')
                    ORDER BY created_at ASC
                    """
                ) as cursor:
                    rows = await cursor.fetchall()
                for row in rows:
                    dispatch_id, payload_raw, status, lease_expires_at = row
                    record = DispatchRequested.from_dict(_loads(payload_raw))
                    if status == DispatchStatus.PENDING.value:
                        claimed = record.with_updates(
                            status=DispatchStatus.CLAIMED,
                            attempt=record.attempt + 1,
                            lease_owner=owner,
                            lease_expires_at=now + timedelta(seconds=lease_seconds),
                        )
                        await self._save_dispatch(conn, claimed)
                        await conn.commit()
                        return claimed
                    if status == DispatchStatus.CLAIMED.value:
                        expires = (
                            datetime.fromisoformat(lease_expires_at)
                            if lease_expires_at
                            else None
                        )
                        if expires is None or _aware(expires) <= now:
                            claimed = record.with_updates(
                                status=DispatchStatus.CLAIMED,
                                attempt=record.attempt + 1,
                                lease_owner=owner,
                                lease_expires_at=now + timedelta(seconds=lease_seconds),
                            )
                            await self._save_dispatch(conn, claimed)
                            await conn.commit()
                            return claimed
                await safe_rollback(conn)
                return None
            except Exception:
                await safe_rollback(conn)
                raise

    async def renew_dispatch_lease(
        self,
        *,
        dispatch_id: str,
        owner: str,
        lease_seconds: float = 30.0,
        now: datetime | None = None,
    ) -> DispatchRequested:
        now = _aware(now or utc_now())
        conn = await self._ensure_connection()
        async with exclusive_sqlite_access(self.db_path):
            await begin_immediate(conn)
            try:
                record = await self._load_dispatch(conn, dispatch_id)
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
                await self._save_dispatch(conn, renewed)
                await conn.commit()
                return renewed
            except Exception:
                await safe_rollback(conn)
                raise

    async def complete_dispatch(
        self,
        *,
        dispatch_id: str,
        owner: str,
        status: Literal["dispatched", "completed", "failed"] = "dispatched",
        last_error: str | None = None,
        now: datetime | None = None,
    ) -> DispatchRequested:
        del now
        conn = await self._ensure_connection()
        async with exclusive_sqlite_access(self.db_path):
            await begin_immediate(conn)
            try:
                record = await self._load_dispatch(conn, dispatch_id)
                if record is None:
                    raise StoreError("dispatch not found", dispatch_id=dispatch_id)
                if record.lease_owner is not None and record.lease_owner != owner:
                    raise StoreError(
                        "lease owner mismatch",
                        dispatch_id=dispatch_id,
                        owner=owner,
                        lease_owner=record.lease_owner,
                    )
                completed = record.with_updates(
                    status=DispatchStatus(status),
                    lease_owner=None,
                    lease_expires_at=None,
                    last_error=last_error,
                )
                await self._save_dispatch(conn, completed)
                await conn.commit()
                return completed
            except Exception:
                await safe_rollback(conn)
                raise

    async def release_dispatch(
        self,
        *,
        dispatch_id: str,
        owner: str,
        last_error: str | None = None,
        now: datetime | None = None,
    ) -> DispatchRequested:
        del now
        conn = await self._ensure_connection()
        async with exclusive_sqlite_access(self.db_path):
            await begin_immediate(conn)
            try:
                record = await self._load_dispatch(conn, dispatch_id)
                if record is None:
                    raise StoreError("dispatch not found", dispatch_id=dispatch_id)
                if record.lease_owner is not None and record.lease_owner != owner:
                    raise StoreError(
                        "lease owner mismatch",
                        dispatch_id=dispatch_id,
                        owner=owner,
                        lease_owner=record.lease_owner,
                    )
                released = record.with_updates(
                    status=DispatchStatus.PENDING,
                    lease_owner=None,
                    lease_expires_at=None,
                    last_error=(
                        last_error if last_error is not None else record.last_error
                    ),
                )
                await self._save_dispatch(conn, released)
                await conn.commit()
                return released
            except Exception:
                await safe_rollback(conn)
                raise

    async def get_dispatch(self, dispatch_id: str) -> DispatchRequested | None:
        conn = await self._ensure_connection()
        async with exclusive_sqlite_access(self.db_path):
            return await self._load_dispatch(conn, dispatch_id)

    async def list_pending_dispatches(
        self,
        *,
        objective_id: str | None = None,
        limit: int = 100,
    ) -> list[DispatchRequested]:
        conn = await self._ensure_connection()
        if objective_id is None:
            sql = """
                SELECT payload FROM objective_dispatch_outbox
                WHERE status IN ('pending', 'claimed')
                ORDER BY created_at ASC
                LIMIT ?
            """
            params: tuple[Any, ...] = (limit,)
        else:
            sql = """
                SELECT payload FROM objective_dispatch_outbox
                WHERE status IN ('pending', 'claimed') AND objective_id = ?
                ORDER BY created_at ASC
                LIMIT ?
            """
            params = (objective_id, limit)
        async with exclusive_sqlite_access(self.db_path):
            async with conn.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
        return [DispatchRequested.from_dict(_loads(row[0])) for row in rows]

    async def list_dispatches(
        self,
        *,
        objective_id: str | None = None,
        limit: int = 100,
    ) -> list[DispatchRequested]:
        conn = await self._ensure_connection()
        if objective_id is None:
            sql = """
                SELECT payload FROM objective_dispatch_outbox
                ORDER BY created_at ASC
                LIMIT ?
            """
            params: tuple[Any, ...] = (limit,)
        else:
            sql = """
                SELECT payload FROM objective_dispatch_outbox
                WHERE objective_id = ?
                ORDER BY created_at ASC
                LIMIT ?
            """
            params = (objective_id, limit)
        async with exclusive_sqlite_access(self.db_path):
            async with conn.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
        return [DispatchRequested.from_dict(_loads(row[0])) for row in rows]


__all__ = ["SQLiteObjectiveStore"]
