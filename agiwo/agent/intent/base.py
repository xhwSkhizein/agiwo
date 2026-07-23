"""SessionIntent store protocol and in-memory implementation (I-S1)."""

import asyncio
import time
from abc import ABC, abstractmethod

from agiwo.agent.intent.models import (
    MAX_INTENT_ENTRIES,
    IntentEntry,
    SessionIntent,
)
from agiwo.agent.models.plan import RunPlan


class SessionIntentStore(ABC):
    """Own persistence for SessionIntent keyed by session_id."""

    def __init__(self) -> None:
        self._append_locks: dict[str, asyncio.Lock] = {}

    async def close(self) -> None:
        """Release resources held by the store."""

    @abstractmethod
    async def get(self, session_id: str) -> SessionIntent | None:
        """Load SessionIntent for one session, or None when absent."""

    @abstractmethod
    async def upsert(self, session_id: str, intent: SessionIntent) -> None:
        """Replace SessionIntent for one session."""

    async def append_entry(
        self,
        session_id: str,
        entry: IntentEntry,
        *,
        last_run_plan: RunPlan | None = None,
    ) -> SessionIntent:
        """Append one entry and optionally refresh last_run_plan.

        Per-session lock covers single-process asyncio interleaving between
        concurrent writers (accept + completion-task run_report).
        """
        lock = self._append_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            return await self._append_entry_unlocked(
                session_id,
                entry,
                last_run_plan=last_run_plan,
            )

    async def _append_entry_unlocked(
        self,
        session_id: str,
        entry: IntentEntry,
        *,
        last_run_plan: RunPlan | None = None,
    ) -> SessionIntent:
        """Default read-modify-write append; backends may override."""
        current = await self.get(session_id)
        if current is None:
            current = SessionIntent()
        current.entries.append(entry)
        if last_run_plan is not None:
            current.last_run_plan = last_run_plan
        current.updated_at = int(time.time())
        if len(current.entries) > MAX_INTENT_ENTRIES * 2:
            current.entries = current.entries[-MAX_INTENT_ENTRIES:]
        await self.upsert(session_id, current)
        # Readers always see at most N newest entries.
        if len(current.entries) > MAX_INTENT_ENTRIES:
            current.entries = current.entries[-MAX_INTENT_ENTRIES:]
        return current


class InMemorySessionIntentStore(SessionIntentStore):
    """In-process SessionIntent store for tests and memory-backed agents."""

    def __init__(self) -> None:
        super().__init__()
        self._intents: dict[str, SessionIntent] = {}

    async def get(self, session_id: str) -> SessionIntent | None:
        stored = self._intents.get(session_id)
        if stored is None:
            return None
        entries = stored.entries
        if len(entries) > MAX_INTENT_ENTRIES:
            entries = entries[-MAX_INTENT_ENTRIES:]
        return SessionIntent(
            entries=list(entries),
            last_run_plan=stored.last_run_plan,
            updated_at=stored.updated_at,
        )

    async def upsert(self, session_id: str, intent: SessionIntent) -> None:
        self._intents[session_id] = SessionIntent(
            entries=list(intent.entries),
            last_run_plan=intent.last_run_plan,
            updated_at=intent.updated_at,
        )

    async def _append_entry_unlocked(
        self,
        session_id: str,
        entry: IntentEntry,
        *,
        last_run_plan: RunPlan | None = None,
    ) -> SessionIntent:
        # Append against the full stored list (get() truncates for readers).
        stored = self._intents.get(session_id)
        if stored is None:
            stored = SessionIntent()
        else:
            stored = SessionIntent(
                entries=list(stored.entries),
                last_run_plan=stored.last_run_plan,
                updated_at=stored.updated_at,
            )
        stored.entries.append(entry)
        if last_run_plan is not None:
            stored.last_run_plan = last_run_plan
        stored.updated_at = int(time.time())
        if len(stored.entries) > MAX_INTENT_ENTRIES * 2:
            stored.entries = stored.entries[-MAX_INTENT_ENTRIES:]
        await self.upsert(session_id, stored)
        loaded = await self.get(session_id)
        assert loaded is not None
        return loaded


__all__ = ["InMemorySessionIntentStore", "SessionIntentStore"]
