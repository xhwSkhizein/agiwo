"""SessionIntent store protocol and in-memory implementation (I-S1)."""

import time
from abc import ABC, abstractmethod

from agiwo.agent.intent.models import IntentEntry, SessionIntent
from agiwo.agent.models.plan import RunPlan


class SessionIntentStore(ABC):
    """Own persistence for SessionIntent keyed by session_id."""

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
        """Append one entry and optionally refresh last_run_plan."""
        current = await self.get(session_id)
        if current is None:
            current = SessionIntent()
        current.entries.append(entry)
        if last_run_plan is not None:
            current.last_run_plan = last_run_plan
        current.updated_at = int(time.time())
        await self.upsert(session_id, current)
        return current


class InMemorySessionIntentStore(SessionIntentStore):
    """In-process SessionIntent store for tests and memory-backed agents."""

    def __init__(self) -> None:
        self._intents: dict[str, SessionIntent] = {}

    async def get(self, session_id: str) -> SessionIntent | None:
        stored = self._intents.get(session_id)
        if stored is None:
            return None
        return SessionIntent(
            entries=list(stored.entries),
            last_run_plan=stored.last_run_plan,
            updated_at=stored.updated_at,
        )

    async def upsert(self, session_id: str, intent: SessionIntent) -> None:
        self._intents[session_id] = SessionIntent(
            entries=list(intent.entries),
            last_run_plan=intent.last_run_plan,
            updated_at=intent.updated_at,
        )


__all__ = ["InMemorySessionIntentStore", "SessionIntentStore"]
