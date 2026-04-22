"""Replayable run-log storage interfaces and in-memory implementation."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

from agiwo.agent.models.log import (
    AssistantStepCommitted,
    CompactionApplied,
    RunLogEntry,
    StepCondensedContentUpdated,
    ToolStepCommitted,
    UserStepCommitted,
    build_compact_metadata_from_entry,
)
from agiwo.agent.models.run import CompactMetadata, RunView
from agiwo.agent.models.step import StepView
from agiwo.agent.storage.serialization import (
    build_run_view_from_entries,
    build_run_views_from_entries,
    build_step_views_from_entries,
)


@dataclass(frozen=True)
class SessionRunStats:
    committed_step_count: int
    run_views: list[RunView]


class RunLogStorage(ABC):
    """Append-only run-log storage plus replay/query helpers."""

    async def close(self) -> None:
        """Close storage and release resources (optional)."""

    @abstractmethod
    async def append_entries(self, entries: list[RunLogEntry]) -> None:
        """Append run-log entries."""
        ...

    @abstractmethod
    async def list_entries(
        self,
        *,
        session_id: str,
        run_id: str | None = None,
        agent_id: str | None = None,
        after_sequence: int | None = None,
        limit: int = 1000,
    ) -> list[RunLogEntry]:
        """List run-log entries in ascending sequence order."""
        ...

    @abstractmethod
    async def get_run_view(self, run_id: str) -> RunView | None:
        """Build one run view from run-log entries."""
        ...

    @abstractmethod
    async def list_run_views(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[RunView]:
        """List run views in descending created-at order."""
        ...

    async def get_latest_run_view(self, session_id: str) -> RunView | None:
        runs = await self.list_run_views(session_id=session_id, limit=1)
        return runs[0] if runs else None

    async def get_session_run_stats(self, session_id: str) -> SessionRunStats:
        run_views = await self.list_run_views(session_id=session_id, limit=100_000)
        return SessionRunStats(
            committed_step_count=await self.get_committed_step_count(session_id),
            run_views=run_views,
        )

    async def batch_count_run_views(self, session_ids: list[str]) -> dict[str, int]:
        result: dict[str, int] = {}
        for session_id in session_ids:
            result[session_id] = len(
                await self.list_run_views(session_id=session_id, limit=100_000)
            )
        return result

    async def batch_get_latest_run_views(
        self, session_ids: list[str]
    ) -> dict[str, RunView | None]:
        result: dict[str, RunView | None] = {}
        for session_id in session_ids:
            result[session_id] = await self.get_latest_run_view(session_id)
        return result

    @abstractmethod
    async def get_committed_step_count(self, session_id: str) -> int:
        """Count committed step entries for one session."""
        ...

    async def batch_get_committed_step_counts(
        self, session_ids: list[str]
    ) -> dict[str, int]:
        result: dict[str, int] = {}
        for session_id in session_ids:
            result[session_id] = await self.get_committed_step_count(session_id)
        return result

    @abstractmethod
    async def list_step_views(
        self,
        *,
        session_id: str,
        start_seq: int | None = None,
        end_seq: int | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 1000,
    ) -> list[StepView]:
        """List committed step views in ascending sequence order."""
        ...

    @abstractmethod
    async def append_step_condensed_content(
        self,
        session_id: str,
        run_id: str,
        agent_id: str,
        step_id: str,
        condensed_content: str,
    ) -> bool:
        """Append a step-condensation fact for an existing committed step."""
        ...

    async def get_step_by_tool_call_id(
        self,
        session_id: str,
        tool_call_id: str,
    ) -> StepView | None:
        steps = await self.list_step_views(session_id=session_id, limit=100_000)
        for step in steps:
            if step.tool_call_id == tool_call_id:
                return step
        return None

    async def get_latest_compact_metadata(
        self, session_id: str, agent_id: str
    ) -> CompactMetadata | None:
        entries = await self.list_entries(
            session_id=session_id,
            agent_id=agent_id,
            limit=100_000,
        )
        compact_entries = [
            entry for entry in entries if isinstance(entry, CompactionApplied)
        ]
        if not compact_entries:
            return None
        return build_compact_metadata_from_entry(compact_entries[-1])

    async def get_compact_history(
        self, session_id: str, agent_id: str
    ) -> list[CompactMetadata]:
        entries = await self.list_entries(
            session_id=session_id,
            agent_id=agent_id,
            limit=100_000,
        )
        compact_entries = [
            entry for entry in entries if isinstance(entry, CompactionApplied)
        ]
        return [build_compact_metadata_from_entry(entry) for entry in compact_entries]

    @abstractmethod
    async def get_max_sequence(self, session_id: str) -> int:
        """Return the current max session sequence, or 0 when empty."""
        ...

    @abstractmethod
    async def allocate_sequence(self, session_id: str) -> int:
        """Allocate the next session sequence atomically."""
        ...


class InMemoryRunLogStorage(RunLogStorage):
    """In-memory run-log storage used by tests and local runtime instances."""

    def __init__(self) -> None:
        self.run_log_entries: dict[str, list[RunLogEntry]] = {}
        self._sequence_counters: dict[str, int] = {}
        self._sequence_locks: dict[str, asyncio.Lock] = {}

    async def append_entries(self, entries: list[RunLogEntry]) -> None:
        for entry in entries:
            bucket = self.run_log_entries.setdefault(entry.session_id, [])
            bucket.append(entry)
            bucket.sort(key=lambda item: item.sequence)
            current = self._sequence_counters.get(entry.session_id, 0)
            if entry.sequence > current:
                self._sequence_counters[entry.session_id] = entry.sequence

    async def list_entries(
        self,
        *,
        session_id: str,
        run_id: str | None = None,
        agent_id: str | None = None,
        after_sequence: int | None = None,
        limit: int = 1000,
    ) -> list[RunLogEntry]:
        entries = list(self.run_log_entries.get(session_id, []))
        if run_id is not None:
            entries = [entry for entry in entries if entry.run_id == run_id]
        if agent_id is not None:
            entries = [entry for entry in entries if entry.agent_id == agent_id]
        if after_sequence is not None:
            entries = [entry for entry in entries if entry.sequence > after_sequence]
        return entries[:limit]

    async def get_run_view(self, run_id: str) -> RunView | None:
        for entries in self.run_log_entries.values():
            run_entries = [entry for entry in entries if entry.run_id == run_id]
            if run_entries:
                return build_run_view_from_entries(run_entries)
        return None

    async def list_run_views(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[RunView]:
        entries: list[RunLogEntry] = []
        session_ids = (
            [session_id] if session_id is not None else list(self.run_log_entries)
        )
        for sid in session_ids:
            entries.extend(self.run_log_entries.get(sid, []))
        run_views = build_run_views_from_entries(entries)
        if user_id is not None:
            run_views = [view for view in run_views if view.user_id == user_id]
        return run_views[offset : offset + limit]

    async def get_committed_step_count(self, session_id: str) -> int:
        return len(await self.list_step_views(session_id=session_id, limit=100_000))

    async def get_session_run_stats(self, session_id: str) -> SessionRunStats:
        entries = list(self.run_log_entries.get(session_id, []))
        run_views = build_run_views_from_entries(entries)
        step_count = len(build_step_views_from_entries(entries))
        return SessionRunStats(
            committed_step_count=step_count,
            run_views=run_views,
        )

    async def list_step_views(
        self,
        *,
        session_id: str,
        start_seq: int | None = None,
        end_seq: int | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 1000,
    ) -> list[StepView]:
        entries = await self.list_entries(
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            limit=100_000,
        )
        step_views = build_step_views_from_entries(entries)
        if start_seq is not None:
            step_views = [step for step in step_views if step.sequence >= start_seq]
        if end_seq is not None:
            step_views = [step for step in step_views if step.sequence <= end_seq]
        return step_views[:limit]

    async def append_step_condensed_content(
        self,
        session_id: str,
        run_id: str,
        agent_id: str,
        step_id: str,
        condensed_content: str,
    ) -> bool:
        bucket = self.run_log_entries.get(session_id, [])
        for entry in bucket:
            if not isinstance(
                entry,
                (UserStepCommitted, AssistantStepCommitted, ToolStepCommitted),
            ):
                continue
            if entry.step_id == step_id:
                sequence = await self.allocate_sequence(session_id)
                await self.append_entries(
                    [
                        StepCondensedContentUpdated(
                            sequence=sequence,
                            session_id=session_id,
                            run_id=run_id,
                            agent_id=agent_id,
                            step_id=step_id,
                            condensed_content=condensed_content,
                        )
                    ]
                )
                return True
        return False

    async def get_max_sequence(self, session_id: str) -> int:
        entries = self.run_log_entries.get(session_id, [])
        return max((entry.sequence for entry in entries), default=0)

    async def allocate_sequence(self, session_id: str) -> int:
        if session_id not in self._sequence_locks:
            self._sequence_locks[session_id] = asyncio.Lock()
        async with self._sequence_locks[session_id]:
            current = self._sequence_counters.get(session_id)
            if current is None:
                current = await self.get_max_sequence(session_id)
            current += 1
            self._sequence_counters[session_id] = current
            return current


__all__ = [
    "InMemoryRunLogStorage",
    "RunLogStorage",
    "SessionRunStats",
]
