"""
Repository interface and in-memory implementation.
"""

import asyncio
import bisect
from abc import ABC, abstractmethod

from agiwo.agent.models.log import RunLogEntry
from agiwo.agent.models.log import (
    AssistantStepCommitted,
    CompactionApplied,
    ToolStepCommitted,
    UserStepCommitted,
    build_compact_metadata_from_entry,
)
from agiwo.agent.models.run import CompactMetadata, Run, RunView
from agiwo.agent.models.step import StepRecord, StepView
from agiwo.agent.storage.serialization import (
    build_run_view_from_run,
    build_run_view_from_entries,
    build_run_views_from_entries,
    build_step_record_from_view,
    build_step_view_from_record,
    build_step_views_from_entries,
)


class RunLogStorage(ABC):
    """Replayable run-log storage interface."""

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


class RunStepStorage(RunLogStorage, ABC):
    """
    Run and Step storage interface.
    Responsible for Run and Step persistence and queries.
    """

    # --- Lifecycle Methods ---

    async def close(self) -> None:
        """Close storage and release resources (optional)."""
        pass

    # --- Run Operations ---

    @abstractmethod
    async def save_run(self, run: Run) -> None:
        """Save Run"""
        ...

    @abstractmethod
    async def get_run(self, run_id: str) -> Run | None:
        """Get Run"""
        ...

    @abstractmethod
    async def list_runs(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Run]:
        """List Runs"""
        ...

    @abstractmethod
    async def count_runs(self, session_id: str) -> int:
        """Count total runs for a session."""
        ...

    @abstractmethod
    async def delete_run(self, run_id: str) -> None:
        """Delete Run"""
        ...

    # --- Step Operations ---

    @abstractmethod
    async def save_step(self, step: StepRecord) -> None:
        """Save single Step"""
        ...

    @abstractmethod
    async def save_steps_batch(self, steps: list[StepRecord]) -> None:
        """Batch save Steps (for fork operations)"""
        ...

    @abstractmethod
    async def get_steps(
        self,
        session_id: str,
        start_seq: int | None = None,
        end_seq: int | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 1000,
    ) -> list[StepRecord]:
        """
        Get Steps (sorted by sequence)

        Args:
            session_id: Session ID
            start_seq: Start sequence (inclusive), None = from beginning
            end_seq: End sequence (inclusive), None = to end
            run_id: Filter by run_id (optional)
            agent_id: Filter by agent_id (optional)
            limit: Maximum return count
        """
        ...

    @abstractmethod
    async def get_last_step(self, session_id: str) -> StepRecord | None:
        """Get last Step"""
        ...

    @abstractmethod
    async def delete_steps(self, session_id: str, start_seq: int) -> int:
        """
        Delete Steps (sequence >= start_seq)

        Returns:
            Number of deleted Steps
        """
        ...

    @abstractmethod
    async def get_step_count(self, session_id: str) -> int:
        """Get total Step count for session"""
        ...

    @abstractmethod
    async def get_max_sequence(self, session_id: str) -> int:
        """
        Get the maximum sequence number in the session.

        Returns:
            Maximum sequence number, or 0 if no steps exist
        """
        ...

    @abstractmethod
    async def allocate_sequence(self, session_id: str) -> int:
        """
        Atomically allocate next sequence number for a session.
        Thread-safe and concurrent-safe operation.

        Args:
            session_id: Session ID

        Returns:
            Next sequence number (starting from 1)
        """
        ...

    # --- Batch Query Operations (for list views) ---

    async def batch_count_runs(self, session_ids: list[str]) -> dict[str, int]:
        """Count runs for multiple sessions in one call.

        Default falls back to per-session queries; subclasses should override
        with a single batch query for efficiency.
        """
        result: dict[str, int] = {}
        for sid in session_ids:
            result[sid] = await self.count_runs(sid)
        return result

    async def batch_count_run_views(self, session_ids: list[str]) -> dict[str, int]:
        """Count replayable runs for multiple sessions in one call."""
        result: dict[str, int] = {}
        for session_id in session_ids:
            result[session_id] = len(
                await self.list_run_views(session_id=session_id, limit=100_000)
            )
        return result

    async def batch_get_step_counts(self, session_ids: list[str]) -> dict[str, int]:
        """Get step counts for multiple sessions in one call.

        Default falls back to per-session queries; subclasses should override
        with a single batch query for efficiency.
        """
        result: dict[str, int] = {}
        for sid in session_ids:
            result[sid] = await self.get_step_count(sid)
        return result

    async def batch_get_latest_runs(
        self, session_ids: list[str]
    ) -> dict[str, "Run | None"]:
        """Get the latest run for each session in one call.

        Default falls back to per-session queries; subclasses should override
        with a single batch query for efficiency.
        """
        result: dict[str, Run | None] = {}
        for sid in session_ids:
            runs = await self.list_runs(session_id=sid, limit=1)
            result[sid] = runs[0] if runs else None
        return result

    async def get_run_view(self, run_id: str) -> RunView | None:
        run = await self.get_run(run_id)
        if run is None:
            return None
        return build_run_view_from_run(run)

    async def list_run_views(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[RunView]:
        runs = await self.list_runs(
            user_id=user_id,
            session_id=session_id,
            limit=limit,
            offset=offset,
        )
        return [build_run_view_from_run(run) for run in runs]

    async def get_latest_run_view(self, session_id: str) -> RunView | None:
        entries = await self.list_entries(session_id=session_id, limit=100_000)
        if not entries:
            runs = await self.list_run_views(session_id=session_id, limit=1)
            return runs[0] if runs else None
        latest_by_run_id: dict[str, list[RunLogEntry]] = {}
        for entry in entries:
            latest_by_run_id.setdefault(entry.run_id, []).append(entry)
        if not latest_by_run_id:
            runs = await self.list_run_views(session_id=session_id, limit=1)
            return runs[0] if runs else None
        latest_entries = max(
            latest_by_run_id.values(),
            key=lambda run_entries: run_entries[-1].sequence,
        )
        return build_run_view_from_entries(latest_entries)

    async def batch_get_latest_run_views(
        self, session_ids: list[str]
    ) -> dict[str, RunView | None]:
        result: dict[str, RunView | None] = {}
        for session_id in session_ids:
            result[session_id] = await self.get_latest_run_view(session_id)
        return result

    async def get_committed_step_count(self, session_id: str) -> int:
        entries = await self.list_entries(session_id=session_id, limit=100_000)
        committed = sum(
            1
            for entry in entries
            if isinstance(
                entry,
                (UserStepCommitted, AssistantStepCommitted, ToolStepCommitted),
            )
        )
        if committed == 0:
            return await self.get_step_count(session_id)
        return committed

    async def batch_get_committed_step_counts(
        self, session_ids: list[str]
    ) -> dict[str, int]:
        result: dict[str, int] = {}
        for session_id in session_ids:
            result[session_id] = await self.get_committed_step_count(session_id)
        return result

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
        if not entries:
            steps = await self.get_steps(
                session_id,
                start_seq=start_seq,
                end_seq=end_seq,
                run_id=run_id,
                agent_id=agent_id,
                limit=limit,
            )
            return [build_step_view_from_record(step) for step in steps]
        step_views = build_step_views_from_entries(entries)
        if not step_views:
            steps = await self.get_steps(
                session_id,
                start_seq=start_seq,
                end_seq=end_seq,
                run_id=run_id,
                agent_id=agent_id,
                limit=limit,
            )
            return [build_step_view_from_record(step) for step in steps]
        if start_seq is not None:
            step_views = [step for step in step_views if step.sequence >= start_seq]
        if end_seq is not None:
            step_views = [step for step in step_views if step.sequence <= end_seq]
        return step_views[:limit]

    # --- Step Content Update (for retrospect) ---

    async def update_step_condensed_content(
        self,
        session_id: str,
        step_id: str,
        condensed_content: str,
    ) -> bool:
        """Update the condensed_content field of a step.

        Returns True if the step was found and updated.
        """
        return False

    # --- Tool Result Query (for cross-agent reference) ---

    async def get_step_by_tool_call_id(
        self,
        session_id: str,
        tool_call_id: str,
    ) -> StepRecord | None:
        """Get a Tool Step by tool_call_id"""
        steps = await self.get_steps(session_id)
        for step in steps:
            if step.tool_call_id == tool_call_id:
                return step
        return None

    # --- Compact Metadata Operations ---

    async def save_compact_metadata(
        self, session_id: str, agent_id: str, metadata: CompactMetadata
    ) -> None:
        """Save compact metadata (append to history). Override in subclasses."""

    async def get_latest_compact_metadata(
        self, session_id: str, agent_id: str
    ) -> CompactMetadata | None:
        """Get the most recent compact metadata."""
        entries = await self.list_entries(
            session_id=session_id,
            agent_id=agent_id,
            limit=100_000,
        )
        compact_entries = [
            entry for entry in entries if isinstance(entry, CompactionApplied)
        ]
        if compact_entries:
            return build_compact_metadata_from_entry(compact_entries[-1])
        return None

    async def get_compact_history(
        self, session_id: str, agent_id: str
    ) -> list[CompactMetadata]:
        """Get all compact metadata history (sorted by created_at ascending)."""
        entries = await self.list_entries(
            session_id=session_id,
            agent_id=agent_id,
            limit=100_000,
        )
        compact_entries = [
            entry for entry in entries if isinstance(entry, CompactionApplied)
        ]
        return [build_compact_metadata_from_entry(entry) for entry in compact_entries]


class InMemoryRunStepStorage(RunStepStorage):
    """
    In-memory implementation (for testing and development)
    """

    def __init__(self) -> None:
        self.runs: dict[str, Run] = {}
        self.steps: dict[str, list[StepRecord]] = {}  # session_id -> list[StepRecord]
        self.run_log_entries: dict[str, list[RunLogEntry]] = {}
        self._id_index: dict[
            str, dict[str, int]
        ] = {}  # session_id -> {step_id -> list_index}
        self._seq_index: dict[
            str, dict[int, int]
        ] = {}  # session_id -> {sequence -> list_index}
        self._sequence_counters: dict[str, int] = {}  # session_id -> counter
        self._sequence_locks: dict[str, asyncio.Lock] = {}  # session_id -> lock
        self._compact_history: dict[str, list[CompactMetadata]] = {}
        self._compact_lock = asyncio.Lock()

    async def append_entries(self, entries: list[RunLogEntry]) -> None:
        for entry in entries:
            bucket = self.run_log_entries.setdefault(entry.session_id, [])
            bucket.append(entry)
            bucket.sort(key=lambda item: item.sequence)

    def _has_committed_step_entries(self, session_id: str) -> bool:
        return any(
            isinstance(
                entry,
                (UserStepCommitted, AssistantStepCommitted, ToolStepCommitted),
            )
            for entry in self.run_log_entries.get(session_id, [])
        )

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

    async def save_run(self, run: Run) -> None:
        self.runs[run.id] = run

    async def get_run(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)

    async def get_run_view(self, run_id: str) -> RunView | None:
        for entries in self.run_log_entries.values():
            run_entries = [entry for entry in entries if entry.run_id == run_id]
            if run_entries:
                return build_run_view_from_entries(run_entries)
        return await super().get_run_view(run_id)

    async def list_runs(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Run]:
        runs = list(self.runs.values())

        if user_id:
            runs = [r for r in runs if r.user_id == user_id]

        if session_id:
            runs = [r for r in runs if r.session_id == session_id]

        runs.sort(key=lambda r: r.created_at, reverse=True)

        return runs[offset : offset + limit]

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
        if not entries:
            return await super().list_run_views(
                user_id=user_id,
                session_id=session_id,
                limit=limit,
                offset=offset,
            )
        run_views = build_run_views_from_entries(entries)
        if user_id is not None:
            run_views = [view for view in run_views if view.user_id == user_id]
        return run_views[offset : offset + limit]

    async def batch_count_run_views(self, session_ids: list[str]) -> dict[str, int]:
        result: dict[str, int] = {sid: 0 for sid in session_ids}
        for sid in session_ids:
            count = len(build_run_views_from_entries(self.run_log_entries.get(sid, [])))
            result[sid] = count if count > 0 else await self.count_runs(sid)
        return result

    async def count_runs(self, session_id: str) -> int:
        return sum(1 for r in self.runs.values() if r.session_id == session_id)

    async def delete_run(self, run_id: str) -> None:
        run = self.runs.pop(run_id, None)
        if run is None:
            return
        session_steps = self.steps.get(run.session_id, [])
        self.steps[run.session_id] = [
            step for step in session_steps if step.run_id != run_id
        ]
        self._rebuild_indexes(run.session_id)

    # --- Step Operations ---

    def _rebuild_indexes(self, session_id: str) -> None:
        """Rebuild id/seq indexes from the steps list."""
        id_idx: dict[str, int] = {}
        seq_idx: dict[int, int] = {}
        for i, s in enumerate(self.steps.get(session_id, [])):
            id_idx[s.id] = i
            seq_idx[s.sequence] = i
        self._id_index[session_id] = id_idx
        self._seq_index[session_id] = seq_idx

    async def save_step(self, step: StepRecord) -> None:
        """
        Save or update a step.

        Handles idempotency: if a step with same (session_id, sequence) exists,
        updates it instead of creating a duplicate.
        """
        sid = step.session_id
        if sid not in self.steps:
            self.steps[sid] = []
            self._id_index[sid] = {}
            self._seq_index[sid] = {}

        id_idx = self._id_index[sid]
        seq_idx = self._seq_index[sid]
        step_list = self.steps[sid]

        # O(1) lookup by step.id
        if step.id in id_idx:
            idx = id_idx[step.id]
            old = step_list[idx]
            if old.sequence != step.sequence:
                step_list.pop(idx)
                self._rebuild_indexes(sid)
                self._insert_new_step(sid, step)
            else:
                step_list[idx] = step
            return

        # O(1) lookup by sequence
        if step.sequence in seq_idx:
            idx = seq_idx[step.sequence]
            old = step_list[idx]
            del id_idx[old.id]
            step_list[idx] = step
            id_idx[step.id] = idx
            return

        self._insert_new_step(sid, step)

    def _insert_new_step(self, sid: str, step: StepRecord) -> None:
        """Bisect-insert a brand-new step with incremental index update."""
        step_list = self.steps[sid]
        id_idx = self._id_index[sid]
        seq_idx = self._seq_index[sid]

        sequences = [s.sequence for s in step_list]
        pos = bisect.bisect_left(sequences, step.sequence)
        step_list.insert(pos, step)

        for existing_id, existing_pos in list(id_idx.items()):
            if existing_pos >= pos:
                id_idx[existing_id] = existing_pos + 1
        for existing_seq, existing_pos in list(seq_idx.items()):
            if existing_pos >= pos:
                seq_idx[existing_seq] = existing_pos + 1

        id_idx[step.id] = pos
        seq_idx[step.sequence] = pos

        current = self._sequence_counters.get(sid)
        if current is not None and step.sequence > current:
            self._sequence_counters[sid] = step.sequence

    async def save_steps_batch(self, steps: list[StepRecord]) -> None:
        for step in steps:
            await self.save_step(step)

    async def get_steps(
        self,
        session_id: str,
        start_seq: int | None = None,
        end_seq: int | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 1000,
    ) -> list[StepRecord]:
        if self._has_committed_step_entries(session_id):
            step_views = await self.list_step_views(
                session_id=session_id,
                start_seq=start_seq,
                end_seq=end_seq,
                run_id=run_id,
                agent_id=agent_id,
                limit=limit,
            )
            return [build_step_record_from_view(step) for step in step_views]

        steps = self.steps.get(session_id, [])

        if start_seq is not None:
            steps = [s for s in steps if s.sequence >= start_seq]
        if end_seq is not None:
            steps = [s for s in steps if s.sequence <= end_seq]
        if run_id is not None:
            steps = [s for s in steps if s.run_id == run_id]
        if agent_id is not None:
            steps = [s for s in steps if s.agent_id == agent_id]

        return steps[:limit]

    async def get_last_step(self, session_id: str) -> StepRecord | None:
        if self._has_committed_step_entries(session_id):
            steps = await self.get_steps(session_id, limit=1_000_000)
            return steps[-1] if steps else None
        steps = self.steps.get(session_id, [])
        return steps[-1] if steps else None

    async def delete_steps(self, session_id: str, start_seq: int) -> int:
        if session_id not in self.steps:
            return 0

        original_count = len(self.steps[session_id])
        self.steps[session_id] = [
            s for s in self.steps[session_id] if s.sequence < start_seq
        ]
        self._rebuild_indexes(session_id)
        return original_count - len(self.steps[session_id])

    # --- Step Content Update (for retrospect) ---

    async def update_step_condensed_content(
        self,
        session_id: str,
        step_id: str,
        condensed_content: str,
    ) -> bool:
        bucket = self.run_log_entries.get(session_id, [])
        for index, entry in enumerate(bucket):
            if not isinstance(
                entry,
                (UserStepCommitted, AssistantStepCommitted, ToolStepCommitted),
            ):
                continue
            if entry.step_id != step_id:
                continue
            bucket[index] = type(entry)(
                sequence=entry.sequence,
                session_id=entry.session_id,
                run_id=entry.run_id,
                agent_id=entry.agent_id,
                step_id=entry.step_id,
                role=entry.role,
                content=entry.content,
                content_for_user=entry.content_for_user,
                reasoning_content=entry.reasoning_content,
                user_input=entry.user_input,
                tool_calls=entry.tool_calls,
                tool_call_id=entry.tool_call_id,
                name=entry.name,
                metrics=entry.metrics,
                condensed_content=condensed_content,
                parent_run_id=entry.parent_run_id,
                depth=entry.depth,
                created_at=entry.created_at,
                **(
                    {"is_error": entry.is_error}
                    if isinstance(entry, ToolStepCommitted)
                    else {}
                ),
            )
            return True
        id_idx = self._id_index.get(session_id, {})
        idx = id_idx.get(step_id)
        if idx is None:
            return False
        step = self.steps[session_id][idx]
        step.condensed_content = condensed_content
        return True

    async def get_step_count(self, session_id: str) -> int:
        if self._has_committed_step_entries(session_id):
            return await self.get_committed_step_count(session_id)
        return len(self.steps.get(session_id, []))

    async def get_max_sequence(self, session_id: str) -> int:
        if self.run_log_entries.get(session_id):
            entries = self.run_log_entries.get(session_id, [])
            return max((entry.sequence for entry in entries), default=0)
        steps = self.steps.get(session_id, [])
        if not steps:
            return 0
        return max(s.sequence for s in steps)

    async def allocate_sequence(self, session_id: str) -> int:
        """Atomically allocate next sequence number."""
        # Initialize lock for this session if not exists
        if session_id not in self._sequence_locks:
            self._sequence_locks[session_id] = asyncio.Lock()

        async with self._sequence_locks[session_id]:
            # Initialize counter from existing steps if not yet initialized
            if session_id not in self._sequence_counters:
                max_seq = await self.get_max_sequence(session_id)
                self._sequence_counters[session_id] = max_seq

            # Increment and return
            self._sequence_counters[session_id] += 1
            return self._sequence_counters[session_id]

    # --- Compact Metadata ---

    def _compact_key(self, session_id: str, agent_id: str) -> str:
        return f"{session_id}:{agent_id}"

    async def save_compact_metadata(
        self, session_id: str, agent_id: str, metadata: CompactMetadata
    ) -> None:
        async with self._compact_lock:
            key = self._compact_key(session_id, agent_id)
            if key not in self._compact_history:
                self._compact_history[key] = []
            self._compact_history[key].append(metadata)

    async def get_latest_compact_metadata(
        self, session_id: str, agent_id: str
    ) -> CompactMetadata | None:
        metadata = await super().get_latest_compact_metadata(session_id, agent_id)
        if metadata is not None:
            return metadata
        key = self._compact_key(session_id, agent_id)
        history = self._compact_history.get(key, [])
        return history[-1] if history else None

    async def get_compact_history(
        self, session_id: str, agent_id: str
    ) -> list[CompactMetadata]:
        history = await super().get_compact_history(session_id, agent_id)
        if history:
            return history
        key = self._compact_key(session_id, agent_id)
        return list(self._compact_history.get(key, []))


class InMemoryRunLogStorage(InMemoryRunStepStorage):
    """Dedicated run-log storage alias for new runtime components."""


__all__ = [
    "InMemoryRunLogStorage",
    "InMemoryRunStepStorage",
    "RunLogStorage",
    "RunStepStorage",
]
