"""
Repository interface and in-memory implementation.
"""

import asyncio
import bisect
from abc import ABC, abstractmethod

from agiwo.agent.models.run import CompactMetadata
from agiwo.agent.models.run import Run
from agiwo.agent.models.step import StepRecord


class RunStepStorage(ABC):
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
        return None

    async def get_compact_history(
        self, session_id: str, agent_id: str
    ) -> list[CompactMetadata]:
        """Get all compact metadata history (sorted by created_at ascending)."""
        return []


class InMemoryRunStepStorage(RunStepStorage):
    """
    In-memory implementation (for testing and development)
    """

    def __init__(self) -> None:
        self.runs: dict[str, Run] = {}
        self.steps: dict[str, list[StepRecord]] = {}  # session_id -> list[StepRecord]
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

    async def save_run(self, run: Run) -> None:
        self.runs[run.id] = run

    async def get_run(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)

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

    async def get_step_count(self, session_id: str) -> int:
        return len(self.steps.get(session_id, []))

    async def get_max_sequence(self, session_id: str) -> int:
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
        key = self._compact_key(session_id, agent_id)
        history = self._compact_history.get(key, [])
        return history[-1] if history else None

    async def get_compact_history(
        self, session_id: str, agent_id: str
    ) -> list[CompactMetadata]:
        key = self._compact_key(session_id, agent_id)
        return list(self._compact_history.get(key, []))


__all__ = ["RunStepStorage", "InMemoryRunStepStorage"]
