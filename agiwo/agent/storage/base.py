"""
Repository interface and in-memory implementation.
"""

import asyncio
from abc import ABC, abstractmethod

from agiwo.agent.schema import Run, StepRecord


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


class InMemoryRunStepStorage(RunStepStorage):
    """
    In-memory implementation (for testing and development)
    """

    def __init__(self) -> None:
        self.runs: dict[str, Run] = {}
        self.steps: dict[str, list[StepRecord]] = {}  # session_id -> list[StepRecord]
        self._sequence_counters: dict[str, int] = {}  # session_id -> counter
        self._sequence_locks: dict[str, asyncio.Lock] = {}  # session_id -> lock

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
        if run_id in self.runs:
            del self.runs[run_id]

    # --- Step Operations ---

    async def save_step(self, step: StepRecord) -> None:
        """
        Save or update a step.

        Handles idempotency: if a step with same (session_id, sequence) exists,
        updates it instead of creating a duplicate.
        """
        if step.session_id not in self.steps:
            self.steps[step.session_id] = []

        # First, try to find by step.id
        existing_idx = None
        for i, s in enumerate(self.steps[step.session_id]):
            if s.id == step.id:
                existing_idx = i
                break

        if existing_idx is not None:
            # Update existing step by id
            self.steps[step.session_id][existing_idx] = step
        else:
            # Check if step with same (session_id, sequence) exists
            seq_existing_idx = None
            for i, s in enumerate(self.steps[step.session_id]):
                if s.session_id == step.session_id and s.sequence == step.sequence:
                    seq_existing_idx = i
                    break

            if seq_existing_idx is not None:
                # Update existing step by (session_id, sequence)
                self.steps[step.session_id][seq_existing_idx] = step
            else:
                # Insert new step
                self.steps[step.session_id].append(step)

            # Keep steps sorted by sequence
            self.steps[step.session_id].sort(key=lambda s: s.sequence)

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


__all__ = ["RunStepStorage", "InMemoryRunStepStorage"]
