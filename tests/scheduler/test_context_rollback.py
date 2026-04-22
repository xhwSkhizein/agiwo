"""Tests for scheduler context rollback — signal propagation and gating."""

import pytest

from agiwo.agent import AssistantStepCommitted, MessageRole, UserStepCommitted
from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.run import RunLedger, RunOutput
from agiwo.agent.storage.base import InMemoryRunLogStorage
from agiwo.agent.storage.sqlite import SQLiteRunLogStorage
from agiwo.scheduler.commands import SleepRequest
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    TimeUnit,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.store.sqlite import SQLiteAgentStateStorage


class TestSleepRequestNoProgress:
    def test_default_is_false(self):
        req = SleepRequest(
            agent_id="a1",
            session_id="s1",
            wake_type=WakeType.PERIODIC,
        )
        assert req.no_progress is False

    def test_explicit_true(self):
        req = SleepRequest(
            agent_id="a1",
            session_id="s1",
            wake_type=WakeType.PERIODIC,
            no_progress=True,
        )
        assert req.no_progress is True


class TestAgentStateNoProgress:
    def _base_state(self, **kwargs) -> AgentState:
        return AgentState(
            id="a1",
            session_id="s1",
            status=AgentStateStatus.RUNNING,
            task="root",
            **kwargs,
        )

    def test_defaults(self):
        state = self._base_state()
        assert state.no_progress is False
        assert state.rollback_count == 0

    def test_with_waiting_propagates_no_progress(self):
        state = self._base_state()
        wc = WakeCondition(
            type=WakeType.PERIODIC,
            time_value=60,
            time_unit=TimeUnit.SECONDS,
        )
        waiting = state.with_waiting(wake_condition=wc, no_progress=True)
        assert waiting.no_progress is True
        assert waiting.status == AgentStateStatus.WAITING

    def test_with_running_resets_no_progress(self):
        state = self._base_state(no_progress=True)
        running = state.with_running()
        assert running.no_progress is False

    def test_rollback_count_preserved_through_updates(self):
        state = self._base_state(rollback_count=3)
        updated = state.with_updates(rollback_count=4)
        assert updated.rollback_count == 4


class TestAgentOptionsRollbackConfig:
    def test_enable_context_rollback_default_true(self):
        opts = AgentOptions()
        assert opts.enable_context_rollback is True

    def test_can_disable(self):
        opts = AgentOptions(enable_context_rollback=False)
        assert opts.enable_context_rollback is False


class TestNoProgressPersistence:
    @pytest.mark.asyncio
    async def test_sqlite_round_trip_preserves_no_progress(self, tmp_path):
        store = SQLiteAgentStateStorage(str(tmp_path / "scheduler.db"))
        state = AgentState(
            id="a1",
            session_id="s1",
            status=AgentStateStatus.WAITING,
            task="root",
            no_progress=True,
            wake_condition=WakeCondition(
                type=WakeType.PERIODIC,
                time_value=60,
                time_unit=TimeUnit.SECONDS,
            ),
        )

        await store.save_state(state)
        restored = await store.get_state("a1")

        assert restored is not None
        assert restored.no_progress is True

        await store.close()


class TestRunOutputMetadata:
    def test_metadata_carries_run_start_seq(self):
        output = RunOutput(metadata={"run_start_seq": 42})
        assert output.metadata["run_start_seq"] == 42

    def test_metadata_defaults_to_empty_dict(self):
        output = RunOutput()
        assert output.metadata == {}


class TestRunLedgerRunStartSeq:
    def test_default_is_zero(self):
        ledger = RunLedger()
        assert ledger.run_start_seq == 0

    def test_can_set(self):
        ledger = RunLedger()
        ledger.run_start_seq = 15
        assert ledger.run_start_seq == 15


class TestRunLogRollback:
    @pytest.mark.asyncio
    async def test_append_run_rollback_hides_steps_from_replay(self):
        storage = InMemoryRunLogStorage()
        await storage.append_entries(
            [
                UserStepCommitted(
                    sequence=10,
                    session_id="sess-1",
                    run_id="run-1",
                    agent_id="agent-1",
                    step_id="step-10",
                    role=MessageRole.USER,
                    content="hello",
                    user_input="hello",
                ),
                AssistantStepCommitted(
                    sequence=11,
                    session_id="sess-1",
                    run_id="run-1",
                    agent_id="agent-1",
                    step_id="step-11",
                    role=MessageRole.ASSISTANT,
                    content="world",
                ),
            ]
        )
        await storage.append_run_rollback(
            session_id="sess-1",
            run_id="run-1",
            agent_id="agent-1",
            start_sequence=10,
            end_sequence=11,
            reason="scheduler_no_progress_periodic",
        )

        visible_steps = await storage.list_step_views(session_id="sess-1")
        all_steps = await storage.list_step_views(
            session_id="sess-1",
            include_rolled_back=True,
        )

        assert visible_steps == []
        assert [step.id for step in all_steps] == ["step-10", "step-11"]

    @pytest.mark.asyncio
    async def test_sqlite_step_replay_respects_later_rollback_metadata(self, tmp_path):
        storage = SQLiteRunLogStorage(str(tmp_path / "run-log.db"))
        try:
            await storage.append_entries(
                [
                    UserStepCommitted(
                        sequence=10,
                        session_id="sess-1",
                        run_id="run-1",
                        agent_id="agent-1",
                        step_id="step-10",
                        role=MessageRole.USER,
                        content="hello",
                        user_input="hello",
                    ),
                    AssistantStepCommitted(
                        sequence=11,
                        session_id="sess-1",
                        run_id="run-1",
                        agent_id="agent-1",
                        step_id="step-11",
                        role=MessageRole.ASSISTANT,
                        content="world",
                    ),
                ]
            )
            await storage.append_run_rollback(
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                start_sequence=10,
                end_sequence=11,
                reason="scheduler_no_progress_periodic",
            )

            visible_steps = await storage.list_step_views(
                session_id="sess-1", end_seq=11
            )
            all_steps = await storage.list_step_views(
                session_id="sess-1",
                end_seq=11,
                include_rolled_back=True,
            )

            assert visible_steps == []
            assert [step.id for step in all_steps] == ["step-10", "step-11"]
        finally:
            await storage.close()
