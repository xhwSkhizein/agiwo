"""
Test cases for StorageFactory.

Tests storage creation and usage for all storage types.
"""

from datetime import datetime, timedelta
import os
import pytest
import tempfile

from agiwo import Agent, AgentConfig, AgentOptions
from agiwo.agent import Run, RunMetrics, RunStatus
from agiwo.agent.options import RunStepStorageConfig, TraceStorageConfig
from agiwo.agent.storage.factory import StorageFactory
from agiwo.agent.storage.base import InMemoryRunStepStorage
from agiwo.agent.storage.sqlite import SQLiteRunStepStorage
from agiwo.observability.factory import create_trace_storage
from agiwo.observability.memory_store import InMemoryTraceStorage
from agiwo.observability.sqlite_store import SQLiteTraceStorage
from agiwo.observability.trace import SpanStatus, Trace


class TestRunStepStorageFactory:
    """Test RunStepStorage creation."""

    def test_create_memory_storage(self):
        config = RunStepStorageConfig(storage_type="memory")
        storage = StorageFactory.create_run_step_storage(config)
        assert isinstance(storage, InMemoryRunStepStorage)

    def test_create_sqlite_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            config = RunStepStorageConfig(
                storage_type="sqlite",
                config={"db_path": db_path},
            )
            storage = StorageFactory.create_run_step_storage(config)
            assert isinstance(storage, SQLiteRunStepStorage)
            assert storage.db_path == db_path

    def test_create_unknown_storage_type(self):
        config = RunStepStorageConfig(storage_type="unknown")  # type: ignore
        with pytest.raises(ValueError, match="Unknown run_step_storage_type"):
            StorageFactory.create_run_step_storage(config)

    @pytest.mark.asyncio
    async def test_memory_storage_works(self):
        config = RunStepStorageConfig(storage_type="memory")
        storage = StorageFactory.create_run_step_storage(config)

        run = Run(
            id="test-run",
            agent_id="test-agent",
            session_id="test-session",
            user_input="test",
            status=RunStatus.COMPLETED,
        )
        run.metrics = RunMetrics()
        await storage.save_run(run)

        retrieved = await storage.get_run("test-run")
        assert retrieved is not None
        assert retrieved.id == "test-run"

    @pytest.mark.asyncio
    async def test_sqlite_storage_lazy_connect(self):
        """SQLite storage connects lazily on first operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            config = RunStepStorageConfig(
                storage_type="sqlite",
                config={"db_path": db_path},
            )
            storage = StorageFactory.create_run_step_storage(config)
            assert not storage._initialized

            run = Run(
                id="test-run",
                agent_id="test-agent",
                session_id="test-session",
                user_input="test",
                status=RunStatus.COMPLETED,
            )
            run.metrics = RunMetrics()
            await storage.save_run(run)
            assert storage._initialized

            retrieved = await storage.get_run("test-run")
            assert retrieved is not None

            await storage.close()
            assert not storage._initialized


class TestTraceStorageFactory:
    """Test TraceStorage creation."""

    def test_create_none_storage(self):
        config = TraceStorageConfig(storage_type=None)
        storage = create_trace_storage(config)
        assert storage is None

    def test_create_memory_storage(self):
        config = TraceStorageConfig(storage_type="memory")
        storage = create_trace_storage(config)
        assert isinstance(storage, InMemoryTraceStorage)

    def test_create_sqlite_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            config = TraceStorageConfig(
                storage_type="sqlite",
                config={"db_path": db_path, "collection_name": "test_traces"},
            )
            storage = create_trace_storage(config)
            assert isinstance(storage, SQLiteTraceStorage)
            assert storage.db_path == db_path

    def test_create_unknown_storage_type(self):
        config = TraceStorageConfig(storage_type="unknown")  # type: ignore
        with pytest.raises(ValueError, match="Unknown trace_storage_type"):
            create_trace_storage(config)

    @pytest.mark.asyncio
    async def test_memory_storage_works(self):
        config = TraceStorageConfig(storage_type="memory")
        storage = create_trace_storage(config)

        trace = Trace(trace_id="test-trace", agent_id="test-agent")
        await storage.save_trace(trace)

        retrieved = await storage.get_trace("test-trace")
        assert retrieved is not None
        assert retrieved.trace_id == "test-trace"

    @pytest.mark.asyncio
    async def test_memory_storage_query_and_recent_follow_shared_contract(self):
        config = TraceStorageConfig(storage_type="memory")
        storage = create_trace_storage(config)

        base_time = datetime(2026, 3, 9, 10, 0, 0)
        traces = [
            Trace(
                trace_id="trace-1",
                agent_id="agent-a",
                session_id="session-1",
                user_id="user-1",
                status=SpanStatus.OK,
                start_time=base_time,
                duration_ms=100.0,
            ),
            Trace(
                trace_id="trace-2",
                agent_id="agent-a",
                session_id="session-1",
                user_id="user-1",
                status=SpanStatus.OK,
                start_time=base_time + timedelta(seconds=1),
                duration_ms=200.0,
            ),
            Trace(
                trace_id="trace-3",
                agent_id="agent-b",
                session_id="session-2",
                user_id="user-2",
                status=SpanStatus.ERROR,
                start_time=base_time + timedelta(seconds=2),
                duration_ms=300.0,
            ),
        ]
        for trace in traces:
            await storage.save_trace(trace)

        recent = await storage.query_traces({"limit": 2})
        assert [trace.trace_id for trace in recent] == ["trace-3", "trace-2"]

        filtered = await storage.query_traces(
            {
                "agent_id": "agent-a",
                "start_time": base_time + timedelta(milliseconds=500),
                "min_duration_ms": 150.0,
                "limit": 10,
                "offset": 0,
            }
        )
        assert [trace.trace_id for trace in filtered] == ["trace-2"]

        paged = await storage.query_traces({"limit": 1, "offset": 1})
        assert [trace.trace_id for trace in paged] == ["trace-2"]

    @pytest.mark.asyncio
    async def test_sqlite_storage_lazy_connect(self):
        """SQLite trace storage connects lazily on first operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            config = TraceStorageConfig(
                storage_type="sqlite",
                config={"db_path": db_path},
            )
            storage = create_trace_storage(config)
            assert not storage._initialized

            trace = Trace(trace_id="test-trace", agent_id="test-agent")
            await storage.save_trace(trace)

            retrieved = await storage.get_trace("test-trace")
            assert retrieved is not None

            await storage.close()

    @pytest.mark.asyncio
    async def test_sqlite_storage_query_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            config = TraceStorageConfig(
                storage_type="sqlite",
                config={"db_path": db_path, "collection_name": "test_traces"},
            )
            storage = create_trace_storage(config)

            base_time = datetime(2026, 3, 9, 10, 0, 0)
            for idx in range(3):
                await storage.save_trace(
                    Trace(
                        trace_id=f"trace-{idx}",
                        status=SpanStatus.OK,
                        start_time=base_time + timedelta(seconds=idx),
                    )
                )

            recent = await storage.query_traces({"limit": 2})
            assert [trace.trace_id for trace in recent] == ["trace-2", "trace-1"]

            await storage.close()


class TestAgentIntegration:
    """Test Agent integration with storage factory."""

    def test_agent_storage_created_in_init(self):
        """Storage is created synchronously in Agent.__init__."""
        agent = Agent(
            AgentConfig(
                name="test-agent",
                description="Test",
                options=AgentOptions(),
            ),
            model=None,  # type: ignore
        )

        assert agent.run_step_storage is not None
        assert isinstance(agent.run_step_storage, InMemoryRunStepStorage)
        assert agent.trace_storage is None  # default: tracing disabled
        assert not hasattr(agent, "system_prompt")

    def test_agent_with_trace_storage(self):
        """Agent creates trace storage when configured."""
        agent = Agent(
            AgentConfig(
                name="test-agent",
                description="Test",
                options=AgentOptions(
                    trace_storage=TraceStorageConfig(storage_type="memory"),
                ),
            ),
            model=None,  # type: ignore
        )

        assert agent.trace_storage is not None
        assert isinstance(agent.trace_storage, InMemoryTraceStorage)

    @pytest.mark.asyncio
    async def test_agent_close(self):
        agent = Agent(
            AgentConfig(
                name="test-agent",
                description="Test",
                options=AgentOptions(),
            ),
            model=None,  # type: ignore
        )

        assert agent.run_step_storage is not None
        await agent.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
