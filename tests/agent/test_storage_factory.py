"""
Test cases for StorageFactory.

Tests storage creation and usage for all storage types.
"""

import os
import pytest
import tempfile

from agiwo.agent.options import RunStepStorageConfig, TraceStorageConfig
from agiwo.agent.storage.factory import StorageFactory
from agiwo.agent.storage.base import InMemoryRunStepStorage
from agiwo.agent.storage.sqlite import SQLiteRunStepStorage
from agiwo.observability.collector import InMemoryTraceStorage
from agiwo.observability.sqlite_store import SQLiteTraceStorage


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

        from agiwo.agent.schema import Run, RunStatus, RunMetrics
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

            from agiwo.agent.schema import Run, RunStatus, RunMetrics
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
        storage = StorageFactory.create_trace_storage(config)
        assert storage is None

    def test_create_memory_storage(self):
        config = TraceStorageConfig(storage_type="memory")
        storage = StorageFactory.create_trace_storage(config)
        assert isinstance(storage, InMemoryTraceStorage)

    def test_create_sqlite_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            config = TraceStorageConfig(
                storage_type="sqlite",
                config={"db_path": db_path, "collection_name": "test_traces"},
            )
            storage = StorageFactory.create_trace_storage(config)
            assert isinstance(storage, SQLiteTraceStorage)
            assert storage.db_path == db_path

    def test_create_unknown_storage_type(self):
        config = TraceStorageConfig(storage_type="unknown")  # type: ignore
        with pytest.raises(ValueError, match="Unknown trace_storage_type"):
            StorageFactory.create_trace_storage(config)

    @pytest.mark.asyncio
    async def test_memory_storage_works(self):
        config = TraceStorageConfig(storage_type="memory")
        storage = StorageFactory.create_trace_storage(config)

        from agiwo.observability.trace import Trace
        trace = Trace(trace_id="test-trace", agent_id="test-agent")
        await storage.save_trace(trace)

        retrieved = await storage.get_trace("test-trace")
        assert retrieved is not None
        assert retrieved.trace_id == "test-trace"

    @pytest.mark.asyncio
    async def test_sqlite_storage_lazy_connect(self):
        """SQLite trace storage connects lazily on first operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            config = TraceStorageConfig(
                storage_type="sqlite",
                config={"db_path": db_path},
            )
            storage = StorageFactory.create_trace_storage(config)
            assert not storage._initialized

            from agiwo.observability.trace import Trace
            trace = Trace(trace_id="test-trace", agent_id="test-agent")
            await storage.save_trace(trace)

            retrieved = await storage.get_trace("test-trace")
            assert retrieved is not None

            await storage.close()


class TestAgentIntegration:
    """Test Agent integration with storage factory."""

    def test_agent_storage_created_in_init(self):
        """Storage is created synchronously in Agent.__init__."""
        from agiwo import Agent, AgentOptions

        agent = Agent(
            id="test-agent",
            description="Test",
            model=None,  # type: ignore
            options=AgentOptions(),
        )

        assert agent.run_step_storage is not None
        assert isinstance(agent.run_step_storage, InMemoryRunStepStorage)
        assert agent.trace_storage is None  # default: tracing disabled

    def test_agent_with_trace_storage(self):
        """Agent creates trace storage when configured."""
        from agiwo import Agent, AgentOptions

        agent = Agent(
            id="test-agent",
            description="Test",
            model=None,  # type: ignore
            options=AgentOptions(
                trace_storage=TraceStorageConfig(storage_type="memory"),
            ),
        )

        assert agent.trace_storage is not None
        assert isinstance(agent.trace_storage, InMemoryTraceStorage)

    @pytest.mark.asyncio
    async def test_agent_close(self):
        from agiwo import Agent, AgentOptions

        agent = Agent(
            id="test-agent",
            description="Test",
            model=None,  # type: ignore
            options=AgentOptions(),
        )

        assert agent.run_step_storage is not None
        await agent.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
