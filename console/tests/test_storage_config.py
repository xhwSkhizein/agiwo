from agiwo.config.settings import settings as sdk_settings
from server.config import ConsoleConfig
from server.services.storage_wiring import (
    build_agent_state_storage_config,
    build_citation_store_config,
    build_run_log_storage_config,
    build_trace_storage_config,
    create_run_log_storage,
    create_trace_storage,
)


def test_storage_config_builders_sqlite(monkeypatch) -> None:
    monkeypatch.setattr(sdk_settings, "sqlite_db_path", "/tmp/agiwo.db")
    monkeypatch.setattr(sdk_settings, "trace_collection_name", "trace_spans")
    config = ConsoleConfig(
        storage={
            "run_log_type": "sqlite",
            "trace_type": "sqlite",
            "metadata_type": "sqlite",
        }
    )

    run_step = build_run_log_storage_config(config)
    trace = build_trace_storage_config(config)
    citation = build_citation_store_config(config)
    agent_state = build_agent_state_storage_config(config)

    assert run_step.storage_type == "sqlite"
    assert run_step.config == {"db_path": "/tmp/agiwo.db"}
    assert trace.storage_type == "sqlite"
    assert trace.config == {
        "db_path": "/tmp/agiwo.db",
        "collection_name": "trace_spans",
    }
    assert citation.storage_type == "sqlite"
    assert citation.sqlite_db_path == "/tmp/agiwo.db"
    assert agent_state.storage_type == "sqlite"


def test_storage_factory_functions_create_correct_storage_types() -> None:
    config = ConsoleConfig(
        storage={
            "run_log_type": "memory",
            "trace_type": "memory",
            "metadata_type": "memory",
        }
    )
    run_step = create_run_log_storage(config)
    trace = create_trace_storage(config)

    assert run_step is not None
    assert trace is not None


def test_all_memory_storage_types() -> None:
    config = ConsoleConfig(
        storage={
            "run_log_type": "memory",
            "trace_type": "memory",
            "metadata_type": "memory",
        }
    )
    run_step = build_run_log_storage_config(config)
    trace = build_trace_storage_config(config)
    agent_state = build_agent_state_storage_config(config)
    citation = build_citation_store_config(config)

    assert run_step.storage_type == "memory"
    assert trace.storage_type == "memory"
    assert agent_state.storage_type == "memory"
    assert citation.storage_type == "memory"


def test_agent_state_sqlite_config(monkeypatch) -> None:
    monkeypatch.setattr(sdk_settings, "sqlite_db_path", "/tmp/test.db")
    config = ConsoleConfig(storage={"metadata_type": "sqlite"})
    result = build_agent_state_storage_config(config)
    assert result.storage_type == "sqlite"
    assert result.config == {"db_path": "/tmp/test.db"}


def test_citation_sqlite_config(monkeypatch) -> None:
    monkeypatch.setattr(sdk_settings, "sqlite_db_path", "/tmp/test.db")
    config = ConsoleConfig(storage={"metadata_type": "sqlite"})
    result = build_citation_store_config(config)
    assert result.storage_type == "sqlite"
    assert result.sqlite_db_path == "/tmp/test.db"
