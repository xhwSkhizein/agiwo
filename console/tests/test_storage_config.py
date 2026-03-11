from agiwo.config.settings import settings as sdk_settings
from server.config import ConsoleConfig
from server.services.storage_manager import (
    StorageManager,
    build_agent_state_storage_config,
    build_citation_store_config,
    build_run_step_storage_config,
    build_trace_storage_config,
)


def _config() -> ConsoleConfig:
    return ConsoleConfig(
        run_step_storage_type="sqlite",
        metadata_storage_type="mongodb",
        mongodb_uri="mongodb://localhost:27017",
        mongodb_db_name="agiwo",
        mongodb_trace_collection="traces",
        trace_storage_type="sqlite",
    )


def test_storage_config_builders_share_console_mapping(monkeypatch) -> None:
    monkeypatch.setattr(sdk_settings, "sqlite_db_path", "/tmp/agiwo.db")
    monkeypatch.setattr(sdk_settings, "trace_collection_name", "trace_spans")
    config = _config()

    run_step = build_run_step_storage_config(config)
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
    assert citation.storage_type == "mongodb"
    assert citation.mongo_uri == "mongodb://localhost:27017"
    assert citation.mongo_db_name == "agiwo"
    assert agent_state.storage_type == "memory"


def test_storage_manager_only_owns_run_step_and_trace_storage() -> None:
    manager = StorageManager(
        ConsoleConfig(
            run_step_storage_type="memory",
            trace_storage_type="memory",
            metadata_storage_type="memory",
        )
    )

    assert hasattr(manager, "run_step_storage")
    assert hasattr(manager, "trace_storage")
    assert not hasattr(manager, "agent_state_storage")
