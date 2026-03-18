"""Console services layer — agent lifecycle, registry, and storage wiring."""

from server.services.agent_lifecycle import (
    PersistentAgentNotFoundError,
    PersistentAgentResumeError,
    PersistentAgentValidationError,
    build_agent,
    build_agent_options,
    build_default_agent_options,
    build_model,
    rehydrate_agent,
    resume_persistent_agent,
)
from server.services.agent_registry import AgentConfigRecord, AgentRegistry
from server.services.storage_wiring import (
    build_agent_state_storage_config,
    build_citation_store_config,
    build_run_step_storage_config,
    build_trace_storage_config,
    create_run_step_storage,
    create_trace_storage,
)

__all__ = [
    # agent_lifecycle
    "build_agent",
    "build_model",
    "build_agent_options",
    "build_default_agent_options",
    "rehydrate_agent",
    "resume_persistent_agent",
    "PersistentAgentNotFoundError",
    "PersistentAgentResumeError",
    "PersistentAgentValidationError",
    # agent_registry
    "AgentRegistry",
    "AgentConfigRecord",
    # storage_wiring
    "build_run_step_storage_config",
    "build_trace_storage_config",
    "build_agent_state_storage_config",
    "build_citation_store_config",
    "create_run_step_storage",
    "create_trace_storage",
]
