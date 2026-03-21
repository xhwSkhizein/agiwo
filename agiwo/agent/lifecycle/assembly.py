from agiwo.agent.config import AgentConfig
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.lifecycle.definition import AgentDefinitionRuntime
from agiwo.agent.lifecycle.resource_owner import AgentResourceOwner
from agiwo.agent.runtime_tools import RuntimeToolLike
from agiwo.agent.storage.factory import StorageFactory
from agiwo.observability.factory import create_trace_storage


def build_agent_definition_runtime(
    *,
    config: AgentConfig,
    agent_id: str,
    provided_tools: list[RuntimeToolLike],
    hooks: AgentHooks | None,
) -> AgentDefinitionRuntime:
    return AgentDefinitionRuntime(
        config=config,
        agent_id=agent_id,
        provided_tools=provided_tools,
        hooks=hooks,
    )


def build_agent_resource_owner(*, config: AgentConfig) -> AgentResourceOwner:
    return AgentResourceOwner(
        run_step_storage=StorageFactory.create_run_step_storage(
            config.options.run_step_storage
        ),
        trace_storage=create_trace_storage(config.options.trace_storage),
        session_storage=StorageFactory.create_session_storage(
            config.options.run_step_storage
        ),
    )


__all__ = ["build_agent_definition_runtime", "build_agent_resource_owner"]
