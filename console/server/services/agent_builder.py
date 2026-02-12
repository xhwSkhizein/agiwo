"""
Agent building utilities — shared between chat and scheduler_chat routers.
"""

import json
from dataclasses import asdict
from typing import Any

from agiwo.agent.agent import Agent
from agiwo.agent.options import AgentOptions, RunStepStorageConfig, TraceStorageConfig
from agiwo.agent.schema import StreamEvent
from agiwo.llm.anthropic import AnthropicModel
from agiwo.llm.deepseek import DeepseekModel
from agiwo.llm.openai import OpenAIModel
from agiwo.tool.agent_tool import AgentTool
from agiwo.tool.base import BaseTool

from server.config import ConsoleConfig
from server.services.agent_registry import AgentConfigRecord, AgentRegistry
from server.tools import create_tools

MODEL_PROVIDER_MAP: dict[str, type] = {
    "openai": OpenAIModel,
    "deepseek": DeepseekModel,
    "anthropic": AnthropicModel,
}

AGENT_TOOL_PREFIX = "agent:"


def build_model(config: AgentConfigRecord) -> Any:
    """Build a Model instance from agent config."""
    provider_cls = MODEL_PROVIDER_MAP.get(config.model_provider)
    if provider_cls is None:
        raise ValueError(f"Unknown model provider: {config.model_provider}")

    params: dict[str, Any] = {
        "id": config.model_name,
        "name": config.model_name,
    }
    params.update(config.model_params)

    return provider_cls(**params)


def build_agent_options(config: AgentConfigRecord, console_config: ConsoleConfig) -> AgentOptions:
    """Build AgentOptions with storage config matching the console storage backend."""
    opts = config.options or {}

    if console_config.storage_type == "sqlite":
        run_step_cfg = RunStepStorageConfig(
            storage_type="sqlite",
            config={"db_path": console_config.sqlite_db_path},
        )
        trace_cfg = TraceStorageConfig(
            storage_type="sqlite",
            config={
                "db_path": console_config.sqlite_db_path,
                "collection_name": console_config.sqlite_trace_collection,
            },
        )
    else:
        run_step_cfg = RunStepStorageConfig(
            storage_type="mongodb",
            config={
                "uri": console_config.mongodb_uri,
                "db_name": console_config.mongodb_db_name,
            },
        )
        trace_cfg = TraceStorageConfig(
            storage_type="mongodb",
            config={
                "mongo_uri": console_config.mongodb_uri,
                "db_name": console_config.mongodb_db_name,
                "collection_name": console_config.mongodb_trace_collection,
            },
        )

    return AgentOptions(
        max_steps=opts.get("max_steps", 10),
        run_timeout=opts.get("run_timeout", 600),
        max_output_tokens=opts.get("max_output_tokens", 8196),
        run_step_storage=run_step_cfg,
        trace_storage=trace_cfg,
    )


def serialize_event(event: StreamEvent) -> str:
    """Serialize a StreamEvent to JSON string for SSE data field."""
    data: dict[str, Any] = {
        "type": event.type.value,
        "run_id": event.run_id,
    }
    if event.delta is not None:
        data["delta"] = asdict(event.delta)
    if event.step is not None:
        step_dict = asdict(event.step)
        step_dict["role"] = event.step.role.value if hasattr(event.step.role, "value") else str(event.step.role)
        if event.step.metrics:
            step_dict["metrics"] = asdict(event.step.metrics)
        data["step"] = step_dict
    if event.data is not None:
        data["data"] = event.data
    if event.agent_id is not None:
        data["agent_id"] = event.agent_id
    if event.span_id is not None:
        data["span_id"] = event.span_id
    return json.dumps(data, default=str)


async def build_agent(
    config: AgentConfigRecord,
    console_config: ConsoleConfig,
    registry: AgentRegistry,
    _building: set[str] | None = None,
) -> Agent:
    """Build an Agent instance from persisted config.

    Supports agent-as-tool references in the tools list via "agent:<agent_id>" syntax.
    The _building set prevents circular agent references.
    """
    if _building is None:
        _building = set()
    if config.id in _building:
        raise ValueError(f"Circular agent reference detected: {config.id}")
    _building.add(config.id)

    model = build_model(config)
    options = build_agent_options(config, console_config)

    builtin_names: list[str] = []
    agent_refs: list[str] = []
    for t in config.tools or []:
        if t.startswith(AGENT_TOOL_PREFIX):
            agent_refs.append(t[len(AGENT_TOOL_PREFIX):])
        else:
            builtin_names.append(t)

    tools: list[BaseTool] = create_tools(builtin_names)

    for ref_id in agent_refs:
        child_config = await registry.get_agent(ref_id)
        if child_config is None:
            continue
        child_agent = await build_agent(child_config, console_config, registry, _building.copy())
        tools.append(AgentTool(child_agent))

    return Agent(
        id=config.id,
        description=config.description,
        model=model,
        tools=tools or None,
        system_prompt=config.system_prompt,
        options=options,
    )
