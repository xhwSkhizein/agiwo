"""
Agent building utilities — shared between chat and scheduler_chat routers.
"""

import json
import os
from dataclasses import asdict
from typing import Any

from pydantic import SecretStr

from agiwo.agent.agent import Agent
from agiwo.agent.options import AgentOptions, RunStepStorageConfig, TraceStorageConfig, normalize_skills_dirs
from agiwo.agent.schema import StreamEvent
from agiwo.config.settings import settings
from agiwo.llm.anthropic import AnthropicModel
from agiwo.llm.bedrock_anthropic import BedrockAnthropicModel
from agiwo.llm.deepseek import DeepseekModel
from agiwo.llm.openai import OpenAIModel
from agiwo.tool.agent_tool import AgentTool
from agiwo.tool.base import BaseTool
from agiwo.tool.storage.citation import (
    InMemoryCitationStore,
    MongoCitationStore,
    SQLiteCitationStore,
)

from server.config import ConsoleConfig
from server.services.agent_registry import AgentConfigRecord, AgentRegistry
from server.tools import create_tools

MODEL_PROVIDER_MAP: dict[str, type] = {
    "openai": OpenAIModel,
    "deepseek": DeepseekModel,
    "anthropic": AnthropicModel,
    "bedrock-anthropic": BedrockAnthropicModel,
    "generic": OpenAIModel,  # Generic OpenAI-compatible API
    "anthropic-generic": AnthropicModel,  # Generic Anthropic-compatible API
}

AGENT_TOOL_PREFIX = "agent:"


def _create_citation_store(console_config: ConsoleConfig) -> Any:
    """Create citation store based on console metadata storage config."""
    if console_config.metadata_storage_type == "sqlite":
        return SQLiteCitationStore(db_path=console_config.sqlite_db_path)
    if console_config.metadata_storage_type == "memory":
        return InMemoryCitationStore()
    return MongoCitationStore(
        uri=console_config.mongodb_uri,
        db_name=console_config.mongodb_db_name,
        collection_name="citation_sources",
    )


def build_model(config: AgentConfigRecord) -> Any:
    """Build a Model instance from agent config."""
    provider_cls = MODEL_PROVIDER_MAP.get(config.model_provider)
    if provider_cls is None:
        raise ValueError(f"Unknown model provider: {config.model_provider}")

    params: dict[str, Any] = {
        "id": config.model_name,
        "name": config.model_name,
    }
    model_params = dict(config.model_params or {})
    if "max_output_tokens_per_call" in model_params:
        model_params["max_tokens"] = model_params.pop("max_output_tokens_per_call")
    else:
        model_params.pop("max_tokens", None)
    params.update(model_params)

    return provider_cls(**params)


def build_agent_options(config: AgentConfigRecord, console_config: ConsoleConfig) -> AgentOptions:
    """Build AgentOptions with storage config matching the console storage backend."""
    opts = config.options or {}
    skills_dirs = normalize_skills_dirs(opts.get("skills_dirs"))
    if skills_dirs is None:
        skills_dirs = normalize_skills_dirs(opts.get("skills_dir"))

    # Run step storage config
    if console_config.run_step_storage_type == "sqlite":
        run_step_cfg = RunStepStorageConfig(
            storage_type="sqlite",
            config={"db_path": console_config.sqlite_db_path},
        )
    elif console_config.run_step_storage_type == "memory":
        run_step_cfg = RunStepStorageConfig(storage_type="memory")
    else:
        run_step_cfg = RunStepStorageConfig(
            storage_type="mongodb",
            config={
                "uri": console_config.mongodb_uri,
                "db_name": console_config.mongodb_db_name,
            },
        )

    # Trace storage config
    effective_trace_type = console_config.effective_trace_storage_type
    if effective_trace_type == "memory":
        trace_cfg = TraceStorageConfig(storage_type="memory")
    elif effective_trace_type == "sqlite":
        trace_cfg = TraceStorageConfig(
            storage_type="sqlite",
            config={
                "db_path": console_config.sqlite_db_path,
                "collection_name": console_config.sqlite_trace_collection,
            },
        )
    else:
        trace_cfg = TraceStorageConfig(
            storage_type="mongodb",
            config={
                "mongo_uri": console_config.mongodb_uri,
                "db_name": console_config.mongodb_db_name,
                "collection_name": console_config.mongodb_trace_collection,
            },
        )

    return AgentOptions(
        config_root=opts.get("config_root", ""),
        max_steps=opts.get("max_steps", 10),
        run_timeout=opts.get("run_timeout", 600),
        max_context_window_tokens=opts.get("max_context_window_tokens", 32768),
        max_tokens_per_run=opts.get("max_tokens_per_run", 131072),
        max_run_token_cost=opts.get("max_run_token_cost", None),
        enable_termination_summary=opts.get("enable_termination_summary", True),
        termination_summary_prompt=opts.get("termination_summary_prompt", ""),
        run_step_storage=run_step_cfg,
        trace_storage=trace_cfg,
        enable_skill=opts.get("enable_skill", settings.is_skills_enabled),
        skills_dirs=skills_dirs,
        relevant_memory_max_token=opts.get("relevant_memory_max_token", 2048),
        stream_cleanup_timeout=opts.get("stream_cleanup_timeout", 300.0),
        compact_prompt=opts.get("compact_prompt", ""),
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

    citation_store = _create_citation_store(console_config)
    tools: list[BaseTool] = create_tools(builtin_names, citation_source_store=citation_store)

    for ref_id in agent_refs:
        child_config = await registry.get_agent(ref_id)
        if child_config is None:
            continue
        child_agent = await build_agent(child_config, console_config, registry, _building.copy())
        tools.append(AgentTool(child_agent))

    return Agent(
        name=config.name,
        description=config.description,
        model=model,
        id=config.id,
        tools=tools or None,
        system_prompt=config.system_prompt,
        options=options,
    )
