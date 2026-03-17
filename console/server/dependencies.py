"""Console runtime container and FastAPI dependency helpers."""

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, FastAPI, Request

from agiwo.agent.storage.base import RunStepStorage
from server.services.storage_wiring import NotifyingTraceStorage
from agiwo.scheduler.scheduler import Scheduler

from server.channels.feishu import FeishuChannelService
from server.config import ConsoleConfig
from server.services.agent_registry import AgentRegistry

_RUNTIME_STATE_KEY = "console_runtime"


@dataclass
class ConsoleRuntime:
    config: ConsoleConfig
    run_step_storage: RunStepStorage
    trace_storage: NotifyingTraceStorage
    agent_registry: AgentRegistry
    scheduler: Scheduler | None = None
    feishu_channel_service: FeishuChannelService | None = None


def bind_console_runtime(app: FastAPI, runtime: ConsoleRuntime) -> None:
    setattr(app.state, _RUNTIME_STATE_KEY, runtime)


def clear_console_runtime(app: FastAPI) -> None:
    if hasattr(app.state, _RUNTIME_STATE_KEY):
        delattr(app.state, _RUNTIME_STATE_KEY)


def get_console_runtime_from_app(app: FastAPI) -> ConsoleRuntime:
    runtime = getattr(app.state, _RUNTIME_STATE_KEY, None)
    if runtime is None:
        raise RuntimeError("ConsoleRuntime not initialized")
    return runtime


def get_console_runtime(request: Request) -> ConsoleRuntime:
    return get_console_runtime_from_app(request.app)


def get_console_config(runtime: "ConsoleRuntimeDep") -> ConsoleConfig:
    return runtime.config


def get_run_step_storage(runtime: "ConsoleRuntimeDep") -> RunStepStorage:
    return runtime.run_step_storage


def get_trace_storage(runtime: "ConsoleRuntimeDep") -> NotifyingTraceStorage:
    return runtime.trace_storage


def get_agent_registry(runtime: "ConsoleRuntimeDep") -> AgentRegistry:
    return runtime.agent_registry


def get_scheduler(runtime: "ConsoleRuntimeDep") -> Scheduler:
    if runtime.scheduler is None:
        raise RuntimeError("Scheduler not initialized")
    return runtime.scheduler


def get_feishu_channel_service(
    runtime: "ConsoleRuntimeDep",
) -> FeishuChannelService | None:
    return runtime.feishu_channel_service


ConsoleRuntimeDep = Annotated[ConsoleRuntime, Depends(get_console_runtime)]
SchedulerDep = Annotated[Scheduler, Depends(get_scheduler)]


__all__ = [
    "ConsoleRuntime",
    "ConsoleRuntimeDep",
    "SchedulerDep",
    "bind_console_runtime",
    "clear_console_runtime",
    "get_agent_registry",
    "get_console_config",
    "get_console_runtime",
    "get_console_runtime_from_app",
    "get_feishu_channel_service",
    "get_scheduler",
    "get_run_step_storage",
    "get_trace_storage",
]
