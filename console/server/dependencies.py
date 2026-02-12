"""
Global dependency instances for the console server.

Initialized during app lifespan, accessed by routers.
"""

from agiwo.scheduler.scheduler import Scheduler

from server.config import ConsoleConfig
from server.services.storage_manager import StorageManager
from server.services.agent_registry import AgentRegistry

_storage_manager: StorageManager | None = None
_agent_registry: AgentRegistry | None = None
_console_config: ConsoleConfig | None = None
_scheduler: Scheduler | None = None


def set_console_config(config: ConsoleConfig) -> None:
    global _console_config
    _console_config = config


def get_console_config() -> ConsoleConfig:
    if _console_config is None:
        raise RuntimeError("ConsoleConfig not initialized")
    return _console_config


def set_storage_manager(manager: StorageManager) -> None:
    global _storage_manager
    _storage_manager = manager


def get_storage_manager() -> StorageManager:
    if _storage_manager is None:
        raise RuntimeError("StorageManager not initialized")
    return _storage_manager


def set_agent_registry(registry: AgentRegistry) -> None:
    global _agent_registry
    _agent_registry = registry


def get_agent_registry() -> AgentRegistry:
    if _agent_registry is None:
        raise RuntimeError("AgentRegistry not initialized")
    return _agent_registry


def set_scheduler(scheduler: Scheduler) -> None:
    global _scheduler
    _scheduler = scheduler


def get_scheduler() -> Scheduler:
    if _scheduler is None:
        raise RuntimeError("Scheduler not initialized")
    return _scheduler
