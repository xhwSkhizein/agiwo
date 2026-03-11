"""Shared helpers for scheduler tool implementations."""

from datetime import datetime, timedelta, timezone

from agiwo.agent.agent import Agent
from agiwo.scheduler.guard import TaskGuard
from agiwo.scheduler.models import (
    AgentStateStatus,
    TimeUnit,
    WaitMode,
    WakeCondition,
    WakeType,
    to_seconds,
)
from agiwo.scheduler.store import AgentStateStorage
from agiwo.tool.base import AgentProcessProbe


async def build_sleep_condition(
    store: AgentStateStorage,
    guard: TaskGuard,
    *,
    agent_id: str,
    session_id: str,
    wake_type: WakeType,
    wait_mode_str: str,
    explicit_wait_for: list[str] | None,
    timeout: float | None,
    delay_seconds: float | int | None,
    time_unit_str: str,
) -> WakeCondition:
    now = datetime.now(timezone.utc)
    wake_condition = WakeCondition(type=wake_type)
    if wake_type == WakeType.WAITSET:
        wake_condition.wait_for = await _resolve_waitset_targets(
            store,
            agent_id=agent_id,
            session_id=session_id,
            explicit_wait_for=explicit_wait_for,
        )
        wake_condition.wait_mode = _resolve_wait_mode(wait_mode_str)
        wake_condition.completed_ids = await _collect_completed_child_ids(
            store,
            wake_condition.wait_for,
        )
        effective_timeout = timeout or guard.limits.default_wait_timeout
        wake_condition.timeout_at = now + timedelta(seconds=effective_timeout)
        return wake_condition

    if delay_seconds is None:
        raise ValueError("delay_seconds is required for timer/periodic wake type")

    time_unit = _resolve_time_unit(time_unit_str)
    wake_condition.time_value = delay_seconds
    wake_condition.time_unit = time_unit
    wake_condition.wakeup_at = now + timedelta(
        seconds=to_seconds(delay_seconds, time_unit)
    )
    if wake_type == WakeType.PERIODIC and timeout is not None:
        wake_condition.timeout_at = now + timedelta(seconds=timeout)
    return wake_condition


def build_sleep_summary(
    *,
    agent_id: str,
    wake_type: WakeType,
    wake_type_str: str,
    wake_condition: WakeCondition,
    delay_seconds: float | int | None,
    time_unit_str: str,
    explain: str | None,
) -> str:
    summary = f"Agent '{agent_id}' entering sleep. Wake condition: {wake_type_str}"
    if wake_type == WakeType.WAITSET:
        summary += (
            " "
            f"(waiting_for={len(wake_condition.wait_for)}, "
            f"mode={wake_condition.wait_mode.value}, "
            f"already_done={len(wake_condition.completed_ids)})"
        )
    elif delay_seconds is not None:
        summary += f" (delay={delay_seconds} {time_unit_str})"
    if explain:
        summary += f" | reason: {explain}"
    return summary


async def list_running_agent_processes(
    agent: Agent | None,
    agent_id: str,
) -> list[dict[str, object]]:
    if agent is None:
        return []

    for tool in agent.tools:
        if not isinstance(tool, AgentProcessProbe):
            continue
        try:
            return await tool.list_agent_processes(agent_id, state="running")
        except Exception:  # noqa: BLE001 - tool capability boundary
            return []

    return []


async def _resolve_waitset_targets(
    store: AgentStateStorage,
    *,
    agent_id: str,
    session_id: str,
    explicit_wait_for: list[str] | None,
) -> list[str]:
    if explicit_wait_for is not None:
        return explicit_wait_for

    children = await store.get_states_by_parent(agent_id)
    return [child.id for child in children if child.session_id == session_id]


def _resolve_wait_mode(wait_mode_str: str) -> WaitMode:
    try:
        return WaitMode(wait_mode_str)
    except ValueError:
        return WaitMode.ALL


async def _collect_completed_child_ids(
    store: AgentStateStorage,
    child_ids: list[str],
) -> list[str]:
    completed_ids: list[str] = []
    for child_id in child_ids:
        child_state = await store.get_state(child_id)
        if child_state is not None and child_state.status in (
            AgentStateStatus.COMPLETED,
            AgentStateStatus.FAILED,
        ):
            completed_ids.append(child_id)
    return completed_ids


def _resolve_time_unit(time_unit_str: str) -> TimeUnit:
    try:
        return TimeUnit(time_unit_str)
    except ValueError:
        return TimeUnit.SECONDS
