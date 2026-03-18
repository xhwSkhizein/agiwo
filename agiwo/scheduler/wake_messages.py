"""Wake-message construction for scheduler-managed agent resumes."""

from agiwo.agent.input import UserInput
from agiwo.scheduler.formatting import (
    build_child_result_detail_lines,
    format_child_results_summary,
)
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerEventType,
    WakeType,
)
from agiwo.scheduler.store import AgentStateStorage


class WakeMessageBuilder:
    def __init__(self, store: AgentStateStorage) -> None:
        self._store = store

    async def build(self, state: AgentState) -> UserInput:
        wc = state.wake_condition
        if wc is None:
            return "You have been woken up. Please continue your task."

        if wc.type == WakeType.WAITSET:
            succeeded, failed = await self._collect_child_results(state)
            done = len(succeeded) + len(failed)
            return format_child_results_summary(
                header=f"Child agents completed ({done}/{len(wc.wait_for)}).",
                succeeded=succeeded,
                failed=failed,
                closing_instruction=(
                    "Please synthesize a final response based on the successful "
                    "results above."
                ),
            )
        if wc.type == WakeType.TIMER:
            return "The scheduled delay has elapsed. Please continue your task."
        if wc.type == WakeType.PERIODIC:
            return (
                "A scheduled periodic check has triggered. "
                "Please check progress and decide whether to continue waiting "
                "or produce a final result."
            )
        return "You have been woken up. Please continue your task."

    async def build_timeout(self, state: AgentState) -> str:
        succeeded, failed = await self._collect_child_results(state)
        wc = state.wake_condition
        done = len(succeeded) + len(failed)
        return format_child_results_summary(
            header="Wait timeout reached.",
            succeeded=succeeded,
            failed=failed,
            progress_line=f"Completed children: {done}/{len(wc.wait_for) if wc else 0}",
            closing_instruction=(
                "Please produce a summary report with whatever results are available."
            ),
        )

    def build_from_events(self, events: list[PendingEvent]) -> str:
        lines = [f"You have {len(events)} new notification(s):\n"]
        for event in events:
            event_label = event.event_type.value.replace("_", " ").title()
            child_id = event.payload.get(
                "child_agent_id", event.source_agent_id or "unknown"
            )
            lines.append(f"### {event_label} - Agent: {child_id}")
            if event.event_type == SchedulerEventType.CHILD_SLEEP_RESULT:
                lines.extend(
                    build_child_result_detail_lines(
                        result=event.payload.get("result", ""),
                        explain=event.payload.get("explain"),
                        periodic=event.payload.get("periodic", False),
                        result_as_block=True,
                    )
                )
            elif event.event_type == SchedulerEventType.CHILD_COMPLETED:
                lines.extend(
                    build_child_result_detail_lines(
                        result=event.payload.get("result", ""),
                        result_as_block=True,
                    )
                )
            elif event.event_type == SchedulerEventType.CHILD_FAILED:
                lines.extend(
                    build_child_result_detail_lines(
                        failure_reason=event.payload.get("reason", "Unknown failure")
                    )
                )
            elif event.event_type == SchedulerEventType.USER_HINT:
                hint = event.payload.get("hint", "")
                if hint:
                    lines.append(f"User hint: {hint}")
            lines.append("")
        lines.append(
            "Please review these notifications and take appropriate action "
            "(e.g., summarize results for the user, cancel stuck agents, etc.)."
        )
        return "\n".join(lines)

    async def _collect_child_results(
        self,
        state: AgentState,
    ) -> tuple[dict[str, str], dict[str, str]]:
        wc = state.wake_condition
        child_ids = wc.wait_for if wc else []
        if not child_ids:
            child_ids = [
                child.id
                for child in await self._store.list_states(
                    parent_id=state.id,
                    limit=1000,
                )
            ]

        succeeded: dict[str, str] = {}
        failed: dict[str, str] = {}
        for child_id in child_ids:
            child = await self._store.get_state(child_id)
            if child is None:
                failed[child_id] = "Agent state not found"
            elif child.status == AgentStateStatus.FAILED:
                failed[child_id] = child.result_summary or "Unknown failure"
            else:
                succeeded[child_id] = (
                    child.result_summary or f"status={child.status.value}"
                )
        return succeeded, failed


__all__ = ["WakeMessageBuilder"]
