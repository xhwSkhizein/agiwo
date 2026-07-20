"""Prefix-safe Session history helpers for Objective user inputs."""

from uuid import uuid4

from agiwo.agent import Agent, UserMessage
from agiwo.agent.models.log import UserStepCommitted
from agiwo.agent.models.step import MessageRole
from agiwo.objective.input import verify_user_inputs_in_history
from agiwo.objective.models import ObjectiveUserInput
from agiwo.objective.projection import ObjectiveView


async def collect_history_input_ids_from_agent(
    agent: Agent,
    *,
    session_id: str,
) -> set[str]:
    steps = await agent.run_log_storage.list_step_views(
        session_id=session_id,
        agent_id=agent.id,
    )
    found: set[str] = set()
    for step in steps:
        if step.user_input is None:
            continue
        message = UserMessage.from_value(step.user_input)
        if message.objective_input_id:
            found.add(message.objective_input_id)
    return found


async def append_objective_user_input_to_history(
    agent: Agent,
    *,
    session_id: str,
    user_input: ObjectiveUserInput,
    objective_id: str | None = None,
) -> None:
    """Append one tagged ObjectiveUserInput to Session history if missing."""
    history_ids = await collect_history_input_ids_from_agent(
        agent,
        session_id=session_id,
    )
    if user_input.input_id in history_ids:
        return
    seed_run_id = (
        f"objective-history-{objective_id}"
        if objective_id
        else f"objective-history-{user_input.input_id}"
    )
    sequence = await agent.run_log_storage.allocate_sequence(session_id)
    entry = UserStepCommitted(
        sequence=sequence,
        session_id=session_id,
        run_id=seed_run_id,
        agent_id=agent.id,
        step_id=str(uuid4()),
        role=MessageRole.USER,
        content=user_input.message.to_message_content(),
        user_input=user_input.message,
    )
    await agent.run_log_storage.append_entries([entry])


async def ensure_user_inputs_in_history(
    agent: Agent,
    view: ObjectiveView,
) -> list[str]:
    """Ensure ObjectiveUserInputs are present in Session history.

    Bootstrap rule: when history has *no* objective_input_id markers yet,
    append missing user inputs at the end (never mid-insert). Otherwise any
    gap fails closed and the missing ids are returned.
    """
    history_ids = await collect_history_input_ids_from_agent(
        agent,
        session_id=view.session_id,
    )
    externalized = {item.input_id for item in view.externalized_inputs}
    missing = verify_user_inputs_in_history(
        view.user_inputs,
        history_input_ids=history_ids,
        externalized_input_ids=externalized,
    )
    if not missing:
        return []
    if history_ids:
        return missing

    seed_run_id = f"objective-history-{view.objective_id}"
    for user_input in view.user_inputs:
        if user_input.input_id not in missing:
            continue
        if user_input.input_id in externalized:
            continue
        sequence = await agent.run_log_storage.allocate_sequence(view.session_id)
        entry = UserStepCommitted(
            sequence=sequence,
            session_id=view.session_id,
            run_id=seed_run_id,
            agent_id=agent.id,
            step_id=str(uuid4()),
            role=MessageRole.USER,
            content=user_input.message.to_message_content(),
            user_input=user_input.message,
        )
        await agent.run_log_storage.append_entries([entry])
    return []


def estimate_required_input_tokens(view: ObjectiveView) -> int:
    """Rough token estimate for must-keep Objective inputs (chars/4)."""
    total_chars = 0
    for item in view.user_inputs:
        if any(e.input_id == item.input_id for e in view.externalized_inputs):
            continue
        total_chars += len(item.message.extract_text())
    if view.current_goal is not None:
        total_chars += len(view.current_goal.intent or "")
        total_chars += len(view.current_goal.scope or "")
        total_chars += len(view.current_goal.success_criteria or "")
    for outcome in view.outcomes[-3:]:
        total_chars += len(outcome.report or "")
    return max(1, total_chars // 4)


__all__ = [
    "append_objective_user_input_to_history",
    "collect_history_input_ids_from_agent",
    "ensure_user_inputs_in_history",
    "estimate_required_input_tokens",
]
