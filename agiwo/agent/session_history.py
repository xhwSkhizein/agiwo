"""Append user-provided messages to Session history (RunLog)."""

from uuid import uuid4

from agiwo.agent.agent import Agent
from agiwo.agent.models.input import UserMessage
from agiwo.agent.models.log import UserStepCommitted
from agiwo.agent.models.step import MessageRole


async def append_session_user_message_to_history(
    agent: Agent,
    *,
    session_id: str,
    user_message: UserMessage,
    run_id: str | None = None,
) -> None:
    """Append a real user message to Session history once."""
    if not user_message.is_user_provided:
        raise ValueError(
            "append_session_user_message_to_history requires user-provided message"
        )
    seed_run_id = run_id or f"session-history-{session_id}"
    sequence = await agent.run_log_storage.allocate_sequence(session_id)
    entry = UserStepCommitted(
        sequence=sequence,
        session_id=session_id,
        run_id=seed_run_id,
        agent_id=agent.id,
        step_id=str(uuid4()),
        role=MessageRole.USER,
        content=user_message.to_message_content(),
        user_input=user_message,
    )
    await agent.run_log_storage.append_entries([entry])


__all__ = ["append_session_user_message_to_history"]
