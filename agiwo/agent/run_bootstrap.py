"""Bootstrap helpers for assembling run context before the main loop."""

from dataclasses import dataclass

from agiwo.agent.models.input import UserInput, UserMessage
from agiwo.agent.models.run import MemoryRecord
from agiwo.agent.models.step import StepView
from agiwo.agent.prompt import assemble_run_messages
from agiwo.agent.review import build_review_state_from_entries
from agiwo.agent.runtime.context import RunContext, RunRuntime
from agiwo.agent.runtime.state_writer import RunStateWriter


@dataclass(frozen=True, slots=True)
class RunBootstrapResult:
    user_step: StepView
    compact_start_seq: int


async def prepare_run_context(
    *,
    context: RunContext,
    runtime: RunRuntime,
    user_input: UserInput | None,
    system_prompt: str,
    writer: RunStateWriter,
) -> RunBootstrapResult:
    """Build all state needed before the main loop starts."""
    before_run_hook_result = await context.hooks.before_run(user_input, context)
    memories = await _retrieve_memories(context, user_input)
    await _restore_review_state(context)

    user_step = await _build_user_step(context, user_input)
    latest_compact = await context.session_runtime.get_latest_compact_metadata(
        context.agent_id
    )
    compact_start_seq = latest_compact.end_seq + 1 if latest_compact is not None else 0
    existing_steps = await _load_existing_steps(
        context=context,
        compact_start_seq=compact_start_seq,
    )
    existing_steps.append(user_step)
    existing_steps.sort(key=lambda step: step.sequence)

    user_message = UserMessage.from_value(user_input)
    assembled_messages = assemble_run_messages(
        system_prompt,
        existing_steps,
        memories,
        before_run_hook_result,
        channel_context=user_message.context,
    )
    await writer.record_context_assembled(
        messages=assembled_messages,
        memory_count=len(memories),
        run_start_seq=user_step.sequence,
        tool_schemas=_build_tool_schemas(runtime),
        latest_compaction=latest_compact,
    )

    return RunBootstrapResult(
        user_step=user_step,
        compact_start_seq=compact_start_seq,
    )


def _build_tool_schemas(runtime: RunRuntime) -> list[dict[str, object]] | None:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.get_definition().name,
                "description": tool.get_definition().description,
                "parameters": tool.get_definition().parameters,
            },
        }
        for tool in runtime.tools_map.values()
    ] or None


async def _retrieve_memories(
    context: RunContext,
    user_input: UserInput | None,
) -> list[MemoryRecord]:
    if user_input is None:
        return []
    return await context.hooks.memory_retrieve(user_input, context)


async def _build_user_step(
    context: RunContext,
    user_input: UserInput | None,
) -> StepView:
    return StepView.user(
        context,
        sequence=await context.session_runtime.allocate_sequence(),
        user_input=user_input,
    )


async def _load_existing_steps(
    *,
    context: RunContext,
    compact_start_seq: int,
) -> list[StepView]:
    return await context.session_runtime.run_log_storage.list_step_views(
        session_id=context.session_id,
        agent_id=context.agent_id,
        start_seq=compact_start_seq if compact_start_seq > 0 else None,
        include_hidden_from_context=False,
    )


async def _restore_review_state(context: RunContext) -> None:
    entries = await context.session_runtime.list_run_log_entries(
        agent_id=context.agent_id,
        limit=100_000,
    )
    context.ledger.review = build_review_state_from_entries(entries)


__all__ = ["RunBootstrapResult", "prepare_run_context"]
