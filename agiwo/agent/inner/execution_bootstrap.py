from dataclasses import dataclass

from agiwo.agent.compact_types import CompactMetadata
from agiwo.agent.input import ChannelContext
from agiwo.agent.inner.compaction.runtime import CompactionRuntime
from agiwo.agent.inner.context import AgentRunContext
from agiwo.agent.inner.llm_handler import LLMStreamHandler
from agiwo.agent.inner.message_assembler import MessageAssembler
from agiwo.agent.inner.run_recorder import RunRecorder
from agiwo.agent.inner.run_state import RunState
from agiwo.agent.memory_types import MemoryRecord
from agiwo.agent.options import AgentOptions
from agiwo.agent.runtime import StepRecord


@dataclass(frozen=True, slots=True)
class PreparedExecution:
    state: RunState
    run_recorder: RunRecorder
    compactor: CompactionRuntime


async def prepare_execution(
    *,
    system_prompt: str,
    user_step: StepRecord,
    context: AgentRunContext,
    memories: list[MemoryRecord] | None,
    before_run_hook_result: str | None,
    channel_context: ChannelContext | None,
    options: AgentOptions,
    run_recorder: RunRecorder,
    tool_schemas: list[dict] | None,
    llm_handler: LLMStreamHandler,
    compact_prompt: str,
    root_path: str,
) -> PreparedExecution:
    session_storage = context.session_runtime.session_storage
    run_step_storage = context.session_runtime.run_step_storage
    last_compact: CompactMetadata | None = (
        await session_storage.get_latest_compact_metadata(
            context.session_id,
            context.agent_id,
        )
    )
    compact_start_seq = last_compact.end_seq + 1 if last_compact is not None else 0

    existing_steps = await run_step_storage.get_steps(
        session_id=context.session_id,
        agent_id=context.agent_id,
        start_seq=compact_start_seq if compact_start_seq > 0 else None,
    )
    if all(step.id != user_step.id for step in existing_steps):
        existing_steps.append(user_step)
        existing_steps.sort(key=lambda step: step.sequence)

    messages = MessageAssembler.assemble(
        system_prompt,
        existing_steps,
        memories,
        before_run_hook_result,
        channel_context=channel_context,
    )
    state = RunState(
        context=context,
        config=options,
        messages=messages,
        tool_schemas=tool_schemas,
        last_compact_metadata=last_compact,
        compact_start_seq=compact_start_seq,
    )
    return PreparedExecution(
        state=state,
        run_recorder=run_recorder.attach_state(state),
        compactor=CompactionRuntime(
            llm_handler=llm_handler,
            session_storage=session_storage,
            compact_prompt=compact_prompt,
            root_path=root_path,
        ),
    )


__all__ = ["PreparedExecution", "prepare_execution"]
