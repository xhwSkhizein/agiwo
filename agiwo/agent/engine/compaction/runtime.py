"""Internal compaction runtime used by ExecutionEngine."""

from datetime import datetime

from agiwo.agent.compact_types import CompactMetadata, CompactResult
from agiwo.agent.engine.compaction.messages import build_compacted_messages
from agiwo.agent.engine.compaction.parser import parse_compact_response
from agiwo.agent.engine.compaction.prompt import DEFAULT_COMPACT_PROMPT
from agiwo.agent.engine.compaction.transcript import save_transcript
from agiwo.agent.engine.llm_handler import LLMStreamHandler
from agiwo.agent.engine.recorder import RunRecorder
from agiwo.agent.engine.state import RunState
from agiwo.agent.storage.session import SessionStorage
from agiwo.config.settings import settings
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class CompactionRuntime:
    """Context compression component owned by the agent executor."""

    def __init__(
        self,
        llm_handler: LLMStreamHandler,
        session_storage: SessionStorage,
        *,
        compact_prompt: str | None = None,
        root_path: str | None = None,
    ) -> None:
        self.llm_handler = llm_handler
        self.session_storage = session_storage
        self.compact_prompt = compact_prompt
        self.root_path = root_path or settings.root_path

    def should_compact(
        self, messages: list[dict], max_context_window: int | None
    ) -> bool:
        if max_context_window is None:
            return False

        estimated_tokens = self.llm_handler.metrics_resolver.estimate_messages_tokens(
            messages
        )
        threshold = int(max_context_window * settings.compact_threshold_ratio)
        return estimated_tokens >= threshold

    async def compact_if_needed(
        self,
        *,
        state: RunState,
        run_recorder: RunRecorder,
        abort_signal: AbortSignal | None,
        max_context_window: int | None,
    ) -> CompactResult | None:
        if not self.should_compact(state.messages, max_context_window):
            return None

        retry_count = settings.compact_retry_count
        last_error: Exception | None = None
        for attempt in range(retry_count + 1):
            try:
                result = await self.compact(
                    state,
                    run_recorder,
                    abort_signal,
                )
                logger.info(
                    "compact_success",
                    run_id=state.context.run_id,
                    start_seq=result.metadata.start_seq,
                    end_seq=result.metadata.end_seq,
                    before_tokens=result.metadata.before_token_estimate,
                    after_tokens=result.metadata.after_token_estimate,
                    attempt=attempt + 1,
                )
                return result
            except Exception as error:  # noqa: BLE001 - compaction retries guard the runtime boundary
                last_error = error
                logger.warning(
                    "compact_attempt_failed",
                    run_id=state.context.run_id,
                    attempt=attempt + 1,
                    max_attempts=retry_count + 1,
                    error=str(error),
                )
                if attempt < retry_count:
                    continue

        logger.error(
            "compact_failed_all_retries",
            run_id=state.context.run_id,
            error=str(last_error),
        )
        return None

    async def compact(
        self,
        state: RunState,
        run_recorder: RunRecorder,
        abort_signal: AbortSignal | None,
    ) -> CompactResult:
        metrics_resolver = self.llm_handler.metrics_resolver
        before_token_estimate = metrics_resolver.estimate_messages_tokens(
            state.messages
        )

        prompt_template = self.compact_prompt or DEFAULT_COMPACT_PROMPT
        previous_summary = ""
        if state.last_compact_metadata:
            previous_summary = state.last_compact_metadata.get_summary()
        compact_prompt_content = prompt_template.format(
            previous_summary=previous_summary or "None",
        )

        compact_user_step = await run_recorder.create_user_step(
            content=compact_prompt_content,
            name="compact_request",
        )
        await run_recorder.commit_step(compact_user_step, append_message=True)

        compact_messages = list(state.messages)
        step, llm_context = await self.llm_handler.stream_assistant_step(
            state,
            run_recorder,
            abort_signal,
            messages=compact_messages,
            tools=None,
        )
        step.name = "compact"
        await run_recorder.commit_step(step, llm=llm_context, append_message=False)

        response_content = step.content or ""
        analysis = parse_compact_response(response_content)
        summary = analysis.get("summary", response_content)

        end_seq = step.sequence
        start_seq = state.compact_start_seq

        system_prompt = self._extract_system_prompt(state.messages)
        latest_user_message = self._extract_latest_user_message(state.messages)

        messages_to_backup = state.messages[1:] if system_prompt else state.messages
        transcript_path = await save_transcript(
            messages=messages_to_backup,
            agent_id=state.context.agent_id,
            session_id=state.context.session_id,
            start_seq=start_seq,
            end_seq=end_seq,
            root_path=self.root_path,
        )

        compacted_messages = build_compacted_messages(
            system_prompt=system_prompt,
            summary=summary,
            transcript_path=transcript_path,
            latest_user_message=latest_user_message,
        )

        after_token_estimate = metrics_resolver.estimate_messages_tokens(
            compacted_messages
        )

        metadata = CompactMetadata(
            session_id=state.context.session_id,
            agent_id=state.context.agent_id,
            start_seq=start_seq,
            end_seq=end_seq,
            before_token_estimate=before_token_estimate,
            after_token_estimate=after_token_estimate,
            message_count=len(messages_to_backup),
            transcript_path=transcript_path,
            analysis=analysis,
            created_at=datetime.now(),
            compact_model=getattr(self.llm_handler.model, "name", "unknown"),
            compact_tokens=(step.metrics.total_tokens if step.metrics else 0),
        )

        state.apply_compaction(compacted_messages, metadata)

        await self.session_storage.save_compact_metadata(
            state.context.session_id,
            state.context.agent_id,
            metadata,
        )

        logger.info(
            "compact_completed",
            run_id=state.context.run_id,
            start_seq=start_seq,
            end_seq=end_seq,
            before_tokens=before_token_estimate,
            after_tokens=after_token_estimate,
        )

        return CompactResult(
            compacted_messages=compacted_messages,
            metadata=metadata,
            step=step,
        )

    @staticmethod
    def _extract_system_prompt(messages: list[dict]) -> str:
        if messages and messages[0].get("role") == "system":
            return messages[0].get("content", "")
        return ""

    @staticmethod
    def _extract_latest_user_message(messages: list[dict]) -> dict | None:
        for message in reversed(messages):
            if message.get("role") == "user":
                return message
        return None


__all__ = ["CompactionRuntime"]
