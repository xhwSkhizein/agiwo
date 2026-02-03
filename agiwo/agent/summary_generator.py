"""
Summary Generator - Generates termination summaries.
"""

from agiwo.agent.summarizer import build_termination_messages
from agiwo.utils.logging import get_logger

from agiwo.agent.llm_handler import LLMStreamHandler
from agiwo.agent.run_state import RunState
from agiwo.utils.abort_signal import AbortSignal

logger = get_logger(__name__)


class SummaryGenerator:
    """终止摘要生成器"""

    SUMMARY_REASONS = frozenset(
        {"max_steps", "timeout", "max_tokens", "error_with_context"}
    )

    def __init__(
        self,
        llm_handler: LLMStreamHandler,
        enabled: bool = False,
        custom_prompt: str | None = None,
    ):
        self.llm_handler = llm_handler
        self.enabled = enabled
        self.custom_prompt = custom_prompt

    async def maybe_generate_summary(
        self,
        state: RunState,
        abort_signal: AbortSignal | None,
    ) -> None:
        """如果需要，生成终止摘要"""
        if not self.enabled:
            return

        if state.termination_reason not in self.SUMMARY_REASONS:
            return

        summary_messages = build_termination_messages(
            messages=state.messages,
            termination_reason=state.termination_reason,
            pending_tool_calls=state.tracker.pending_tool_calls,
            custom_prompt=self.custom_prompt,
        )

        step = await self.llm_handler.stream_assistant_step(
            state,
            abort_signal,
            messages=summary_messages,
            tools=None,
            append_message=False,
        )

        logger.info(
            "summary_generated",
            tokens=step.metrics.total_tokens if step.metrics else 0,
        )
