"""
Compactor - Context compression component.

Design Philosophy:
- Uses the same Model as the Agent (preserves LLM KVCache)
- Continues conversation by appending User message requesting summary
- Similar to termination summary pattern
- Never truncates or discards existing context
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any
import aiofiles

from agiwo.agent.compact.prompt import (
    DEFAULT_COMPACT_PROMPT,
    DEFAULT_ASSISTANT_RESPONSE,
)
from agiwo.agent.inner.event_emitter import EventEmitter
from agiwo.agent.inner.llm_handler import LLMStreamHandler
from agiwo.agent.inner.run_state import RunState
from agiwo.agent.compact_types import CompactMetadata, CompactResult
from agiwo.agent.runtime import StepRecord
from agiwo.agent.storage.session import SessionStorage
from agiwo.config.settings import settings
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


def build_compacted_messages(
    system_prompt: str,
    summary: str,
    transcript_path: str,
    latest_user_message: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Build the compacted message list after compact operation.

    Args:
        system_prompt: Original system prompt
        summary: The compact summary from LLM
        transcript_path: Path to the transcript backup file
        latest_user_message: The most recent user message to preserve

    Returns:
        Compacted messages list
    """
    assistant_response = (
        settings.compact_assistant_response or DEFAULT_ASSISTANT_RESPONSE
    )

    compacted_messages: list[dict[str, Any]] = []

    # 1. System prompt (unchanged)
    if system_prompt:
        compacted_messages.append({"role": "system", "content": system_prompt})

    # 2. Compact summary as user message
    compact_user_content = f"[Conversation compressed. original source: {transcript_path}]\n\n# Summary\n{summary}"
    compacted_messages.append({"role": "user", "content": compact_user_content})

    # 3. Assistant acknowledgment
    compacted_messages.append({"role": "assistant", "content": assistant_response})

    # 4. Latest user message (preserved)
    if latest_user_message and latest_user_message.get("role") == "user":
        compacted_messages.append(latest_user_message)

    return compacted_messages


async def save_transcript(
    messages: list[dict[str, Any]],
    agent_id: str,
    session_id: str,
    start_seq: int,
    end_seq: int,
    root_path: str | None = None,
) -> str:
    """
    Save messages to transcript file (async).

    Args:
        messages: Messages to backup
        agent_id: Agent ID
        session_id: Session ID
        start_seq: Start sequence number (StepRecord.sequence)
        end_seq: End sequence number (StepRecord.sequence)
        root_path: Root path for storage

    Returns:
        Path to the saved transcript file
    """

    root = root_path or settings.root_path
    transcript_dir = Path(root) / "transcripts" / agent_id / session_id
    transcript_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{start_seq}_{end_seq}_{date_str}.jsonl"
    filepath = transcript_dir / filename

    async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
        for msg in messages:
            await f.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")

    return str(filepath)


def parse_compact_response(response_content: str) -> dict[str, Any]:
    """
    Parse LLM response as JSON analysis.

    Uses bracket counting to find the outermost JSON object,
    handling nested objects correctly.

    Args:
        response_content: Raw LLM response

    Returns:
        Parsed analysis dict with 'summary' key
    """
    # Try to parse as pure JSON first
    stripped = response_content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Find outermost JSON object using bracket counting
    json_start = response_content.find("{")
    if json_start < 0:
        return {"summary": response_content}

    depth = 0
    json_end = -1
    in_string = False
    escape_next = False

    for i, char in enumerate(response_content[json_start:], start=json_start):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                json_end = i + 1
                break

    if json_end > json_start:
        try:
            json_str = response_content[json_start:json_end]
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    logger.warning(
        "compact_json_parse_failed",
        response_preview=response_content[:200],
    )
    return {"summary": response_content}


class Compactor:
    """
    Context compression component (stateless).

    Responsibilities:
    - Check if compact is needed based on token threshold
    - Compress conversation history using LLM summarization
    - Generate CompactMetadata for persistence
    - Backup original messages to transcript file

    Note: Compact state (last_compact_metadata, compact_start_seq) is held by RunState.
    """

    def __init__(
        self,
        llm_handler: "LLMStreamHandler",
        emitter: "EventEmitter",
        session_storage: "SessionStorage",
        *,
        compact_prompt: str | None = None,
        root_path: str | None = None,
    ):
        self.llm_handler = llm_handler
        self.emitter = emitter
        self.session_storage = session_storage
        self.compact_prompt = compact_prompt
        self.root_path = root_path or settings.root_path

    def should_compact(
        self, messages: list[dict], max_context_window: int | None
    ) -> bool:
        """Check if compact should be triggered."""
        if max_context_window is None:
            return False

        estimated_tokens = self.llm_handler.metrics_resolver.estimate_messages_tokens(
            messages
        )
        threshold = int(max_context_window * settings.compact_threshold_ratio)
        return estimated_tokens >= threshold

    async def compact(
        self,
        state: "RunState",
        abort_signal: "AbortSignal | None",
    ) -> CompactResult:
        """
        Perform compact operation using Summary-like pattern.

        1. Build compact messages by appending user request to existing messages
        2. Call LLM to generate summary (preserves KVCache)
        3. Build compacted messages and metadata
        """
        metrics_resolver = self.llm_handler.metrics_resolver
        before_token_estimate = metrics_resolver.estimate_messages_tokens(
            state.messages
        )

        # Build compact prompt
        prompt_template = self.compact_prompt or DEFAULT_COMPACT_PROMPT
        previous_summary = ""
        if state.last_compact_metadata:
            previous_summary = state.last_compact_metadata.get_summary()
        compact_prompt_content = prompt_template.format(
            previous_summary=previous_summary or "None",
        )

        # Record Compact UserStep (the compact request)
        user_seq = await state.next_sequence()
        compact_user_step = StepRecord.user(
            state.context,
            sequence=user_seq,
            content=compact_prompt_content,
            name="compact_request",
        )
        await self.emitter.emit_step_completed(compact_user_step)
        state.track_step(compact_user_step, append_message=True)

        # Build compact messages (now includes the compact user step)
        compact_messages = list(state.messages)

        # Call LLM for summarization (metrics auto-resolved by LLMStreamHandler)
        step, llm_context = await self.llm_handler.stream_assistant_step(
            state,
            self.emitter.emit_step_delta,
            abort_signal,
            messages=compact_messages,
            tools=None,
        )
        step.name = "compact"
        await self.emitter.emit_step_completed(step, llm=llm_context)
        state.track_step(step, append_message=False)

        # Parse response
        response_content = step.content or ""
        analysis = parse_compact_response(response_content)
        summary = analysis.get("summary", response_content)

        # Calculate sequence range (based on StepRecord.sequence)
        end_seq = step.sequence
        start_seq = state.compact_start_seq

        # Extract system prompt and latest user message
        system_prompt = self._extract_system_prompt(state.messages)
        latest_user_message = self._extract_latest_user_message(state.messages)

        # Backup original messages to transcript
        messages_to_backup = state.messages[1:] if system_prompt else state.messages
        transcript_path = await save_transcript(
            messages=messages_to_backup,
            agent_id=state.context.agent_id,
            session_id=state.context.session_id,
            start_seq=start_seq,
            end_seq=end_seq,
            root_path=self.root_path,
        )

        # Build compacted messages
        compacted_messages = build_compacted_messages(
            system_prompt=system_prompt,
            summary=summary,
            transcript_path=transcript_path,
            latest_user_message=latest_user_message,
        )

        after_token_estimate = metrics_resolver.estimate_messages_tokens(
            compacted_messages
        )

        # Build metadata
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

        # Update RunState with compacted messages
        state.messages = compacted_messages
        state.last_compact_metadata = metadata
        state.compact_start_seq = end_seq + 1

        # Persist metadata
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
        """Extract system prompt from messages."""
        if messages and messages[0].get("role") == "system":
            return messages[0].get("content", "")
        return ""

    @staticmethod
    def _extract_latest_user_message(messages: list[dict]) -> dict | None:
        """Extract the latest user message."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg
        return None


__all__ = [
    "build_compacted_messages",
    "save_transcript",
    "parse_compact_response",
    "Compactor",
]
