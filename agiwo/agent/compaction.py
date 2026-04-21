"""
Compaction runtime — context compression for long conversations.

Merges compaction/runtime.py + messages.py + parser.py + prompt.py + transcript.py.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles

from agiwo.agent.models.run import CompactMetadata
from agiwo.agent.llm_caller import stream_assistant_step
from agiwo.agent.models.stream import (
    CompactionAppliedEvent,
    MessagesRebuiltEvent,
)
from agiwo.agent.models.step import StepRecord
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.state_ops import (
    record_compaction_metadata,
    replace_messages,
)
from agiwo.agent.runtime.step_committer import commit_step
from agiwo.agent.runtime.state_writer import (
    build_compaction_applied_entry,
    build_messages_rebuilt_entry,
)
from agiwo.llm.base import Model
from agiwo.llm.usage_resolver import ModelUsageEstimator
from agiwo.config.settings import get_settings
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


DEFAULT_COMPACT_PROMPT = """**IMPORTANT: Context Compression Required**

The conversation context is approaching the limit. Please provide a comprehensive summary of the conversation so far.

## Instructions

1. Analyze the entire conversation history above
2. Extract and preserve:
   - Key decisions made by the user or assistant
   - Important facts, data, or conclusions
   - File paths, URLs, or references mentioned
   - Tool calls and their significant results
   - User preferences or explicit requests to remember
   - Current task state and progress

3. Output a JSON object with the following structure:
```json
{{
  "summary": "A comprehensive summary of the conversation...",
  "key_decisions": ["decision 1", "decision 2"],
  "important_refs": ["file/path/1", "https://url.com"],
  "tool_calls_summary": [
    {{"name": "tool_name", "result_summary": "brief result"}}
  ],
  "user_preferences": ["preference 1"],
  "current_task_state": "Description of where we are in the task"
}}
```

## Previous Compact Summary (if any)

{previous_summary}

## Output

Respond ONLY with the JSON object, no additional text.
"""

DEFAULT_ASSISTANT_RESPONSE = (
    "Understood. I have the context from the summary. Continuing."
)


# ---------------------------------------------------------------------------
# Transcript persistence
# ---------------------------------------------------------------------------


async def save_transcript(
    messages: list[dict[str, object]],
    agent_id: str,
    session_id: str,
    start_seq: int,
    end_seq: int,
    root_path: str | None = None,
) -> str:
    """Persist compacted source messages to a transcript file."""
    root = root_path or get_settings().root_path
    transcript_dir = Path(root) / "compaction" / "transcripts" / agent_id / session_id
    transcript_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{start_seq}_{end_seq}_{date_str}.jsonl"
    filepath = transcript_dir / filename

    async with aiofiles.open(filepath, "w", encoding="utf-8") as handle:
        for message in messages:
            await handle.write(
                json.dumps(message, ensure_ascii=False, default=str) + "\n"
            )

    return str(filepath)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _try_load_json(payload: str) -> dict[str, Any] | None:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _find_outer_json_bounds(response_content: str, json_start: int) -> tuple[int, int]:
    depth = 0
    in_string = False
    escape_next = False

    for index, char in enumerate(response_content[json_start:], start=json_start):
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
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return json_start, index + 1

    return json_start, -1


def parse_compact_response(response_content: str) -> dict[str, Any]:
    """Parse the model response and extract the outermost JSON object when present."""
    stripped = response_content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        parsed = _try_load_json(stripped)
        if parsed is not None:
            return parsed

    json_start = response_content.find("{")
    if json_start < 0:
        return {"summary": response_content}

    _, json_end = _find_outer_json_bounds(response_content, json_start)
    if json_end > json_start:
        parsed = _try_load_json(response_content[json_start:json_end])
        if parsed is not None:
            return parsed

    logger.warning(
        "compact_json_parse_failed",
        response_preview=response_content[:200],
    )
    return {"summary": response_content}


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------


def build_compacted_messages(
    system_prompt: str,
    summary: str,
    transcript_path: str,
    latest_user_message: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the compacted message list after compaction."""
    assistant_response = (
        get_settings().compact_assistant_response or DEFAULT_ASSISTANT_RESPONSE
    )

    compacted_messages: list[dict[str, Any]] = []

    if system_prompt:
        compacted_messages.append({"role": "system", "content": system_prompt})

    compact_user_content = (
        f"[Conversation compressed. original source: {transcript_path}]\n\n"
        f"# Summary\n{summary}"
    )
    compacted_messages.append({"role": "user", "content": compact_user_content})
    compacted_messages.append({"role": "assistant", "content": assistant_response})

    if latest_user_message and latest_user_message.get("role") == "user":
        compacted_messages.append(latest_user_message)

    return compacted_messages


# ---------------------------------------------------------------------------
# Compaction functions
# ---------------------------------------------------------------------------


@dataclass
class CompactResult:
    metadata: CompactMetadata | None = None
    failed: bool = False
    error: str | None = None


async def compact_if_needed(
    *,
    state: RunContext,
    model: Model,
    abort_signal: AbortSignal | None,
    max_context_window: int | None,
    compact_prompt: str | None = None,
    compact_start_seq: int,
    root_path: str | None = None,
) -> CompactResult:
    if max_context_window is None:
        return CompactResult()
    _s = get_settings()
    metrics_resolver = ModelUsageEstimator(model)
    estimated_tokens = metrics_resolver.estimate_messages_tokens(
        state.snapshot_messages()
    )
    threshold = int(max_context_window * _s.compact_threshold_ratio)
    if estimated_tokens < threshold:
        return CompactResult()

    resolved_root_path = root_path if root_path is not None else _s.root_path
    retry_count = _s.compact_retry_count
    last_error: Exception | None = None
    for attempt in range(retry_count + 1):
        try:
            metadata = await _compact(
                state,
                model,
                abort_signal,
                compact_prompt=compact_prompt,
                compact_start_seq=compact_start_seq,
                root_path=resolved_root_path,
            )
            logger.info(
                "compact_success",
                run_id=state.run_id,
                start_seq=metadata.start_seq,
                end_seq=metadata.end_seq,
                before_tokens=metadata.before_token_estimate,
                after_tokens=metadata.after_token_estimate,
                attempt=attempt + 1,
            )
            return CompactResult(metadata=metadata)
        except Exception as error:  # noqa: BLE001 - compaction retries guard the runtime boundary
            last_error = error
            logger.warning(
                "compact_attempt_failed",
                run_id=state.run_id,
                attempt=attempt + 1,
                max_attempts=retry_count + 1,
                error=str(error),
            )
            if attempt < retry_count:
                continue

    logger.error(
        "compact_failed_all_retries",
        run_id=state.run_id,
        error=str(last_error),
    )
    return CompactResult(
        failed=True,
        error=str(last_error) if last_error is not None else "unknown",
    )


async def _compact(
    state: RunContext,
    model: Model,
    abort_signal: AbortSignal | None,
    *,
    compact_prompt: str | None,
    compact_start_seq: int,
    root_path: str,
) -> CompactMetadata:
    metrics_resolver = ModelUsageEstimator(model)
    current_messages = state.snapshot_messages()
    before_token_estimate = metrics_resolver.estimate_messages_tokens(current_messages)

    prompt_template = compact_prompt or DEFAULT_COMPACT_PROMPT
    previous_summary = ""
    if state.ledger.compaction.last_metadata:
        previous_summary = state.ledger.compaction.last_metadata.get_summary()
    compact_prompt_content = prompt_template.format(
        previous_summary=previous_summary or "None",
    )
    latest_user_message = next(
        (m for m in reversed(current_messages) if m.get("role") == "user"),
        None,
    )

    sequence = await state.session_runtime.allocate_sequence()
    compact_user_step = StepRecord.user(
        state,
        sequence=sequence,
        content=compact_prompt_content,
        name="compact_request",
    )
    await commit_step(state, compact_user_step, append_message=True)

    step, llm_context = await stream_assistant_step(
        model,
        state,
        abort_signal,
        messages=state.snapshot_messages(),
        use_state_tools=False,
        name="compact",
    )
    await commit_step(state, step, llm=llm_context, append_message=False)

    response_content = step.content or ""
    analysis = parse_compact_response(response_content)
    summary = analysis.get("summary", response_content)

    end_seq = step.sequence
    start_seq = compact_start_seq

    system_prompt = (
        current_messages[0].get("content", "")
        if current_messages and current_messages[0].get("role") == "system"
        else ""
    )

    messages_to_backup = current_messages[1:] if system_prompt else current_messages
    transcript_path = await save_transcript(
        messages=messages_to_backup,
        agent_id=state.agent_id,
        session_id=state.session_id,
        start_seq=start_seq,
        end_seq=end_seq,
        root_path=root_path,
    )

    compacted_messages = build_compacted_messages(
        system_prompt=system_prompt,
        summary=summary,
        transcript_path=transcript_path,
        latest_user_message=latest_user_message,
    )

    after_token_estimate = metrics_resolver.estimate_messages_tokens(compacted_messages)

    metadata = CompactMetadata(
        session_id=state.session_id,
        agent_id=state.agent_id,
        start_seq=start_seq,
        end_seq=end_seq,
        before_token_estimate=before_token_estimate,
        after_token_estimate=after_token_estimate,
        message_count=len(messages_to_backup),
        transcript_path=transcript_path,
        analysis=analysis,
        created_at=datetime.now(),
        compact_model=getattr(model, "name", "unknown"),
        compact_tokens=(step.metrics.total_tokens if step.metrics else 0),
    )

    replace_messages(state, compacted_messages)
    record_compaction_metadata(state, metadata)
    await state.session_runtime.save_compact_metadata(
        state.agent_id,
        metadata,
    )
    rebuilt_entry = build_messages_rebuilt_entry(
        state,
        sequence=await state.session_runtime.allocate_sequence(),
        reason="compaction",
        messages=state.snapshot_messages(),
    )
    compaction_entry = build_compaction_applied_entry(
        state,
        sequence=await state.session_runtime.allocate_sequence(),
        metadata=metadata,
    )
    await state.session_runtime.append_run_log_entries(
        [rebuilt_entry, compaction_entry]
    )
    await state.session_runtime.publish(
        MessagesRebuiltEvent.from_context(
            state,
            reason="compaction",
            message_count=len(state.snapshot_messages()),
        )
    )
    await state.session_runtime.publish(
        CompactionAppliedEvent.from_context(
            state,
            start_sequence=metadata.start_seq,
            end_sequence=metadata.end_seq,
            transcript_path=metadata.transcript_path,
            summary=metadata.get_summary() or None,
        )
    )

    return metadata


__all__ = [
    "CompactResult",
    "compact_if_needed",
    "DEFAULT_ASSISTANT_RESPONSE",
    "DEFAULT_COMPACT_PROMPT",
]
