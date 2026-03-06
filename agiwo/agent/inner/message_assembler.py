"""
MessageAssembler - Assembles complete LLM message lists.
"""

from typing import Any

from agiwo.agent.schema import (
    ChannelContext,
    MemoryRecord,
    StepRecord,
    steps_to_messages,
)


def _render_channel_context(ctx: ChannelContext) -> str:
    lines = [f"source: {ctx.source}"]
    for key, value in ctx.metadata.items():
        if key in ("recent_dm_messages", "recent_group_messages") and isinstance(value, list):
            if value:
                lines.append(f"{key}:")
                for msg in value:
                    lines.append(f"  - {msg}")
        elif isinstance(value, (str, int, float, bool)):
            lines.append(f"{key}: {value}")
    return "<channel-context>\n" + "\n".join(lines) + "\n</channel-context>"


def _render_memories(memories: list[MemoryRecord]) -> str:
    content = "\n\n".join(m.content for m in memories)
    return f"<relevant-memories>\n{content}\n</relevant-memories>"


def _render_hook_result(result: str) -> str:
    return f"<before_run_hook_result>\n{result}\n</before_run_hook_result>"


def _prepend_to_user_message(msg: dict[str, Any], preamble: str) -> None:
    """Prepend preamble text to the last user message content."""
    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = preamble + "\n\n" + content
    elif isinstance(content, list):
        msg["content"] = [{"type": "text", "text": preamble}] + content
    else:
        msg["content"] = preamble


class MessageAssembler:
    """
    Assembles complete message lists for LLM calls.
    """

    @staticmethod
    def assemble(
        system_prompt: str,
        existing_steps: list[StepRecord] | None = None,
        memories: list[MemoryRecord] | None = None,
        before_run_hook_result: str | None = None,
        *,
        channel_context: ChannelContext | None = None,
    ) -> list[dict]:
        """
        Assemble the complete message list.

        All dynamic context (channel metadata, memories, hook result) is prepended
        to the last user message content to protect system prompt KV cache.

        Args:
            system_prompt: The static system prompt.
            existing_steps: Conversation history steps (user step already included).
            memories: Relevant memory records to inject.
            before_run_hook_result: Result from before-run hook if provided.
            channel_context: Structured channel metadata (source, chat info, history).

        Returns:
            List of message dicts ready for the LLM.
        """
        if existing_steps is None:
            existing_steps = []
        if memories is None:
            memories = []

        messages: list[dict] = steps_to_messages(existing_steps)

        filtered_memories = MessageAssembler._filter_memories(messages, memories)

        preamble_parts: list[str] = []
        if channel_context:
            preamble_parts.append(_render_channel_context(channel_context))
        if filtered_memories:
            preamble_parts.append(_render_memories(filtered_memories))
        if before_run_hook_result:
            preamble_parts.append(_render_hook_result(before_run_hook_result))

        if preamble_parts and messages:
            last_msg = messages[-1]
            if last_msg.get("role") == "user":
                preamble_text = "\n\n".join(preamble_parts)
                _prepend_to_user_message(last_msg, preamble_text)

        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        return messages

    @staticmethod
    def _filter_memories(
        messages: list[dict], memories: list[MemoryRecord]
    ) -> list[MemoryRecord]:
        if not memories:
            return []

        MIN_RELEVANCE_SCORE = 0.5
        SIMILARITY_THRESHOLD = 0.8

        existing_texts: list[str] = [
            msg.get("content", "") for msg in messages[:-1] if isinstance(msg.get("content"), str)
        ]

        def _is_similar_to_history(content: str) -> bool:
            content_lower = content.lower()
            for text in existing_texts:
                if (
                    MessageAssembler._text_similarity(content_lower, text.lower())
                    > SIMILARITY_THRESHOLD
                ):
                    return True
            return False

        filtered: list[MemoryRecord] = []
        seen_contents: set[str] = set()

        for memory in sorted(
            [m for m in memories if m.relevance_score is not None],
            key=lambda m: m.relevance_score or 0,
            reverse=True,
        ):
            if memory.relevance_score < MIN_RELEVANCE_SCORE:
                continue

            content_normalized = memory.content.strip()
            if content_normalized in seen_contents:
                continue
            seen_contents.add(content_normalized)

            if _is_similar_to_history(content_normalized):
                continue

            filtered.append(memory)

        return filtered

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        if a in b or b in a:
            return 0.9

        a_words = set(a.split())
        b_words = set(b.split())
        if not a_words or not b_words:
            return 0.0

        intersection = a_words & b_words
        union = a_words | b_words
        return len(intersection) / len(union)
