"""
MessageAssembler - Assembles complete LLM message lists.
"""

from agiwo.agent.schema import StepRecord, steps_to_messages, MemoryRecord


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
    ) -> list[dict]:
        """
        Assemble the complete message list.

        Args:
            system_prompt: Optional system prompt to include at the beginning.
            existing_steps: Conversation history steps.
            user_input: User input.
            memories: List of memory records that are relevant to the user input.
            before_run_hook_result: Result from before-run hook if provided.

        Returns:
            List of message dicts ready for the LLM.
        """
        if existing_steps is None:
            existing_steps = []
        if memories is None:
            memories = []

        # use step is include in messages
        messages: list[dict] = steps_to_messages(existing_steps)

        momos: str | None = MessageAssembler._build_memory_injection_msg(
            messages, memories
        )
        last_user_input = messages[-1]
        if momos:
            if last_user_input["role"] == "user":
                last_user_input["content"] += (
                    "\n\n<relevant_memories>" + momos + "</relevant_memories>"
                )

        # re-think about this
        if before_run_hook_result:
            last_user_input["content"] += (
                "\n\n<before_run_hook_result>"
                + before_run_hook_result
                + "</before_run_hook_result>"
            )

        # Insert system prompt if provided
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        return messages

    @staticmethod
    def _build_memory_injection_msg(
        messages: list[dict], memories: list[MemoryRecord]
    ) -> str | None:
        if not memories:
            return None

        MIN_RELEVANCE_SCORE = 0.5
        SIMILARITY_THRESHOLD = 0.8

        existing_texts: list[str] = [
            msg.get("content", "") for msg in messages[:-1] if msg.get("content")
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

        filtered_memories: list[MemoryRecord] = []
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

            filtered_memories.append(memory)

        if not filtered_memories:
            return None

        return "\n\n".join([m.content for m in filtered_memories])

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
