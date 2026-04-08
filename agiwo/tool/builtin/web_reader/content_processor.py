import asyncio

from agiwo.llm import ModelSpec, create_model
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class WebContentProcessor:
    def __init__(self, *, model_config: ModelSpec, max_length: int) -> None:
        self._model_config = model_config
        self._max_length = max_length
        self._llm_model = None
        self._model_init_lock = asyncio.Lock()

    @property
    def llm_model(self):
        return self._llm_model

    async def process(
        self,
        content: str,
        *,
        summarize: bool,
        search_query: str | None,
        abort_signal: AbortSignal | None,
        llm_model=None,
    ) -> str:
        if llm_model is not None:
            self._llm_model = llm_model
        if summarize or len(content) > self._max_length:
            return await self._summarize_content(
                content,
                abort_signal=abort_signal,
            )
        if search_query:
            return await self._extract_by_query(
                content,
                search_query,
                abort_signal=abort_signal,
            )
        return content

    async def _run_model_prompt(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback_text: str,
        abort_signal: AbortSignal | None = None,
    ) -> str:
        if abort_signal and abort_signal.is_aborted():
            return fallback_text

        try:
            model = await self._get_llm_model()
        except Exception as exc:  # noqa: BLE001
            logger.error("web_reader_model_init_failed", error=str(exc))
            return fallback_text

        content_parts: list[str] = []
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            async for chunk in model.arun_stream(messages=messages, tools=None):
                if abort_signal and abort_signal.is_aborted():
                    logger.info("web_reader_model_processing_aborted")
                    return fallback_text
                if chunk.content:
                    content_parts.append(chunk.content)
        except Exception as exc:  # noqa: BLE001
            logger.error("web_reader_model_processing_failed", error=str(exc))
            return fallback_text

        result = "".join(content_parts).strip()
        return result or fallback_text

    async def _get_llm_model(self):
        """Get or lazily initialize the LLM model with async lock protection."""
        if self._llm_model is not None:
            return self._llm_model

        async with self._model_init_lock:
            # Double-check pattern: another coroutine may have initialized while waiting
            if self._llm_model is None:
                self._llm_model = create_model(self._model_config)

        return self._llm_model

    async def _summarize_content(
        self,
        text: str,
        abort_signal: AbortSignal | None = None,
    ) -> str:
        system_prompt = """You are a professional content summarization assistant.
Your task is to:
1. Extract the core information and key insights from the content user provided
2. Maintain factual accuracy and preserve important details
3. Keep the summary comprehensive but concise (aim for 25-35% of original length)
"""
        user_prompt = f"""summarize the following content:

{text}
"""
        return await self._run_model_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback_text=text,
            abort_signal=abort_signal,
        )

    async def _extract_by_query(
        self,
        text: str,
        query: str,
        abort_signal: AbortSignal | None = None,
    ) -> str:
        system_prompt = """You are a professional content extraction assistant.
Your task is to:
1. Extract content specifically relevant to the user's query from the content user provided
2. Maintain full context and accuracy of extracted information
3. Preserve technical details, data, and important specifics
4. If no relevant content exists, clearly state so
5. Focus on substantive information, not just keyword matches
"""
        user_prompt = f"""the content is:

{text}

user's query is:
{query}
"""
        return await self._run_model_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback_text=text,
            abort_signal=abort_signal,
        )
