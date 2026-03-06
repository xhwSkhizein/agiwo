"""
LLM API 客户端。

用于调用统一的 LLM API 进行文本处理（总结、提取等）。
遵循单一职责原则。
"""

from agiwo.utils.abort_signal import AbortSignal
from agiwo.tool.builtin.http_client import AsyncHttpClient
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


def _extract_content(response: dict) -> str:
    """Extract text content from LLM API response.

    Supports:
    - OpenAI-style: choices[0].message.content
    - Legacy: content, response
    """
    content = response.get("content") or response.get("response")
    if content:
        return str(content)
    choices = response.get("choices")
    if choices and len(choices) > 0:
        msg = choices[0].get("message")
        if isinstance(msg, dict):
            c = msg.get("content")
            if c is not None:
                return str(c)
    return ""


class LlmApiClient:
    """LLM API 客户端，用于文本处理任务。"""

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        """初始化 LLM API 客户端。
        
        Args:
            base_url: API 基础 URL
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self._http_client = AsyncHttpClient(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def summarize(
        self,
        text: str,
        abort_signal: AbortSignal | None = None,
    ) -> str:
        """使用 LLM 总结文本。
        
        Args:
            text: 要总结的文本
            abort_signal: 中止信号
            
        Returns:
            总结后的文本
        """
        if abort_signal and abort_signal.is_aborted():
            return text

        system_prompt = """You are a professional content summarization assistant.
Your task is to:
1. Extract the core information and key insights from the content user provided
2. Maintain factual accuracy and preserve important details
3. Keep the summary comprehensive but concise (aim for 25-35% of original length)
"""

        user_message = f"""summarize the following content:

{text}
"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            response = await self._http_client.post_json(
                endpoint="/api/ai/llm",
                data={"messages": messages},
            )

            summary = _extract_content(response)
            if summary:
                logger.info("llm_summarization_success", text_length=len(text), summary_length=len(summary))
                return summary
            else:
                logger.warning("llm_summarization_empty_response")
                return text

        except Exception as e:
            logger.error("llm_summarization_failed", error=str(e))
            # 失败时返回原文
            return text

    async def extract_by_query(
        self,
        text: str,
        query: str,
        abort_signal: AbortSignal | None = None,
    ) -> str:
        """使用 LLM 根据查询提取相关内容。
        
        Args:
            text: 源文本
            query: 查询字符串
            abort_signal: 中止信号
            
        Returns:
            提取的相关内容
        """
        if abort_signal and abort_signal.is_aborted():
            return text

        system_prompt = """You are a professional content extraction assistant.
Your task is to:
1. Extract content specifically relevant to the user's query from the content user provided
2. Maintain full context and accuracy of extracted information
3. Preserve technical details, data, and important specifics
4. If no relevant content exists, clearly state so
5. Focus on substantive information, not just keyword matches
"""

        user_message = f"""the content is:

{text}

user's query is:
{query}
"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            response = await self._http_client.post_json(
                endpoint="/api/ai/llm",
                data={"messages": messages},
            )

            extracted = _extract_content(response)
            if extracted:
                logger.info(
                    "llm_extraction_success",
                    query=query,
                    text_length=len(text),
                    extracted_length=len(extracted),
                )
                return extracted
            else:
                logger.warning("llm_extraction_empty_response", query=query)
                return text

        except Exception as e:
            logger.error("llm_extraction_failed", query=query, error=str(e))
            # 失败时返回原文
            return text
