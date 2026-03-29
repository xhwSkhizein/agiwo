"""
通用 HTTP 客户端工具类。

遵循单一职责原则，专门处理 HTTP 请求。
"""

import asyncio
from typing import Any

import httpx
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class AsyncHttpClient:
    """异步 HTTP 客户端，用于调用外部 API。

    遵循依赖倒置原则，客户端不依赖具体的 API 实现。
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        """初始化 HTTP 客户端。

        Args:
            base_url: API 基础 URL
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries

    async def post_json(
        self,
        endpoint: str,
        data: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """发送 POST JSON 请求。

        Args:
            endpoint: API 端点路径
            data: 请求数据
            headers: 可选的自定义请求头

        Returns:
            API 响应的 JSON 数据

        Raises:
            httpx.HTTPError: HTTP 请求失败
            ValueError: 响应不是有效的 JSON
        """
        url = f"{self._base_url}/{endpoint.lstrip('/')}"

        default_headers = {"Content-Type": "application/json"}
        if headers:
            default_headers.update(headers)

        last_error = None
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(self._max_retries):
                try:
                    logger.debug(
                        "http_request_attempt",
                        url=url,
                        attempt=attempt + 1,
                        max_retries=self._max_retries,
                    )

                    response = await client.post(
                        url,
                        json=data,
                        headers=default_headers,
                    )
                    response.raise_for_status()
                    result = response.json()

                    logger.info(
                        "http_request_success",
                        url=url,
                        status_code=response.status_code,
                    )
                    return result

                except httpx.HTTPError as e:
                    last_error = e
                    logger.warning(
                        "http_request_failed",
                        url=url,
                        attempt=attempt + 1,
                        error=str(e),
                    )

                    if attempt < self._max_retries - 1:
                        await asyncio.sleep(2**attempt)
                    continue

                except Exception as e:
                    logger.error(
                        "http_request_unexpected_error",
                        url=url,
                        error=str(e),
                    )
                    raise

        error_msg = (
            f"HTTP request failed after {self._max_retries} attempts: {last_error}"
        )
        logger.error("http_request_max_retries_exceeded", url=url, error=error_msg)
        raise last_error or Exception(error_msg)
