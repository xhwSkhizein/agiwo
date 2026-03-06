"""
Curl-cffi HTTP client module.

Provides lightweight HTTP client for web content fetching using curl_cffi.
"""

from urllib.parse import urlparse

from curl_cffi import requests
from curl_cffi.requests import Response

from agiwo.tool.builtin.html_extract import (
    HtmlContent,
    extract_content_from_html,
)
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class SimpleAsyncClient:
    """
    Simple async HTTP client.

    Provides production-grade web scraping capabilities with automatic
    session lifecycle management, custom configuration, and context management.
    """

    def __init__(self) -> None:
        # Timeout in seconds
        self.timeout = 10
        self.impersonate = "chrome"

    async def fetch(self, url: str, **kwargs) -> HtmlContent | None:
        """
        Fetch URL and return extracted HTML content.

        Args:
            url: Target URL to fetch

        Returns:
            Extracted HtmlContent or None if fetch fails
        """
        try:
            referer: str = f"https://{urlparse(url).netloc}/"
            response: Response = requests.get(
                url,
                timeout=self.timeout,
                impersonate=self.impersonate,
                headers=kwargs.get(
                    "headers",
                    {
                        "Referer": referer,
                    },
                ),
                **kwargs,
            )
            if response.status_code != 200:
                logger.warning(
                    f"HTTP {response.status_code}: {response.text[:200]}", url=url
                )
                return None

            return extract_content_from_html(html=response.text, original_url=url)
        except Exception as e:
            logger.error(f"Error fetching content with curl_cffi: {e}", url=url)
            return None


# Global shared instance

_default_client: SimpleAsyncClient | None = None


async def get_default_client() -> SimpleAsyncClient:
    """Get global default client instance (lazy-loaded)."""
    global _default_client
    if _default_client is None:
        _default_client = SimpleAsyncClient()
    return _default_client


async def fetch(url: str) -> HtmlContent | None:
    """
    Simple global fetch function.

    Uses shared default client and session, suitable for single-call scenarios.
    For multiple requests, consider using SimpleAsyncClient context manager.
    """
    client = await get_default_client()
    return await client.fetch(url)
