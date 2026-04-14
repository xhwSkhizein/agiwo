"""
Browser CLI adapter module.

Provides adapter for browser_cli task runtime to fetch web content.
"""

from browser_cli.task_runtime.read import ReadRequest, run_read_request

from agiwo.tool.builtin.config import WebReaderApiConfig
from agiwo.tool.builtin.html_extract import (
    HtmlContent,
    extract_content_from_html,
)
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class BrowserCliAdapter:
    """Adapter for browser_cli task runtime."""

    def __init__(self, config: WebReaderApiConfig | None = None) -> None:
        self._config = config or WebReaderApiConfig()

    async def fetch(self, url: str) -> tuple[HtmlContent | None, str | None]:
        """
        Fetch URL using browser_cli and return extracted HTML content.

        Args:
            url: Target URL to fetch

        Returns:
            Tuple of (HtmlContent or None, error message or None)
        """
        try:
            request = ReadRequest(url=url, output_mode="html")
            result = await run_read_request(request)
            html = result.body

            if not html or not html.strip():
                logger.warning("browser_cli returned empty HTML", url=url)
                return None, "Empty HTML content"

            content = extract_content_from_html(html=html, original_url=url)
            if content is None:
                logger.warning("Failed to extract content from HTML", url=url)
                return None, "Content extraction failed"

            logger.info("web_reader_fetch_succeeded", strategy="browser_cli", url=url)
            return content, None

        except Exception as exc:  # noqa: BLE001
            logger.error(f"Error fetching content with browser_cli: {exc}", url=url)
            return None, str(exc)
