from agiwo.tool.builtin.html_extract import HtmlContent
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class WebContentFetcher:
    def __init__(self, *, client, browser_cli_adapter) -> None:
        self._client = client
        self._browser_cli_adapter = browser_cli_adapter

    async def fetch(self, url: str) -> tuple[HtmlContent | None, str | None]:
        content = await self._client.fetch(url)
        if content is not None:
            logger.info("web_reader_fetch_succeeded", strategy="curl_cffi", url=url)
            return content, None

        logger.info("web_reader_fetch_fallback_browser_cli", url=url)
        return await self._browser_cli_adapter.fetch(url)
