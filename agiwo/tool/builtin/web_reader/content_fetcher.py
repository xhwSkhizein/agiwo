from agiwo.tool.builtin.html_extract import HtmlContent
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class WebContentFetcher:
    def __init__(self, *, client, crawler) -> None:
        self._client = client
        self._crawler = crawler

    async def fetch(self, url: str) -> tuple[HtmlContent | None, str | None]:
        content = await self._client.fetch(url)
        if content is not None:
            logger.info("web_reader_fetch_succeeded", strategy="curl_cffi", url=url)
            return content, None

        logger.info("web_reader_fetch_fallback_playwright", url=url)
        started = False
        try:
            await self._crawler.start()
            started = True
            return await self._crawler.crawl_url(url), None
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)
        finally:
            if started:
                await self._crawler.stop()
