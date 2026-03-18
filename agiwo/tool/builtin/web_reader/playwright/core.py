"""
Playwright core web fetching module.

Provides core functionality for fetching web content using Playwright.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from playwright.async_api import Page

from agiwo.tool.builtin.config import WebReaderApiConfig
from agiwo.tool.builtin.html_extract import (
    HtmlContent,
    extract_content_from_html,
)
from agiwo.tool.builtin.web_reader.playwright.chrome_session import (
    ChromeSessionManager,
)
from agiwo.tool.builtin.web_reader.playwright.browser_pool import (
    BrowserLease,
    BrowserPool,
)
from agiwo.tool.builtin.web_reader.playwright.exceptions import (
    BlockedException,
    SessionInvalidException,
)
from agiwo.utils.logging import get_logger


class PlaywrightCrawler:
    """Production-grade web crawler."""

    def __init__(self, config: WebReaderApiConfig | None = None) -> None:
        self.logger = get_logger(__name__)
        self._config = config or WebReaderApiConfig()
        self.session_manager: ChromeSessionManager | None = None
        self._pool = BrowserPool(config=self._config)
        self._lease: BrowserLease | None = None
        self._start_lock = asyncio.Lock()
        self._started = False
        self.site_configs: dict[str, dict[str, Any]] = {
            "wechat": {
                "login_url": "https://mp.weixin.qq.com/",
                "content_selectors": ["#js_content", ".rich_media_content"],
                "title_selectors": ["#activity-name", ".rich_media_title"],
                "auth_indicators": [".user_info", ".account_meta_value"],
                "name": "微信公众号",
            },
            "zhihu": {
                "login_url": "https://www.zhihu.com/",
                "content_selectors": [".RichContent-inner", ".Post-RichText"],
                "title_selectors": [".QuestionHeader-title", ".Post-Title"],
                "auth_indicators": [".AppHeader-userInfo", ".Avatar"],
                "name": "知乎",
            },
            "weibo": {
                "login_url": "https://weibo.com/",
                "content_selectors": [".WB_text", ".WB_detail"],
                "title_selectors": [".WB_text", ".WB_info"],
                "auth_indicators": [".gn_name", ".username"],
                "name": "微博",
            },
        }

        self.stats: dict[str, int | float | None] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "blocked_requests": 0,
            "start_time": None,
        }

    async def __aenter__(self):
        """Context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.stop()

    async def start(self):
        """Start crawler (thread-safe, ensures single start)."""
        async with self._start_lock:
            # If already started, return directly
            if (
                self._started
                and self.session_manager
                and self.session_manager.is_connected()
            ):
                return

            self.stats["start_time"] = time.time()
            self.logger.info("Starting production-grade crawler")
            if not self._lease:
                self._lease = await self._pool.acquire()
            self.session_manager = self._lease.session_manager
            self._started = True

    async def stop(self):
        """Stop crawler."""
        async with self._start_lock:
            if not self._started:
                return

            self.logger.info("Stopping crawler")

            # Print statistics
            self._print_stats()

            # Release lease back to pool
            if self._lease:
                broken = (
                    not self.session_manager or not self.session_manager.is_connected()
                )
                await self._pool.release(self._lease, broken=broken)
                self._lease = None
                self.session_manager = None

            self._started = False

    def _print_stats(self):
        """Print statistics."""
        start_time = self.stats["start_time"]
        if start_time is None:
            return

        duration = time.time() - start_time
        total_requests = self.stats["total_requests"]
        successful_requests = self.stats["successful_requests"]
        success_rate = (
            (successful_requests / total_requests * 100) if total_requests > 0 else 0
        )

        self.logger.info(
            f"\nCrawler statistics:\n"
            f"   Total requests: {self.stats['total_requests']}\n"
            f"   Successful: {self.stats['successful_requests']}\n"
            f"   Failed: {self.stats['failed_requests']}\n"
            f"   Blocked: {self.stats['blocked_requests']}\n"
            f"   Success rate: {success_rate:.1f}%\n"
            f"   Runtime: {duration:.1f}s"
        )

    def _is_blocked(self, page: Page) -> bool:
        """Check if request is blocked."""
        url = page.url.lower()
        blocked_keywords = [
            "captcha",
            "verify",
            "validation",
            "robots",
            "checkpoint",
            "challenge",
            "recaptcha",
        ]
        return any(keyword in url for keyword in blocked_keywords)

    async def _extract_content(self, page: Page, url: str) -> HtmlContent | None:
        """Extract page content."""

        # FIXME use Trafilatura to extract content
        original_html = await page.content()

        content = extract_content_from_html(html=original_html, original_url=url)
        if not content:
            return None
        return content

    async def crawl_url(self, url: str, retries: int = 0) -> HtmlContent | None:
        """
        Crawl a single URL.

        Args:
            url: Target URL
            retries: Number of retries

        Returns:
            Extracted content data or None
        """
        if self.session_manager is None or not self.session_manager.is_connected():
            raise SessionInvalidException("Not connected to Chrome")

        self.stats["total_requests"] += 1
        self.logger.info(f"Crawling: {url}")
        page: Page | None = None
        try:
            if self.session_manager.context is None:
                raise SessionInvalidException("Browser context not available")
            page = await self.session_manager.context.new_page()
            # Set page timeout
            page.set_default_timeout(self._config.timeout_seconds * 1000)

            # Visit page
            wait_strategy = self._config.wait_strategy
            if isinstance(wait_strategy, str):
                # Convert string to Literal type for Playwright
                wait_strategy = wait_strategy  # type: ignore[arg-type]
            response = await page.goto(url, wait_until=wait_strategy)  # type: ignore[arg-type]

            if not response or response.status != 200:
                self.logger.warning(
                    f"HTTP status abnormal: {response.status if response else 'None'}"
                )

                self.stats["failed_requests"] += 1
                return None

            # Check if blocked
            if self._is_blocked(page):
                self.logger.warning(f"Request blocked: {page.url}")
                self.stats["blocked_requests"] += 1
                raise BlockedException("Request blocked")

            # Extract content
            content = await self._extract_content(page, url)
            self.stats["successful_requests"] += 1
            self.logger.info(
                f"Crawl successful: title:{content.title if content else 'None'}",
                url=url,
            )

            return content

        except BlockedException:
            if retries < self._config.max_retries:
                self.logger.info(f"Retrying {retries + 1}/{self._config.max_retries}")
                # Exponential backoff
                await asyncio.sleep(5 * (retries + 1))
                return await self.crawl_url(url, retries + 1)

        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Crawl failed: {e}", url=url)
            failed_requests = self.stats.get("failed_requests", 0)
            if isinstance(failed_requests, int):
                self.stats["failed_requests"] = failed_requests + 1

            # Health check, reconnect if connection is broken
            if self.session_manager is None:
                raise SessionInvalidException("Session manager not available") from None
            if not await self.session_manager.health_check():
                self.logger.info(
                    "Connection broken detected, attempting to reconnect..."
                )
                await self.session_manager.connect()
            return None

        finally:
            if page:
                await page.close()

    async def crawl_batch(
        self, urls: list[str], save_dir: Path | None = None
    ) -> list[dict[str, Any]]:
        """
        Crawl URLs in batch.

        Args:
            urls: List of URLs
            save_dir: Save directory

        Returns:
            List of successfully crawled content
        """
        results = []
        save_dir = save_dir or Path("crawled_data")
        save_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Starting batch crawl, total {len(urls)} URLs")

        for i, url in enumerate(urls, 1):
            self.logger.info(f"\n[{i}/{len(urls)}] Processing {url}")

            try:
                content = await self.crawl_url(url, retries=self._config.max_retries)
                if content:
                    results.append(content)

                    # Save to file
                    file_name = f"{content['timestamp']}_{hash(url)}.json"
                    (save_dir / file_name).write_text(
                        json.dumps(content, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            except Exception as e:  # noqa: BLE001
                self.logger.error(f"Failed to process URL: {e}")

        self.logger.info(
            f"Batch crawl completed, successful {len(results)}/{len(urls)}"
        )
        return results
