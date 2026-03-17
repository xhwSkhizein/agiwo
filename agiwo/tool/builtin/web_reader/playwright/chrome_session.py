"""
Chrome session management module.

Manages browser sessions and contexts for Playwright-based web fetching.
"""

import asyncio
import os

from playwright.async_api import (
    BrowserContext,
    BrowserType,
    Playwright,
    async_playwright,
)

from agiwo.tool.builtin.config import WebReaderApiConfig
from agiwo.tool.builtin.web_reader.playwright.cdp_browser import (
    CDPBrowserManager,
)
from agiwo.tool.builtin.web_reader.playwright.exceptions import (
    SessionInvalidException,
)
from agiwo.config.settings import settings
from agiwo.utils.logging import get_logger


class ChromeSessionManager:
    """Chrome session manager."""

    def __init__(self, *, config: WebReaderApiConfig | None = None) -> None:
        self._config = config or WebReaderApiConfig()
        self.logger = get_logger(__name__)
        self.cdp_browser: CDPBrowserManager | None = None
        self.context: BrowserContext | None = None
        self.playwright: Playwright | None = None
        self._connected = False
        self._connect_lock = asyncio.Lock()

    async def connect(self) -> bool:
        """Connect to Chrome browser (thread-safe, ensures single connection)."""
        async with self._connect_lock:
            # If already connected and connection is valid, return directly
            if self._connected and self.is_connected():
                return True

            try:
                # If playwright exists but connection is broken, clean up first
                if self.playwright and not self.is_connected():
                    await self.disconnect()

                self.playwright = await async_playwright().start()
                self.logger.info("[ChromeSessionManager] Starting browser in CDP mode")
                if self.playwright is None:
                    raise RuntimeError("Failed to start Playwright")
                self.context = await self.launch_browser_with_cdp(
                    self.playwright,
                    None,
                    None,
                    headless=self._config.headless,
                )
                self._connected = True
                self.logger.info("Successfully connected to Chrome")
                return True

            except Exception as e:
                self.logger.error(f"Connection failed: {e}")
                if self.cdp_browser:
                    self.logger.error(
                        "Please ensure Chrome is started with remote debugging: "
                        f"chrome --remote-debugging-port={self.cdp_browser.debug_port}"
                    )
                raise SessionInvalidException(f"Failed to connect to Chrome: {e}")

    async def launch_browser(
        self,
        chromium: BrowserType,
        playwright_proxy: dict | None,
        user_agent: str | None,
        headless: bool = True,
    ) -> BrowserContext:
        """Launch browser and create browser context"""
        if self._config.save_login_state:
            user_data_dir = self._config.user_data_dir
            context = await chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                accept_downloads=True,
                headless=headless,
                proxy=playwright_proxy,  # type: ignore[arg-type]
                viewport={"width": 1920, "height": 1080},
                user_agent=user_agent,
            )
            return context
        else:
            browser = await chromium.launch(headless=headless, proxy=playwright_proxy)  # type: ignore[arg-type]
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080}, user_agent=user_agent
            )
            return context

    async def launch_browser_with_cdp(
        self,
        playwright: Playwright,
        playwright_proxy: dict | None,
        user_agent: str | None,
        headless: bool = True,
    ) -> BrowserContext:
        """Launch browser in CDP mode."""
        try:
            self.cdp_browser = CDPBrowserManager(config=self._config)
            context = await self.cdp_browser.launch_and_connect(
                playwright=playwright,
                playwright_proxy=playwright_proxy,
                user_agent=user_agent,
                headless=headless,
            )
            self.context = context
            # Add anti-detection scripts
            await self.cdp_browser.add_stealth_script()

            # Display browser information
            browser_info = await self.cdp_browser.get_browser_info()
            self.logger.info(f"CDP browser info: {browser_info}")

            return self.context

        except Exception as e:
            self.logger.error(
                f"CDP mode launch failed, falling back to standard mode: {e}"
            )
            # Fall back to standard mode
            chromium = playwright.chromium
            context = await self.launch_browser(
                chromium, playwright_proxy, user_agent, headless
            )
            self.context = context
            return context

    async def disconnect(self):
        """Disconnect (thread-safe)."""
        async with self._connect_lock:
            if not self._connected:
                return

            if self.playwright:
                await self.playwright.stop()
                self.playwright = None
            if self.cdp_browser:
                await self.cdp_browser.cleanup()
                self.cdp_browser = None
            self.context = None
            self._connected = False

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected and self.context is not None

    async def health_check(self) -> bool:
        """Health check."""
        if not self.is_connected():
            return False

        try:
            # Try creating a new page to test connection
            if self.context is None:
                return False
            page = await self.context.new_page()
            await page.close()
            return True
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            self._connected = False
            return False
