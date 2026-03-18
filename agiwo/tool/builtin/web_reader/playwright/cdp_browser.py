"""
CDP browser management module.

Manages Chrome DevTools Protocol (CDP) browser connections for web fetching.
"""

import asyncio
import os
import socket
from typing import Any

import httpx
from playwright.async_api import Browser, BrowserContext, Playwright

from agiwo.tool.builtin.config import WebReaderApiConfig
from agiwo.utils.logging import get_logger

from .chrome_launcher import ChromeLauncher

logger = get_logger(__name__)


class CDPBrowserManager:
    """CDP browser manager."""

    def __init__(self, *, config: WebReaderApiConfig | None = None) -> None:
        self._config = config or WebReaderApiConfig()
        self.launcher = ChromeLauncher()
        self.browser: Browser | None = None
        self.browser_context: BrowserContext | None = None
        self.debug_port: int | None = None

    async def launch_and_connect(
        self,
        playwright: Playwright,
        playwright_proxy: dict | None = None,
        user_agent: str | None = None,
        headless: bool = True,
    ) -> BrowserContext:
        """
        Launch browser and connect via CDP.
        """
        try:
            # 1. Detect browser path
            browser_path = await self._get_browser_path()

            # 2. Get available port
            self.debug_port = self.launcher.find_free_port(start_port=19222)

            # 3. Launch browser
            await self._launch_browser(browser_path, headless)

            # 4. Connect via CDP
            await self._connect_via_cdp(playwright)

            # 5. Create browser context
            browser_context = await self._create_browser_context(
                playwright_proxy, user_agent
            )

            self.browser_context = browser_context
            return browser_context

        except Exception as e:  # noqa: BLE001
            logger.error(f"[CDPBrowserManager] CDP browser launch failed: {e}")
            await self.cleanup()
            raise

    async def _get_browser_path(self) -> str:
        """
        Get browser path.
        """
        # Auto-detect browser path
        browser_paths = self.launcher.detect_browser_paths()

        if not browser_paths:
            raise RuntimeError(
                "No available browser found. Please ensure Chrome or Edge is installed, "
                "or set CUSTOM_BROWSER_PATH in config file to specify browser path."
            )

        browser_path = browser_paths[0]  # Use first found browser
        browser_name, browser_version = self.launcher.get_browser_info(browser_path)

        logger.info(
            f"[CDPBrowserManager] Detected browser: {browser_name} ({browser_version})"
        )
        logger.info(f"[CDPBrowserManager] Browser path: {browser_path}")

        return browser_path

    async def _test_cdp_connection(self, debug_port: int) -> bool:
        """
        Test if CDP connection is available.
        """
        try:
            # Simple socket connection test
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                result = s.connect_ex(("localhost", debug_port))
                if result == 0:
                    logger.info(
                        f"[CDPBrowserManager] CDP port {debug_port} is accessible"
                    )
                    return True
                else:
                    logger.warning(
                        f"[CDPBrowserManager] CDP port {debug_port} is not accessible"
                    )
                    return False
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[CDPBrowserManager] CDP connection test failed: {e}")
            return False

    async def _launch_browser(self, browser_path: str, headless: bool):
        """
        Launch browser process.
        """
        save_login_state = self._config.save_login_state
        browser_launch_timeout = self._config.browser_launch_timeout
        # Set user data directory (if save login state is enabled)
        user_data_dir = None
        if save_login_state:
            user_data_dir = os.path.join(
                os.getcwd(),
                self._config.user_data_dir,
            )
            os.makedirs(user_data_dir, exist_ok=True)
            logger.info(f"[CDPBrowserManager] User data directory: {user_data_dir}")

        # Launch browser
        if self.debug_port is None:
            raise RuntimeError("Debug port not set")
        self.launcher.browser_process = self.launcher.launch_browser(
            browser_path=browser_path,
            debug_port=self.debug_port,
            headless=headless,
            user_data_dir=user_data_dir,
        )

        # Wait for browser to be ready
        if self.debug_port is None:
            raise RuntimeError("Debug port not set")
        if not self.launcher.wait_for_browser_ready(
            self.debug_port, browser_launch_timeout
        ):
            raise RuntimeError(
                f"Browser failed to start within {browser_launch_timeout} seconds"
            )

        # Wait an extra second for CDP service to fully start
        await asyncio.sleep(1)

        # Test CDP connection
        if self.debug_port is None:
            raise RuntimeError("Debug port not set")
        if not await self._test_cdp_connection(self.debug_port):
            logger.warning(
                "[CDPBrowserManager] CDP connection test failed, "
                "but will continue to try connecting"
            )

    async def _get_browser_websocket_url(self, debug_port: int) -> str:
        """
        Get browser WebSocket connection URL.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{debug_port}/json/version", timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    ws_url = data.get("webSocketDebuggerUrl")
                    if ws_url and isinstance(ws_url, str):
                        logger.info(
                            f"[CDPBrowserManager] Got browser WebSocket URL: {ws_url}"
                        )
                        return ws_url
                    else:
                        raise RuntimeError("webSocketDebuggerUrl not found")
                else:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"[CDPBrowserManager] Failed to get WebSocket URL: {e}")
            raise

    async def _connect_via_cdp(self, playwright: Playwright):
        """
        Connect to browser via CDP.
        """
        try:
            # Get correct WebSocket URL
            if self.debug_port is None:
                raise RuntimeError("Debug port not set")
            ws_url = await self._get_browser_websocket_url(self.debug_port)
            logger.info(f"[CDPBrowserManager] Connecting to browser via CDP: {ws_url}")

            # Use Playwright's connectOverCDP method to connect
            self.browser = await playwright.chromium.connect_over_cdp(ws_url)

            if self.browser.is_connected():
                logger.info("[CDPBrowserManager] Successfully connected to browser")
                logger.info(
                    f"[CDPBrowserManager] Browser context count: "
                    f"{len(self.browser.contexts)}"
                )
            else:
                raise RuntimeError("CDP connection failed")

        except Exception as e:  # noqa: BLE001
            logger.error(f"[CDPBrowserManager] CDP connection failed: {e}")
            raise

    async def _create_browser_context(
        self, playwright_proxy: dict | None = None, user_agent: str | None = None
    ) -> BrowserContext:
        """
        Create or get browser context.
        """
        if not self.browser:
            raise RuntimeError("Browser not connected")

        # Get existing context or create new context
        contexts = self.browser.contexts

        if contexts:
            # Use existing first context
            browser_context = contexts[0]
            logger.info("[CDPBrowserManager] Using existing browser context")
        else:
            # Create new context
            context_options = {
                "viewport": {"width": 1920, "height": 1080},
                "accept_downloads": True,
            }

            # Set user agent
            if user_agent:
                context_options["user_agent"] = user_agent
                logger.info(f"[CDPBrowserManager] Set user agent: {user_agent}")

            # Note: Proxy settings may not work in CDP mode since browser is already launched
            if playwright_proxy:
                logger.warning(
                    "[CDPBrowserManager] Warning: Proxy settings may not work in CDP mode, "
                    "suggest configuring system proxy or browser proxy extension before browser launch"
                )

            browser_context = await self.browser.new_context(**context_options)  # type: ignore[arg-type]
            logger.info("[CDPBrowserManager] Created new browser context")

        return browser_context

    async def add_stealth_script(self, script_path: str = "libs/stealth.min.js"):
        """
        Add anti-detection script.
        """
        if self.browser_context and os.path.exists(script_path):
            try:
                await self.browser_context.add_init_script(path=script_path)
                logger.info(
                    f"[CDPBrowserManager] Added anti-detection script: {script_path}"
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"[CDPBrowserManager] Failed to add anti-detection script: {e}"
                )

    async def add_cookies(self, cookies: list):
        """
        Add cookies.
        """
        if self.browser_context:
            try:
                await self.browser_context.add_cookies(cookies)
                logger.info(f"[CDPBrowserManager] Added {len(cookies)} cookies")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[CDPBrowserManager] Failed to add cookies: {e}")

    async def get_cookies(self) -> list:
        """
        Get current cookies.
        """
        if self.browser_context:
            try:
                cookies = await self.browser_context.cookies()
                return cookies
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[CDPBrowserManager] Failed to get cookies: {e}")
                return []
        return []

    async def cleanup(self):
        """
        Clean up resources.
        """
        try:
            # Close browser context
            if self.browser_context:
                try:
                    await self.browser_context.close()
                    logger.info("[CDPBrowserManager] Browser context closed")
                except Exception as context_error:  # noqa: BLE001
                    logger.warning(
                        f"[CDPBrowserManager] Failed to close browser context: "
                        f"{context_error}"
                    )
                finally:
                    self.browser_context = None

            # Disconnect browser connection
            if self.browser:
                try:
                    await self.browser.close()
                    logger.info("[CDPBrowserManager] Browser connection disconnected")
                except Exception as browser_error:  # noqa: BLE001
                    logger.warning(
                        f"[CDPBrowserManager] Failed to close browser connection: {browser_error}"
                    )
                finally:
                    self.browser = None

            # Close browser process (if configured to auto-close)
            if self._config.auto_close_browser:
                self.launcher.cleanup()
            else:
                logger.info(
                    "[CDPBrowserManager] Browser process kept running (AUTO_CLOSE_BROWSER=False)"
                )

        except Exception as e:  # noqa: BLE001
            logger.error(f"[CDPBrowserManager] Error during cleanup: {e}")

    def is_connected(self) -> bool:
        """
        Check if connected to browser.
        """
        return self.browser is not None and self.browser.is_connected()

    async def get_browser_info(self) -> dict[str, Any]:
        """Get browser information."""
        if not self.browser:
            return {}

        try:
            version = self.browser.version
            contexts_count = len(self.browser.contexts)

            return {
                "version": version,
                "contexts_count": contexts_count,
                "debug_port": self.debug_port,
                "is_connected": self.is_connected(),
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to get browser info", error=str(e))
            return {}
