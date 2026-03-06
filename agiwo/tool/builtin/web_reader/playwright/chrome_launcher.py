"""
Chrome launcher module.

Handles launching and managing Chrome browser instances for web fetching.
"""

import os
import platform
import signal
import socket
import subprocess
import time

from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class ChromeLauncher:
    """Chrome browser launcher."""

    def __init__(self) -> None:
        self.system = platform.system()
        self.browser_process: subprocess.Popen[bytes] | None = None
        self.debug_port: int | None = None

    def detect_browser_paths(self) -> list[str]:
        """
        Detect available browser paths in the system.

        Returns:
            List of browser paths sorted by priority
        """
        paths = []

        if self.system == "Windows":
            # Common Chrome/Edge installation paths on Windows
            possible_paths = [
                # Chrome paths
                os.path.expandvars(
                    r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"
                ),
                os.path.expandvars(
                    r"%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe"
                ),
                os.path.expandvars(
                    r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"
                ),
                # Edge paths
                os.path.expandvars(
                    r"%PROGRAMFILES%\Microsoft\Edge\Application\msedge.exe"
                ),
                os.path.expandvars(
                    r"%PROGRAMFILES(X86)%\Microsoft\Edge\Application\msedge.exe"
                ),
                # Chrome Beta/Dev/Canary
                os.path.expandvars(
                    r"%LOCALAPPDATA%\Google\Chrome Beta\Application\chrome.exe"
                ),
                os.path.expandvars(
                    r"%LOCALAPPDATA%\Google\Chrome Dev\Application\chrome.exe"
                ),
                os.path.expandvars(
                    r"%LOCALAPPDATA%\Google\Chrome SxS\Application\chrome.exe"
                ),
            ]
        elif self.system == "Darwin":  # macOS
            # Common Chrome/Edge installation paths on macOS
            possible_paths = [
                # Chrome paths
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta",
                "/Applications/Google Chrome Dev.app/Contents/MacOS/Google Chrome Dev",
                "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
                # Edge paths
                "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                "/Applications/Microsoft Edge Beta.app/Contents/MacOS/Microsoft Edge Beta",
                "/Applications/Microsoft Edge Dev.app/Contents/MacOS/Microsoft Edge Dev",
                "/Applications/Microsoft Edge Canary.app/Contents/MacOS/Microsoft Edge Canary",
            ]
        else:
            # Linux and other systems
            possible_paths = [
                "/usr/bin/google-chrome",
                "/usr/bin/google-chrome-stable",
                "/usr/bin/google-chrome-beta",
                "/usr/bin/google-chrome-unstable",
                "/usr/bin/chromium-browser",
                "/usr/bin/chromium",
                "/snap/bin/chromium",
                "/usr/bin/microsoft-edge",
                "/usr/bin/microsoft-edge-stable",
                "/usr/bin/microsoft-edge-beta",
                "/usr/bin/microsoft-edge-dev",
            ]

        # Check if paths exist and are executable
        for path in possible_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                paths.append(path)

        return paths

    def find_free_port(self, start_port: int = 9222) -> int:
        """Find an available port."""
        for port in range(start_port, start_port + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("localhost", port)) != 0:
                    return port
        raise RuntimeError("No available port found")

    def launch_browser(
        self,
        browser_path: str,
        debug_port: int,
        headless: bool = False,
        user_data_dir: str | None = None,
    ) -> subprocess.Popen:
        """
        Launch browser process.
        """
        # Basic launch arguments
        args = [
            browser_path,
            f"--remote-debugging-port={debug_port}",
            "--remote-debugging-address=127.0.0.1",  # Only allow local access, fix security vulnerability
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-features=TranslateUI",
            "--disable-ipc-flooding-protection",
            "--disable-hang-monitor",
            "--disable-prompt-on-repost",
            "--disable-sync",
            "--disable-dev-shm-usage",  # Avoid shared memory issues
            "--no-sandbox",  # Disable sandbox in CDP mode
            # Key anti-detection parameters
            "--disable-blink-features=AutomationControlled",  # Disable automation control flag
            "--exclude-switches=enable-automation",  # Exclude automation switch
            "--disable-infobars",  # Disable info bars
            "--disable-setuid-sandbox",  # Disable setuid sandbox
        ]

        # Headless mode
        if headless:
            args.extend(
                [
                    "--headless=new",  # Use new headless mode
                    "--disable-gpu",
                ]
            )
        else:
            # Additional parameters for non-headless mode
            args.extend(
                [
                    "--start-maximized",  # Maximize window, more like real user
                ]
            )

        # User data directory
        if user_data_dir:
            args.append(f"--user-data-dir={user_data_dir}")

        logger.info(f"[BrowserLauncher] Launching browser: {browser_path}")
        logger.info(f"[BrowserLauncher] Debug port: {debug_port}")
        logger.info(f"[BrowserLauncher] Headless mode: {headless}")

        try:
            # On Windows, use CREATE_NEW_PROCESS_GROUP to avoid Ctrl+C affecting child process
            if self.system == "Windows":
                process = subprocess.Popen(
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # type: ignore[attr-defined]
                )
            else:
                process = subprocess.Popen(
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid,  # Create new process group
                )

            self.browser_process = process
            return process

        except Exception as e:
            logger.error(f"[BrowserLauncher] Failed to launch browser: {e}")
            raise

    def wait_for_browser_ready(self, debug_port: int, timeout: int = 30) -> bool:
        """
        Wait for browser to be ready.
        """
        logger.info(
            f"[BrowserLauncher] Waiting for browser to be ready on port {debug_port}..."
        )

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex(("localhost", debug_port))
                    if result == 0:
                        logger.info(
                            f"[BrowserLauncher] Browser is ready on port {debug_port}"
                        )
                        return True
            except Exception:
                pass

            time.sleep(0.5)

        logger.error(
            f"[BrowserLauncher] Browser failed to be ready within {timeout} seconds"
        )
        return False

    def get_browser_info(self, browser_path: str) -> tuple[str, str]:
        """
        Get browser information (name and version).
        """
        try:
            if "chrome" in browser_path.lower():
                name = "Google Chrome"
            elif "edge" in browser_path.lower() or "msedge" in browser_path.lower():
                name = "Microsoft Edge"
            elif "chromium" in browser_path.lower():
                name = "Chromium"
            else:
                name = "Unknown Browser"

            # Try to get version information
            try:
                result = subprocess.run(
                    [browser_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                version = result.stdout.strip() if result.stdout else "Unknown Version"
            except Exception:
                version = "Unknown Version"

            return name, version

        except Exception:
            return "Unknown Browser", "Unknown Version"

    def cleanup(self):
        """
        Clean up resources, close browser process.
        """
        if not self.browser_process:
            return

        process = self.browser_process

        if process.poll() is not None:
            logger.info(
                "[BrowserLauncher] Browser process already exited, no cleanup needed"
            )
            self.browser_process = None
            return

        logger.info("[BrowserLauncher] Closing browser process...")

        try:
            if self.system == "Windows":
                # Try normal termination first
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "[BrowserLauncher] Normal termination timeout, "
                        "using taskkill to force end"
                    )
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                        capture_output=True,
                        check=False,
                    )
                    process.wait(timeout=5)
            else:
                pgid = os.getpgid(process.pid)
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    logger.info(
                        "[BrowserLauncher] Browser process group does not exist, "
                        "may have exited"
                    )
                else:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning(
                            "[BrowserLauncher] Graceful shutdown timeout, "
                            "sending SIGKILL"
                        )
                        os.killpg(pgid, signal.SIGKILL)
                        process.wait(timeout=5)

            logger.info("[BrowserLauncher] Browser process closed")
        except Exception as e:
            logger.warning(f"[BrowserLauncher] Error closing browser process: {e}")
        finally:
            self.browser_process = None
