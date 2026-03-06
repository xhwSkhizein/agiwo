"""
Playwright configuration management.
"""

from dataclasses import dataclass


@dataclass
class PlaywrightConfig:
    """Playwright configuration."""

    # Browser configuration
    headless: bool = True
    """Headless mode."""

    browser_type: str = "chromium"
    """Browser type."""

    browser_path: str | None = None
    """Custom browser path."""

    # Page configuration
    viewport_width: int = 1920
    """Viewport width."""

    viewport_height: int = 1080
    """Viewport height."""

    # Performance optimization
    disable_images: bool = True
    """Disable image loading."""

    disable_fonts: bool = True
    """Disable font loading."""

    disable_css: bool = False
    """Disable CSS."""

    disable_javascript: bool = False
    """Disable JavaScript."""

    # Wait configuration
    wait_until: str = "domcontentloaded"
    """Page load wait condition."""

    navigation_timeout: int = 30000
    """Navigation timeout (milliseconds)."""

    # Security configuration
    disable_sandbox: bool = False
    """Disable sandbox."""

    restrict_debug_access: bool = True
    """Restrict debug access."""
