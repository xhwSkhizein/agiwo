"""
Web Fetch Tool - Advanced web content extraction tool.

Provides dual web scraping solution based on curl_cffi and browser_cli:
- curl_cffi: Lightweight HTTP client, suitable for static pages
- browser_cli: Browser automation, suitable for JavaScript-rendered pages
"""

from .web_reader_tool import WebReaderTool

__all__ = ["WebReaderTool"]
