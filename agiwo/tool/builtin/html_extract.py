"""
HTML content extraction module.

Provides utilities for extracting and preprocessing HTML content,
including lazy-loaded image handling and content cleaning.
"""

import json
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Comment
from pydantic import BaseModel
from trafilatura import extract
from trafilatura.settings import use_config

from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

config = use_config()

# Adjust language detection threshold (commented out)
# config.set("DEFAULT", "language_score_threshold", "0.5")
# Skip URLs longer than 2048 characters
config.set("DEFAULT", "image_len_threshold", "2048")
# Prevent image verification timeout
config.set("DEFAULT", "timeout", "10")
# Disable precision-first mode
config.set("DEFAULT", "favor_precision", "False")
# Enable fallback strategy
config.set("DEFAULT", "no_fallback", "False")
# Reduce minimum text length from default 10-15 to 5 characters
config.set("DEFAULT", "min_extracted_size", "5")
config.set("DEFAULT", "min_output_size", "5")
# Increase link density tolerance
# Allow up to 60% link density
config.set("DEFAULT", "link_density_max", "0.6")


class HtmlContent(BaseModel):
    """
    Extracted HTML content model.

    Contains structured content extracted from HTML pages,
    including metadata, text content, and media references.
    """

    title: str | None = None
    author: str | None = None
    hostname: str | None = None
    date: str | None = None
    fingerprint: str | None = None
    id: str | None = None
    license: str | None = None
    comments: str | None = None
    raw_text: str | None = None
    text: str | None = None
    raw_html: str | None = None
    language: str | None = None
    image: str | None = None
    pagetype: str | None = None
    filedate: str | None = None
    source: str | None = None
    source_hostname: str | None = None
    excerpt: str | None = None
    categories: str | None = None
    tags: str | None = None


def clean_base_url(url: str) -> str:
    """Remove query parameters and anchors."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def normalize_image_url(url: str, base_url: str) -> str:
    """Convert relative path to absolute URL."""
    if not url or url.startswith(("http://", "https://", "data:")):
        return url

    # Handle protocol-relative URLs like //cdn.example.com/img.jpg
    if url.startswith("//"):
        return f"https:{url}"

    # Use base_url to complete relative paths
    return urljoin(clean_base_url(base_url), url)


AD_PATTERNS = [
    "ad-",
    "ads-",
    "advert-",
    "sponsor-",
    "google-",
    "doubleclick",
    "taboola",
]


def remove_empty_tags(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove elements containing only whitespace."""
    for tag in soup.find_all():
        if not tag.get_text(strip=True) and not tag.find("img"):
            tag.decompose()
    return soup


def remove_ads(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove advertisement elements from HTML."""
    for tag in soup.find_all(class_=re.compile("|".join(AD_PATTERNS))):
        tag.decompose()
    return soup


def remove_framework_noise(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove framework-specific attributes from HTML elements."""
    for tag in soup.find_all(attrs={"data-v": True}):
        del tag["data-v"]
    return soup


def preprocess_lazy_images(html: str, base_url: str) -> str:
    """
    Preprocess lazy-loaded images in HTML.

    Args:
        html: Raw HTML string
        base_url: Full URL of the article page, used to complete relative paths
    """
    if not base_url:
        raise ValueError("base_url cannot be empty, used for processing relative paths")

    soup = BeautifulSoup(html, "html.parser")

    soup = remove_ads(soup)
    soup = remove_empty_tags(soup)
    soup = remove_framework_noise(soup)
    # Process image links
    for img in soup.find_all("img"):
        # Check if there is a valid src (avoid overwriting existing src)
        current_src = img.get("src")
        if (
            current_src
            and isinstance(current_src, str)
            and not current_src.startswith("data:")
        ):
            img["src"] = normalize_image_url(current_src, base_url)
            continue

        # Find various lazy-loading attributes (in priority order)
        lazy_attrs = [
            "data-src",
            "data-original",
            "data-lazy-src",
            "data-srcset",
            "data-lazy-srcset",
        ]

        for attr in lazy_attrs:
            attr_value = img.get(attr)
            if attr_value:
                # Special handling for data-srcset/srcset (responsive images)
                if "srcset" in attr:
                    if isinstance(attr_value, str):
                        img["srcset"] = normalize_image_url(attr_value, base_url)
                else:
                    if isinstance(attr_value, str):
                        img["src"] = normalize_image_url(attr_value, base_url)
                break

    # Delete <script> tags but keep <script type="application/ld+json">
    for script in soup.find_all("script"):
        if script.get("type") != "application/ld+json":
            script.decompose()

    # Extract key CSS (image, table related) then delete <style>
    for style in soup.find_all("style"):
        style.decompose()

    # Delete comments and useless attributes
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Find all elements with aria-hidden="true"
    for tag in soup.find_all(attrs={"aria-hidden": "true"}):
        # Check if there are important child elements (e.g., <img>)
        # If image is in aria-hidden container but data-src is elsewhere, may need to keep
        img_tag = tag.find("img")
        if img_tag:
            # Move image out of container instead of deleting container
            tag.insert_before(img_tag)
        # Delete container
        tag.decompose()

    return str(soup)


def extract_content_from_html(html: str, original_url: str) -> HtmlContent | None:
    """
    Extract content from HTML.

    Args:
        html: HTML content to process
        original_url: Original URL of the page

    Returns:
        Extracted content as HtmlContent, or None if extraction fails
    """
    if not html or not html.strip():
        logger.warning("Empty HTML content provided")
        return None

    try:
        # Handle lazy-loaded images
        html = preprocess_lazy_images(html, base_url=original_url)

        result = extract(
            html,
            url=original_url,
            output_format="json",
            with_metadata=True,
            # Extended elements
            include_links=True,
            include_images=True,
            # Preserve tables
            include_tables=True,
            # Preserve bold, italic, and other formatting
            include_formatting=True,
            # Usually don't need comment sections
            include_comments=True,
            config=config,
        )
        if not result or len(result) == 0:
            logger.warning("No content extracted")
            return None

        json_data = json.loads(result)

        return HtmlContent(**json_data, raw_html=html)
    except Exception as e:
        logger.error("Content extraction failed", error=str(e))
        return None
