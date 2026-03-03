"""
Citation system data models.

Used for citation management in web_search and web_fetch tools,
reducing token usage.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CitationSourceType(str, Enum):
    """Citation source type."""

    SEARCH = "search"  # From web_search tool
    DIRECT_URL = "direct_url"  # Direct URL fetch


class CitationSourceRaw(BaseModel):
    """Complete citation source model (for internal storage)."""

    citation_id: str = Field(description="Unique citation ID")
    session_id: str = Field(description="Session ID")
    source_type: CitationSourceType = Field(description="Source type")

    # Basic information
    url: str = Field(description="Web page URL")
    title: str | None = Field(default=None, description="Title")
    snippet: str | None = Field(default=None, description="Snippet (search result)")
    date_published: str | None = Field(default=None, description="Publication date")
    source: str | None = Field(default=None, description="Source")

    # Content information
    full_content: str | None = Field(
        default=None, description="Full content (fetch result)"
    )
    processed_content: str | None = Field(
        default=None, description="Processed content for LLM"
    )
    original_content: dict[str, Any] | None = Field(
        default=None, description="Original content metadata"
    )

    # Relationships
    related_citation_id: str | None = Field(
        default=None, description="Related citation_id (fetch linked to search)"
    )
    related_index: int | None = Field(
        default=None, description="Related index (fetch linked to search index)"
    )

    # Metadata
    query: str | None = Field(default=None, description="Search query (search type)")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Processing parameters (search_query, summarize, etc.)",
    )
    index: int | None = Field(
        default=None,
        description="Numeric index (for search results, used in web_fetch(index=N))",
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )


class CitationSourceSimplified(BaseModel):
    """Simplified citation source model (for LLM, hides detailed information)."""

    citation_id: str = Field(description="Citation ID")
    source_type: CitationSourceType = Field(description="Source type")
    url: str = Field(description="Web page URL")
    index: int | None = Field(
        default=None,
        description="Numeric index (for search results, used in web_fetch(index=N))",
    )
    title: str | None = Field(default=None, description="Title")
    snippet: str | None = Field(default=None, description="Snippet")
    date_published: str | None = Field(default=None, description="Publication date")
    source: str | None = Field(default=None, description="Source")
    created_at: datetime | None = Field(default=None, description="Creation timestamp")
