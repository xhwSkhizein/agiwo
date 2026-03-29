"""Citation storage module.

Citation storage implementations for web_search and web_reader tools.
"""

from .memory_store import InMemoryCitationStore
from .factory import CitationStoreConfig, create_citation_store
from .models import (
    CitationSourceRaw,
    CitationSourceSimplified,
    CitationSourceType,
)
from .protocols import (
    CitationSourceRepository,
)
from .sqlite_store import SQLiteCitationStore
from .utils import (
    generate_citation_id,
)

__all__ = [
    "CitationSourceRaw",
    "CitationSourceSimplified",
    "CitationSourceType",
    "CitationSourceRepository",
    "CitationStoreConfig",
    "create_citation_store",
    "generate_citation_id",
    "InMemoryCitationStore",
    "SQLiteCitationStore",
]
