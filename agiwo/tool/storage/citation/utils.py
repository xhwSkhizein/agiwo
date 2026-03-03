"""
Citation utilities - Helper functions for citation system.
"""

import secrets
from datetime import datetime


def generate_citation_id(prefix: str = "cite") -> str:
    """
    Generate unique citation ID.

    Args:
        prefix: ID prefix (e.g., "search", "fetch")

    Returns:
        Unique ID in format "{prefix}-{timestamp}-{random}"
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = secrets.token_hex(4)
    return f"{prefix}-{timestamp}-{random_suffix}"
