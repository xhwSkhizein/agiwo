"""
Web fetch utility functions.

Provides helper functions for web content processing.
"""


def truncate_middle(text: str, max_length: int, ellipsis: str = "...") -> str:
    """
    Truncate text from the middle, preserving beginning and end.

    Useful for maintaining context while reducing length.
    """
    if len(text) <= max_length:
        return text

    # Ensure ellipsis fits
    if max_length < len(ellipsis):
        return ellipsis[:max_length]

    available_length = max_length - len(ellipsis)

    # Split available length between start and end
    front_length = (available_length + 1) // 2
    back_length = available_length // 2

    front_part = text[:front_length]
    back_part = text[-back_length:] if back_length > 0 else ""

    return front_part + ellipsis + back_part
