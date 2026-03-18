"""Shared text-encoding fallback for token estimation and chunking."""

from functools import lru_cache

import tiktoken

from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class _FallbackEncoding:
    """Deterministic offline fallback when tiktoken resources are unavailable."""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]


@lru_cache(maxsize=128)
def resolve_text_encoding(
    model_name: str = "",
    default_encoding: str = "cl100k_base",
):
    """Resolve a text encoding, falling back to a local deterministic encoder."""
    if model_name:
        try:
            return tiktoken.encoding_for_model(model_name)
        except Exception:  # noqa: BLE001 - tokenizer fallback boundary
            pass

    try:
        return tiktoken.get_encoding(default_encoding)
    except Exception as error:  # noqa: BLE001 - tokenizer fallback boundary
        logger.warning(
            "tiktoken_encoding_unavailable",
            model_name=model_name,
            default_encoding=default_encoding,
            error=str(error),
        )
        return _FallbackEncoding()
