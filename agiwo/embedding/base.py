"""
Base classes for embedding models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class EmbeddingError(Exception):
    """Raised when embedding operation fails."""

    pass


@dataclass
class EmbeddingModel(ABC):
    """
    Abstract base for all embedding model implementations.

    Subclasses must implement embed() to provide embedding vectors.
    """

    id: str
    name: str
    dimensions: int
    api_key: str | None = None
    base_url: str | None = None
    provider: str = ""

    @property
    def model_id(self) -> str:
        """Return the model identifier for caching."""
        return f"{self.provider}:{self.id}"

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts into vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each is a list of floats).

        Raises:
            EmbeddingError: If embedding fails.
        """

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text string."""
        results = await self.embed([text])
        return results[0]

    async def close(self) -> None:
        """Close any underlying connections."""
        pass


__all__ = ["EmbeddingModel", "EmbeddingError"]
