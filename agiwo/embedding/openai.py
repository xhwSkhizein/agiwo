"""
OpenAI and OpenAI-compatible embedding models.
"""

import os
from dataclasses import dataclass, field

import httpx

from agiwo.embedding.base import EmbeddingError, EmbeddingModel
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OpenAIEmbedding(EmbeddingModel):
    """
    OpenAI embedding model.

    Supports official OpenAI API and OpenAI-compatible APIs (e.g., SiliconFlow, vLLM).

    Environment variables:
        OPENAI_API_KEY: API key (fallback if api_key not provided)
        OPENAI_BASE_URL: Base URL (fallback if base_url not provided)

    For OpenAI-compatible APIs, set base_url to the provider's endpoint.
    """

    id: str = "text-embedding-3-small"
    name: str = "OpenAI Embedding"
    dimensions: int = 1536
    provider: str = "openai"
    timeout: float = 60.0
    max_batch_size: int = 100
    _client: httpx.AsyncClient | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY", "")
        if self.base_url is None:
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.base_url = self.base_url.rstrip("/")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI API."""
        if not texts:
            return []

        if not self.api_key:
            raise EmbeddingError(
                "OpenAI API key required. Set OPENAI_API_KEY or provide api_key."
            )

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]
            embeddings = await self._embed_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch."""
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "model": self.id,
            "input": texts,
        }
        if self.dimensions:
            payload["dimensions"] = self.dimensions

        client = self._get_client()
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]

        except httpx.HTTPStatusError as e:
            logger.error(
                "openai_embedding_error",
                status_code=e.response.status_code,
                detail=e.response.text[:500],
            )
            raise EmbeddingError(f"OpenAI API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            logger.error("openai_embedding_request_error", error=str(e))
            raise EmbeddingError(f"Request error: {e}") from e

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


__all__ = ["OpenAIEmbedding"]
