"""
Local embedding models using llama-cpp-python.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from agiwo.config.settings import load_settings
from agiwo.embedding.base import EmbeddingError, EmbeddingModel
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class LlamaEmbeddingClient(Protocol):
    def embed(self, text: str) -> list[float] | list[list[float]]: ...

    def n_embd(self) -> int: ...


@dataclass
class LocalEmbedding(EmbeddingModel):
    """
    Local embedding model using llama-cpp-python.

    Supports GGUF format models like:
    - embeddinggemma-300M-Q8_0.gguf
    - nomic-embed-text-v1.5.Q8_0.gguf
    - bge-small-en-v1.5-q8_0.gguf

    Environment variables:
        AGIWO_LOCAL_EMBEDDING_MODEL_PATH: Path to the GGUF model file

    The model file should be downloaded and placed in a local directory.
    """

    id: str = "local"
    name: str = "Local Embedding"
    dimensions: int = 768
    provider: str = "local"
    model_path: str | None = None
    n_ctx: int = 2048
    n_batch: int = 512
    n_threads: int | None = None
    _llama: LlamaEmbeddingClient | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.model_path is None:
            runtime_settings = load_settings()
            self.model_path = runtime_settings.local_embedding_model_path or ""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using local GGUF model."""
        if not texts:
            return []

        llama = self._get_llama()
        embeddings: list[list[float]] = []

        for text in texts:
            try:
                embedding = llama.embed(text)
                if isinstance(embedding[0], list):
                    embeddings.append(embedding[0])
                else:
                    embeddings.append(embedding)
            except Exception as e:
                logger.error("local_embedding_error", text_len=len(text), error=str(e))
                raise EmbeddingError(f"Local embedding failed: {e}") from e

        return embeddings

    def _get_llama(self) -> LlamaEmbeddingClient:
        """Get or create Llama instance."""
        if self._llama is not None:
            return self._llama

        if not self.model_path:
            raise EmbeddingError(
                "Local model path required. Set AGIWO_LOCAL_EMBEDDING_MODEL_PATH "
                "or provide model_path."
            )

        model_path = Path(self.model_path).expanduser()
        if not model_path.exists():
            raise EmbeddingError(f"Model file not found: {model_path}")

        if Llama is None:
            raise EmbeddingError(
                "llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python"
            )

        logger.info("loading_local_embedding_model", path=str(model_path))

        self._llama = Llama(
            model_path=str(model_path),
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            n_threads=self.n_threads,
            embedding=True,
            verbose=False,
        )

        if hasattr(self._llama, "n_embd"):
            self.dimensions = self._llama.n_embd()

        logger.info(
            "local_embedding_model_loaded",
            path=str(model_path),
            dimensions=self.dimensions,
        )

        return self._llama

    async def close(self) -> None:
        """Release model resources."""
        if self._llama is not None:
            del self._llama
            self._llama = None


__all__ = ["LocalEmbedding"]
