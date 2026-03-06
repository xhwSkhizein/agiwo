"""
Factory for creating embedding models based on configuration.
"""

from typing import Literal

from agiwo.config.settings import AgiwoSettings, load_settings
from agiwo.embedding.base import EmbeddingError, EmbeddingModel
from agiwo.embedding.local import LocalEmbedding
from agiwo.embedding.openai import OpenAIEmbedding
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

EmbeddingProvider = Literal["openai", "openai-like", "local", "auto", "disabled"]


class EmbeddingFactory:
    """
    Factory for creating embedding models.

    Supports three provider types:
    - openai: Official OpenAI API (text-embedding-3-small, etc.)
    - openai-like: OpenAI-compatible APIs (SiliconFlow, vLLM, Ollama, etc.)
    - local: Local GGUF models via llama-cpp-python

    Environment variables:
        AGIWO_EMBEDDING_PROVIDER: Provider type (openai/openai-like/local/auto/disabled)
        AGIWO_EMBEDDING_MODEL: Model ID (e.g., text-embedding-3-small)
        AGIWO_EMBEDDING_DIMENSIONS: Embedding dimensions (default: 1536)
        AGIWO_EMBEDDING_API_KEY: API key for openai/openai-like
        AGIWO_EMBEDDING_BASE_URL: Base URL for openai-like providers
        AGIWO_LOCAL_EMBEDDING_MODEL_PATH: Path to local GGUF model file
    """

    @staticmethod
    def create(
        provider: EmbeddingProvider | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model_path: str | None = None,
    ) -> EmbeddingModel | None:
        """
        Create an embedding model based on configuration.

        Args:
            provider: Provider type. If None, reads from AGIWO_EMBEDDING_PROVIDER.
            model: Model ID. If None, reads from AGIWO_EMBEDDING_MODEL.
            dimensions: Embedding dimensions. If None, reads from AGIWO_EMBEDDING_DIMENSIONS.
            api_key: API key for openai/openai-like. Falls back to env vars.
            base_url: Base URL for openai-like. Falls back to env vars.
            model_path: Path to local GGUF model. Falls back to env vars.

        Returns:
            EmbeddingModel instance, or None if disabled or unavailable.

        Raises:
            EmbeddingError: If configuration is invalid.
        """
        runtime_settings = load_settings()
        provider = provider or runtime_settings.embedding_provider
        provider = provider.lower()

        if provider == "disabled":
            logger.info("embedding_disabled")
            return None

        model = model or runtime_settings.embedding_model
        dimensions = dimensions or runtime_settings.embedding_dimensions

        if provider == "auto":
            return EmbeddingFactory._create_auto(
                model,
                dimensions,
                api_key,
                base_url,
                model_path,
                runtime_settings,
            )

        if provider == "openai":
            return EmbeddingFactory._create_openai(
                model,
                dimensions,
                api_key,
                runtime_settings,
            )

        if provider == "openai-like":
            return EmbeddingFactory._create_openai_like(
                model,
                dimensions,
                api_key,
                base_url,
                runtime_settings,
            )

        if provider == "local":
            return EmbeddingFactory._create_local(
                model_path,
                dimensions,
                runtime_settings,
            )

        logger.warning("unknown_embedding_provider", provider=provider)
        return None

    @staticmethod
    def _create_auto(
        model: str,
        dimensions: int,
        api_key: str | None,
        base_url: str | None,
        model_path: str | None,
        runtime_settings: AgiwoSettings,
    ) -> EmbeddingModel | None:
        """Auto-detect available provider."""
        local_path = model_path or runtime_settings.local_embedding_model_path or ""
        if local_path:
            try:
                return EmbeddingFactory._create_local(
                    local_path,
                    dimensions,
                    runtime_settings,
                )
            except EmbeddingError as e:
                logger.warning("auto_local_failed", error=str(e))

        openai_key = api_key or runtime_settings.get_embedding_api_key()
        if openai_key:
            custom_base = base_url or runtime_settings.embedding_base_url
            if custom_base:
                return EmbeddingFactory._create_openai_like(
                    model,
                    dimensions,
                    openai_key,
                    custom_base,
                    runtime_settings,
                )
            return EmbeddingFactory._create_openai(
                model,
                dimensions,
                openai_key,
                runtime_settings,
            )

        logger.info("embedding_auto_no_provider", msg="No embedding provider available")
        return None

    @staticmethod
    def _create_openai(
        model: str,
        dimensions: int,
        api_key: str | None,
        runtime_settings: AgiwoSettings,
    ) -> OpenAIEmbedding:
        """Create OpenAI embedding model."""
        key = api_key or runtime_settings.get_embedding_api_key()
        if not key:
            raise EmbeddingError(
                "OpenAI API key required. Set OPENAI_API_KEY or AGIWO_EMBEDDING_API_KEY."
            )

        return OpenAIEmbedding(
            id=model,
            name=f"OpenAI {model}",
            dimensions=dimensions,
            api_key=key,
            base_url="https://api.openai.com/v1",
        )

    @staticmethod
    def _create_openai_like(
        model: str,
        dimensions: int,
        api_key: str | None,
        base_url: str | None,
        runtime_settings: AgiwoSettings,
    ) -> OpenAIEmbedding:
        """Create OpenAI-compatible embedding model."""
        key = api_key or runtime_settings.get_embedding_api_key()
        url = base_url or runtime_settings.get_embedding_base_url()

        if not key:
            raise EmbeddingError(
                "API key required for openai-like provider. "
                "Set AGIWO_EMBEDDING_API_KEY or OPENAI_API_KEY."
            )
        if not url:
            raise EmbeddingError(
                "Base URL required for openai-like provider. "
                "Set AGIWO_EMBEDDING_BASE_URL or OPENAI_BASE_URL."
            )

        return OpenAIEmbedding(
            id=model,
            name=f"OpenAI-like {model}",
            dimensions=dimensions,
            api_key=key,
            base_url=url,
            provider="openai-like",
        )

    @staticmethod
    def _create_local(
        model_path: str | None,
        dimensions: int,
        runtime_settings: AgiwoSettings,
    ) -> LocalEmbedding:
        """Create local embedding model."""
        path = model_path or runtime_settings.local_embedding_model_path
        if not path:
            raise EmbeddingError(
                "Local model path required. Set AGIWO_LOCAL_EMBEDDING_MODEL_PATH."
            )

        return LocalEmbedding(
            id="local",
            name="Local Embedding",
            dimensions=dimensions,
            model_path=path,
        )


__all__ = ["EmbeddingFactory", "EmbeddingProvider"]
