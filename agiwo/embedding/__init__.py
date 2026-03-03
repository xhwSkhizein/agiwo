"""
Embedding models for text vectorization.

Supports multiple providers:
- OpenAI: Official OpenAI embedding API
- OpenAI-like: OpenAI-compatible APIs (SiliconFlow, vLLM, Ollama, etc.)
- Local: Local GGUF models via llama-cpp-python

Configuration via environment variables:
    AGIWO_EMBEDDING_PROVIDER: openai | openai-like | local | auto | disabled
    AGIWO_EMBEDDING_MODEL: Model ID (e.g., text-embedding-3-small)
    AGIWO_EMBEDDING_DIMENSIONS: Embedding dimensions (default: 1536)
    AGIWO_EMBEDDING_API_KEY: API key for openai/openai-like
    AGIWO_EMBEDDING_BASE_URL: Base URL for openai-like providers
    AGIWO_LOCAL_EMBEDDING_MODEL_PATH: Path to local GGUF model file
"""

from agiwo.embedding.base import EmbeddingError, EmbeddingModel
from agiwo.embedding.factory import EmbeddingFactory, EmbeddingProvider
from agiwo.embedding.local import LocalEmbedding
from agiwo.embedding.openai import OpenAIEmbedding

__all__ = [
    "EmbeddingModel",
    "EmbeddingError",
    "EmbeddingFactory",
    "EmbeddingProvider",
    "OpenAIEmbedding",
    "LocalEmbedding",
]
