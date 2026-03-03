"""Tests for embedding module."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agiwo.embedding import (
    EmbeddingError,
    EmbeddingFactory,
    EmbeddingModel,
    LocalEmbedding,
    OpenAIEmbedding,
)


class TestOpenAIEmbedding:
    """Tests for OpenAIEmbedding."""

    def test_init_with_env_vars(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://test.api.com/v1",
        }):
            model = OpenAIEmbedding()
            assert model.api_key == "test-key"
            assert model.base_url == "https://test.api.com/v1"

    def test_init_with_explicit_params(self):
        model = OpenAIEmbedding(
            id="text-embedding-ada-002",
            api_key="explicit-key",
            base_url="https://custom.api.com/v1",
            dimensions=768,
        )
        assert model.id == "text-embedding-ada-002"
        assert model.api_key == "explicit-key"
        assert model.base_url == "https://custom.api.com/v1"
        assert model.dimensions == 768

    def test_model_id_property(self):
        model = OpenAIEmbedding(id="test-model", api_key="key")
        assert model.model_id == "openai:test-model"

    @pytest.mark.asyncio
    async def test_embed_empty_list(self):
        model = OpenAIEmbedding(api_key="key")
        result = await model.embed([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            model = OpenAIEmbedding(api_key="")
            with pytest.raises(EmbeddingError, match="API key required"):
                await model.embed(["test"])

    @pytest.mark.asyncio
    async def test_embed_single(self):
        model = OpenAIEmbedding(api_key="key")
        mock_response = {
            "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]
        }
        
        with patch.object(model, "_embed_batch", new_callable=AsyncMock) as mock:
            mock.return_value = [[0.1, 0.2, 0.3]]
            result = await model.embed_single("test")
            assert result == [0.1, 0.2, 0.3]


class TestLocalEmbedding:
    """Tests for LocalEmbedding."""

    def test_init_with_env_var(self):
        with patch.dict(os.environ, {
            "AGIWO_LOCAL_EMBEDDING_MODEL_PATH": "/path/to/model.gguf"
        }):
            model = LocalEmbedding()
            assert model.model_path == "/path/to/model.gguf"

    def test_init_with_explicit_path(self):
        model = LocalEmbedding(model_path="/custom/path.gguf")
        assert model.model_path == "/custom/path.gguf"

    def test_model_id_property(self):
        model = LocalEmbedding()
        assert model.model_id == "local:local"

    @pytest.mark.asyncio
    async def test_embed_empty_list(self):
        model = LocalEmbedding(model_path="/path/to/model.gguf")
        result = await model.embed([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_requires_model_path(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AGIWO_LOCAL_EMBEDDING_MODEL_PATH", None)
            model = LocalEmbedding(model_path="")
            with pytest.raises(EmbeddingError, match="model path required"):
                await model.embed(["test"])

    @pytest.mark.asyncio
    async def test_embed_model_not_found(self):
        model = LocalEmbedding(model_path="/nonexistent/model.gguf")
        with pytest.raises(EmbeddingError, match="not found"):
            await model.embed(["test"])


class TestEmbeddingFactory:
    """Tests for EmbeddingFactory."""

    def test_create_disabled(self):
        result = EmbeddingFactory.create(provider="disabled")
        assert result is None

    def test_create_openai(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = EmbeddingFactory.create(provider="openai")
            assert isinstance(result, OpenAIEmbedding)
            assert result.base_url == "https://api.openai.com/v1"

    def test_create_openai_requires_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("AGIWO_EMBEDDING_API_KEY", None)
            with pytest.raises(EmbeddingError, match="API key required"):
                EmbeddingFactory.create(provider="openai")

    def test_create_openai_like(self):
        result = EmbeddingFactory.create(
            provider="openai-like",
            api_key="custom-key",
            base_url="https://custom.api.com/v1",
            model="custom-model",
        )
        assert isinstance(result, OpenAIEmbedding)
        assert result.api_key == "custom-key"
        assert result.base_url == "https://custom.api.com/v1"
        assert result.id == "custom-model"
        assert result.provider == "openai-like"

    def test_create_openai_like_requires_base_url(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_BASE_URL", None)
            os.environ.pop("AGIWO_EMBEDDING_BASE_URL", None)
            with pytest.raises(EmbeddingError, match="Base URL required"):
                EmbeddingFactory.create(
                    provider="openai-like",
                    api_key="key",
                )

    def test_create_local(self):
        result = EmbeddingFactory.create(
            provider="local",
            model_path="/path/to/model.gguf",
        )
        assert isinstance(result, LocalEmbedding)
        assert result.model_path == "/path/to/model.gguf"

    def test_create_local_requires_path(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AGIWO_LOCAL_EMBEDDING_MODEL_PATH", None)
            with pytest.raises(EmbeddingError, match="model path required"):
                EmbeddingFactory.create(provider="local")

    def test_create_auto_with_openai_key(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
        }, clear=True):
            os.environ.pop("AGIWO_LOCAL_EMBEDDING_MODEL_PATH", None)
            result = EmbeddingFactory.create(provider="auto")
            assert isinstance(result, OpenAIEmbedding)

    def test_create_auto_no_provider(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("AGIWO_EMBEDDING_API_KEY", None)
            os.environ.pop("AGIWO_LOCAL_EMBEDDING_MODEL_PATH", None)
            result = EmbeddingFactory.create(provider="auto")
            assert result is None

    def test_create_with_custom_dimensions(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "key"}):
            result = EmbeddingFactory.create(
                provider="openai",
                dimensions=768,
            )
            assert result.dimensions == 768

    def test_create_unknown_provider(self):
        result = EmbeddingFactory.create(provider="unknown")
        assert result is None
