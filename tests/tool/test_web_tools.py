from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from agiwo.tool.context import ToolContext
from agiwo.llm.base import StreamChunk
from agiwo.tool.builtin.config import WebReaderApiConfig
from agiwo.tool.builtin.html_extract import HtmlContent
from agiwo.tool.builtin.web_reader.web_reader_tool import WebReaderTool
from agiwo.tool.builtin.web_search.web_search_tool import WebSearchTool
from agiwo.tool.storage.citation import CitationStoreConfig
from tests.utils.agent_context import build_tool_context


def _make_context(session_id: str) -> ToolContext:
    return build_tool_context(
        session_id=session_id,
        run_id=f"run-{session_id}",
        agent_id="agent-1",
        agent_name="agent-1",
    )


class StubToolModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    async def arun_stream(self, messages: list[dict], tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        for response in self.responses:
            yield StreamChunk(content=response)


def test_web_tool_schemas_and_descriptions_match_runtime_support() -> None:
    with patch(
        "agiwo.tool.builtin.web_search.web_search_tool.settings"
    ) as mock_settings:
        mock_settings.web_search_serper_api_key = None
        mock_settings.web_search_api_timeout = 10
        mock_settings.web_search_api_max_results = 5
        mock_settings.web_search_api_recency_days = 0
        mock_settings.web_search_api_base_url = "https://google.serper.dev"
        mock_settings.web_search_api_max_retries = 2
        search_tool = WebSearchTool(
            citation_store_config=CitationStoreConfig(
                storage_type="memory",
            ),
        )
    reader_tool = WebReaderTool(
        config=WebReaderApiConfig(),
        citation_store_config=CitationStoreConfig(
            storage_type="memory",
        ),
    )

    assert search_tool.get_parameters()["properties"]["query"]["type"] == "string"
    reader_schema = reader_tool.get_parameters()
    assert reader_schema["properties"]["url"]["type"] == "string"
    assert "oneOf" not in reader_schema
    assert reader_schema["required"] == []
    assert "web_fetch" not in search_tool.description
    assert "web_reader" in search_tool.description


def test_web_reader_openai_schema_stays_provider_compatible() -> None:
    reader_tool = WebReaderTool(
        citation_store_config=CitationStoreConfig(
            storage_type="memory",
        ),
    )

    schema = reader_tool.to_openai_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "web_reader"
    assert "oneOf" not in schema["function"]["parameters"]
    assert schema["function"]["parameters"]["required"] == []
    assert set(schema["function"]["parameters"]["properties"]) == {
        "index",
        "url",
        "search_query",
        "summarize",
    }


@pytest.mark.asyncio
async def test_web_search_without_api_key_returns_runtime_error() -> None:
    with patch(
        "agiwo.tool.builtin.web_search.web_search_tool.settings"
    ) as mock_settings:
        mock_settings.web_search_serper_api_key = None
        mock_settings.web_search_api_timeout = 10
        mock_settings.web_search_api_max_results = 5
        mock_settings.web_search_api_recency_days = 0
        mock_settings.web_search_api_base_url = "https://google.serper.dev"
        mock_settings.web_search_api_max_retries = 2
        tool = WebSearchTool(
            citation_store_config=CitationStoreConfig(
                storage_type="memory",
            ),
        )

    result = await tool.execute({"query": "latest agiwo"}, _make_context("no-key"))

    assert result.is_success is False
    assert result.error is not None
    assert "SERPER_API_KEY" in result.error


@pytest.mark.asyncio
async def test_web_reader_summarize_uses_internal_model() -> None:
    tool = WebReaderTool(
        citation_store_config=CitationStoreConfig(
            storage_type="memory",
        )
    )
    stub_model = StubToolModel(["summary result"])
    tool._llm_model = stub_model
    tool._curl_cffi_client.fetch = AsyncMock(
        return_value=HtmlContent(title="Example", text="Long content body")
    )

    result = await tool.execute(
        {"url": "https://example.com", "summarize": True},
        _make_context("reader-summary"),
    )

    assert result.is_success is True
    assert "summary result" in result.content
    assert len(stub_model.calls) == 1
    assert stub_model.calls[0]["tools"] is None


@pytest.mark.asyncio
async def test_web_reader_search_query_uses_internal_model() -> None:
    tool = WebReaderTool(
        citation_store_config=CitationStoreConfig(
            storage_type="memory",
        )
    )
    stub_model = StubToolModel(["filtered result"])
    tool._llm_model = stub_model
    tool._curl_cffi_client.fetch = AsyncMock(
        return_value=HtmlContent(title="Example", text="Long content body")
    )

    result = await tool.execute(
        {"url": "https://example.com", "search_query": "filtered"},
        _make_context("reader-query"),
    )

    assert result.is_success is True
    assert "filtered result" in result.content
    assert len(stub_model.calls) == 1
    assert "user's query is" in stub_model.calls[0]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_web_reader_model_init_failure_falls_back_to_original_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    tool = WebReaderTool(
        config=WebReaderApiConfig(
            model_provider="openai",
            model_name="gpt-4o-mini",
            model_base_url=None,
        ),
        citation_store_config=CitationStoreConfig(
            storage_type="memory",
        ),
    )
    tool._curl_cffi_client.fetch = AsyncMock(
        return_value=HtmlContent(title="Example", text="Original content body")
    )

    result = await tool.execute(
        {"url": "https://example.com", "summarize": True},
        _make_context("reader-no-key"),
    )

    assert result.is_success is True
    assert "Original content body" in result.content


@pytest.mark.asyncio
async def test_web_reader_fallback_to_browser_cli_when_curl_cffi_fails() -> None:
    """Test that web_reader falls back to browser_cli when curl_cffi fails."""
    tool = WebReaderTool(
        citation_store_config=CitationStoreConfig(
            storage_type="memory",
        )
    )
    # Mock curl_cffi to fail (return None)
    tool._curl_cffi_client.fetch = AsyncMock(return_value=None)
    # Mock browser_cli adapter to succeed
    tool._browser_cli_adapter.fetch = AsyncMock(
        return_value=(
            HtmlContent(title="Browser CLI Title", text="Browser CLI content"),
            None,
        )
    )

    result = await tool.execute(
        {"url": "https://example.com"},
        _make_context("reader-fallback"),
    )

    assert result.is_success is True
    assert "Browser CLI content" in result.content
    # Verify curl_cffi was tried first
    tool._curl_cffi_client.fetch.assert_awaited_once_with("https://example.com")
    # Verify browser_cli was called as fallback
    tool._browser_cli_adapter.fetch.assert_awaited_once_with("https://example.com")


@pytest.mark.asyncio
async def test_web_search_and_reader_share_citations_via_config() -> None:
    citation_config = CitationStoreConfig(
        storage_type="memory",
    )
    with patch(
        "agiwo.tool.builtin.web_search.web_search_tool.settings"
    ) as mock_settings:
        mock_settings.web_search_serper_api_key = SecretStr("test-key")
        mock_settings.web_search_api_timeout = 10
        mock_settings.web_search_api_max_results = 5
        mock_settings.web_search_api_recency_days = 0
        mock_settings.web_search_api_base_url = "https://google.serper.dev"
        mock_settings.web_search_api_max_retries = 2
        search_tool = WebSearchTool(
            citation_store_config=citation_config,
        )
    search_tool._http_client.post_json = AsyncMock(
        return_value={
            "organic": [
                {
                    "link": "https://example.com/article",
                    "title": "Example Article",
                    "snippet": "Example snippet",
                    "date": "2026-03-06",
                    "source": "example.com",
                }
            ]
        }
    )

    reader_tool = WebReaderTool(citation_store_config=citation_config)
    reader_tool._curl_cffi_client.fetch = AsyncMock(
        return_value=HtmlContent(title="Example Article", text="Full article body")
    )

    context = _make_context("shared-session")
    search_result = await search_tool.execute({"query": "example"}, context)
    reader_result = await reader_tool.execute({"index": 0}, context)

    assert search_result.is_success is True
    assert reader_result.is_success is True
    reader_tool._curl_cffi_client.fetch.assert_awaited_once_with(
        "https://example.com/article"
    )
