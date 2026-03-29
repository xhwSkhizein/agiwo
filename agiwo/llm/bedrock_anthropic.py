"""AWS Bedrock Anthropic model implementation."""

import asyncio
import json
import queue as _queue
from typing import Any, AsyncIterator

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    raise ImportError("Please install boto3: pip install boto3") from None

from agiwo.llm.base import Model, ModelConfig, StreamChunk
from agiwo.config.settings import get_settings
from agiwo.llm.event_normalizer import (
    AnthropicStreamTranslator,
    normalize_bedrock_anthropic_event,
)
from agiwo.llm.message_converter import (
    convert_openai_messages_to_anthropic,
    convert_openai_tools_to_anthropic,
)
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class BedrockAnthropicModel(Model):
    """AWS Bedrock Anthropic Claude model implementation.

    Uses boto3 Bedrock Runtime client with invoke_model_with_response_stream.
    Supports any Claude model available on Bedrock.

    Required environment variables:
    - AWS_ACCESS_KEY_ID or AWS_PROFILE
    - AWS_SECRET_ACCESS_KEY (if using key-based auth)
    - AWS_REGION (default: us-east-1)

    Example model_id: "anthropic.claude-3-5-sonnet-20240620-v1:0"
    """

    def __init__(
        self,
        id: str,
        name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        aws_region: str | None = None,
        aws_profile: str | None = None,
        **model_kwargs: Any,
    ):
        config = ModelConfig(
            id=id,
            name=name,
            api_key=api_key,
            base_url=base_url,
            provider="bedrock-anthropic",
            **model_kwargs,
        )
        super().__init__(config)
        _s = get_settings()
        self.aws_region = aws_region or _s.aws_region
        self.aws_profile = aws_profile or _s.aws_profile
        self._client = None

    def _get_client(self):
        """Lazy initialization of Bedrock Runtime client."""
        if self._client is None:
            session_kwargs = {}
            if self.aws_profile:
                session_kwargs["profile_name"] = self.aws_profile

            session = boto3.Session(**session_kwargs)
            self._client = session.client(
                service_name="bedrock-runtime",
                region_name=self.aws_region,
            )
        return self._client

    async def arun_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Call AWS Bedrock Anthropic API and return streaming output."""
        system_prompt, anthropic_messages = convert_openai_messages_to_anthropic(
            messages,
            wrap_user_text=True,
            assistant_text_blocks=True,
            include_reasoning=False,
        )
        anthropic_tools = convert_openai_tools_to_anthropic(tools)

        # Build request body
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "messages": anthropic_messages,
        }

        if system_prompt:
            body["system"] = system_prompt

        if anthropic_tools:
            body["tools"] = anthropic_tools

        if self.top_p is not None:
            body["top_p"] = self.top_p

        logger.info(
            "bedrock_request",
            model=self.id,
            region=self.aws_region,
            messages_count=len(anthropic_messages),
            tools_count=len(anthropic_tools) if anthropic_tools else 0,
        )

        try:
            client = self._get_client()
            response = await asyncio.to_thread(
                client.invoke_model_with_response_stream,
                modelId=self.id,
                body=json.dumps(body),
            )

            translator = AnthropicStreamTranslator(include_reasoning=False)

            sync_queue: _queue.SimpleQueue[dict | None] = _queue.SimpleQueue()

            def _read_stream() -> None:
                try:
                    for event in response["body"]:
                        sync_queue.put(json.loads(event["chunk"]["bytes"].decode()))
                finally:
                    sync_queue.put(None)

            loop = asyncio.get_running_loop()
            reader_task = loop.run_in_executor(None, _read_stream)

            while True:
                chunk_data = await loop.run_in_executor(None, sync_queue.get)
                if chunk_data is None:
                    break
                stream_chunk = translator.process(
                    normalize_bedrock_anthropic_event(chunk_data)
                )
                if stream_chunk is not None:
                    yield stream_chunk

            await reader_task

        except ClientError as e:
            logger.error(
                "bedrock_request_failed",
                model=self.id,
                error=str(e),
                error_code=(
                    e.response["Error"]["Code"] if hasattr(e, "response") else None
                ),
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(
                "bedrock_request_failed",
                model=self.id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise


__all__ = ["BedrockAnthropicModel"]
