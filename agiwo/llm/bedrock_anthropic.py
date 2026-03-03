"""AWS Bedrock Anthropic model implementation."""

import json
import os
from typing import AsyncIterator

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    raise ImportError("Please install boto3: pip install boto3")

from agiwo.llm.base import Model, StreamChunk
from agiwo.llm.helper import normalize_usage_metrics
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
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        cache_hit_price: float = 0.0,
        input_price: float = 0.0,
        output_price: float = 0.0,
        aws_region: str | None = None,
        aws_profile: str | None = None,
    ):
        super().__init__(
            id=id,
            name=name,
            api_key=api_key,
            base_url=base_url,
            provider="bedrock-anthropic",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            cache_hit_price=cache_hit_price,
            input_price=input_price,
            output_price=output_price,
        )
        self.aws_region = aws_region or os.getenv("AWS_REGION", "us-east-1")
        self.aws_profile = aws_profile or os.getenv("AWS_PROFILE")
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

    def _convert_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Convert OpenAI format messages to Anthropic format for Bedrock."""
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                system_prompt = content
            elif role == "user":
                anthropic_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": content}]
                })
            elif role == "assistant":
                content_blocks = []
                if content:
                    content_blocks.append({"type": "text", "text": content})

                # Handle tool_calls if present
                if "tool_calls" in msg:
                    for tool_call in msg["tool_calls"]:
                        func = tool_call["function"]
                        args = func["arguments"]
                        if isinstance(args, str):
                            args = json.loads(args)

                        content_blocks.append({
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": func["name"],
                            "input": args,
                        })

                anthropic_messages.append({
                    "role": "assistant",
                    "content": content_blocks
                })
            elif role == "tool":
                # Tool result
                tool_result_content = content
                if not isinstance(tool_result_content, str):
                    tool_result_content = json.dumps(tool_result_content)

                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id"),
                        "content": tool_result_content,
                    }]
                })

        return system_prompt, anthropic_messages

    def _convert_tools(self, tools: list[dict] | None) -> list[dict] | None:
        """Convert OpenAI format tools to Anthropic format."""
        if not tools:
            return None

        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })

        return anthropic_tools if anthropic_tools else None

    async def arun_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Call AWS Bedrock Anthropic API and return streaming output."""
        system_prompt, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)

        # Build request body
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
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
            response = client.invoke_model_with_response_stream(
                modelId=self.id,
                body=json.dumps(body),
            )

            # Track accumulated state
            current_content = ""
            tool_calls_buffer = {}
            usage_info = {
                "input_tokens": 0,
                "output_tokens": 0,
            }

            for event in response["body"]:
                chunk = json.loads(event["chunk"]["bytes"].decode())
                stream_chunk = StreamChunk()

                # Handle message_start (contains usage)
                if chunk.get("type") == "message_start":
                    message = chunk.get("message", {})
                    usage = message.get("usage", {})
                    if usage:
                        usage_info["input_tokens"] = usage.get("input_tokens", 0)
                        usage_info["output_tokens"] = usage.get("output_tokens", 0)
                        stream_chunk.usage = normalize_usage_metrics(usage_info)

                # Handle content_block_start
                elif chunk.get("type") == "content_block_start":
                    index = chunk.get("index", 0)
                    content_block = chunk.get("content_block", {})

                    if content_block.get("type") == "tool_use":
                        tool_calls_buffer[index] = {
                            "id": content_block.get("id"),
                            "name": content_block.get("name"),
                            "input": "",
                        }

                # Handle content_block_delta
                elif chunk.get("type") == "content_block_delta":
                    delta = chunk.get("delta", {})
                    index = chunk.get("index", 0)

                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        stream_chunk.content = text
                        current_content += text

                    elif delta.get("type") == "input_json_delta":
                        partial_json = delta.get("partial_json", "")
                        if index in tool_calls_buffer:
                            tool_calls_buffer[index]["input"] += partial_json

                # Handle content_block_stop (tool call complete)
                elif chunk.get("type") == "content_block_stop":
                    index = chunk.get("index", 0)
                    if index in tool_calls_buffer:
                        tool_call = tool_calls_buffer[index]
                        try:
                            arguments = json.loads(tool_call["input"])
                        except json.JSONDecodeError:
                            arguments = tool_call["input"]

                        stream_chunk.tool_calls = [{
                            "index": index,
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": arguments,
                            },
                        }]

                # Handle message_delta (stop reason and usage)
                elif chunk.get("type") == "message_delta":
                    delta = chunk.get("delta", {})
                    usage = chunk.get("usage", {})

                    if usage:
                        usage_info["output_tokens"] = usage.get("output_tokens", usage_info["output_tokens"])
                        stream_chunk.usage = normalize_usage_metrics(usage_info)

                    stop_reason = delta.get("stop_reason")
                    if stop_reason:
                        if stop_reason == "tool_use":
                            stream_chunk.finish_reason = "tool_calls"
                        elif stop_reason == "max_tokens":
                            stream_chunk.finish_reason = "length"
                        else:
                            stream_chunk.finish_reason = "stop"

                # Handle message_stop
                elif chunk.get("type") == "message_stop":
                    if stream_chunk.finish_reason is None:
                        stream_chunk.finish_reason = "stop"

                # Yield if has content
                if (
                    stream_chunk.content is not None
                    or stream_chunk.tool_calls is not None
                    or stream_chunk.usage is not None
                    or stream_chunk.finish_reason is not None
                ):
                    yield stream_chunk

        except ClientError as e:
            logger.error(
                "bedrock_request_failed",
                model=self.id,
                error=str(e),
                error_code=e.response["Error"]["Code"] if hasattr(e, "response") else None,
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
