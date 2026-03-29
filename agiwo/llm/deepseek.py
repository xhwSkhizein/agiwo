import re
import uuid
from typing import Any, AsyncIterator
import json

from agiwo.config.settings import get_settings
from agiwo.llm.base import StreamChunk
from agiwo.llm.openai import OpenAIModel


# DSML function_calls pattern
DSML_FUNCTION_CALLS_PATTERN = re.compile(
    r"<｜DSML｜function_calls>\s*"
    r"(<｜DSML｜invoke[^>]*>.*?</｜DSML｜invoke>)\s*"
    r"</｜DSML｜function_calls>",
    re.DOTALL,
)

DSML_INVOKE_PATTERN = re.compile(
    r'<｜DSML｜invoke\s+name="([^"]+)">' r"(.*?)</｜DSML｜invoke>",
    re.DOTALL,
)

DSML_PARAMETER_PATTERN = re.compile(
    r'<｜DSML｜parameter\s+name="([^"]+)"\s+[^>]*>(.*?)</｜DSML｜parameter>', re.DOTALL
)


def parse_dsml_function_calls(content: str) -> list[dict] | None:
    """
    Parse DSML function_calls format from content.

    Example input:
        <｜DSML｜function_calls>
        <｜DSML｜invoke name="bash">
        <｜DSML｜parameter name="command" string="true">ls -la</｜DSML｜parameter>
        </｜DSML｜invoke>
        </｜DSML｜function_calls>

    Returns list of tool_calls in OpenAI format, or None if no DSML found.
    """
    if "<｜DSML｜function_calls>" not in content:
        return None

    match = DSML_FUNCTION_CALLS_PATTERN.search(content)
    if not match:
        return None

    tool_calls = []
    invoke_blocks = DSML_INVOKE_PATTERN.findall(match.group(0))

    for idx, (func_name, invoke_content) in enumerate(invoke_blocks):
        # Parse parameters
        arguments: dict[str, str] = {}
        for param_name, param_value in DSML_PARAMETER_PATTERN.findall(invoke_content):
            arguments[param_name] = param_value.strip()

        tool_call = {
            "index": idx,
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": json.dumps(arguments),
            },
        }
        tool_calls.append(tool_call)

    return tool_calls if tool_calls else None


class DeepseekModel(OpenAIModel):
    def __init__(
        self,
        id: str = "deepseek/deepseek-chat",
        name: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        **model_kwargs: Any,
    ):
        super().__init__(
            id=id,
            name=name,
            api_key=api_key,
            base_url=base_url,
            provider="deepseek",
            **model_kwargs,
        )

    def _resolve_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        _s = get_settings()
        if _s.deepseek_api_key:
            return _s.deepseek_api_key.get_secret_value()
        return None

    def _resolve_base_url(self) -> str | None:
        return (
            self.base_url
            or get_settings().deepseek_base_url
            or "https://api.deepseek.com"
        )

    def _preprocess_messages_for_thinking_mode(
        self, messages: list[dict]
    ) -> list[dict]:
        """
        Preprocess messages for DeepSeek thinking mode.

        According to DeepSeek API docs:
        - If the last message is a user message (new turn), remove reasoning_content from all history assistant messages
        - If using deepseek-reasoner model, ensure all assistant messages have reasoning_content field (set to None if missing)

        Args:
            messages: List of messages in OpenAI format

        Returns:
            Preprocessed messages ready for API call
        """
        if not messages:
            return messages

        # Check if this is a thinking mode model
        model_name = self.name
        is_thinking_mode = "reasoner" in model_name.lower()

        # Check if last message is a user message (new conversation turn)
        last_message = messages[-1]
        is_new_turn = last_message.get("role") == "user"

        # Create a copy to avoid modifying the original
        processed_messages = []

        for i, msg in enumerate(messages):
            msg_copy = msg.copy()

            # Only process assistant messages
            if msg_copy.get("role") == "assistant":
                if is_new_turn:
                    # New turn: remove reasoning_content from history messages
                    # (per DeepSeek docs: only include content, not reasoning_content in history)
                    if "reasoning_content" in msg_copy:
                        del msg_copy["reasoning_content"]
                else:
                    # Same turn: preserve reasoning_content if present
                    # If thinking mode and reasoning_content not present, set to None
                    if is_thinking_mode and "reasoning_content" not in msg_copy:
                        msg_copy["reasoning_content"] = None

            processed_messages.append(msg_copy)

        return processed_messages

    async def arun_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Call DeepSeek API with thinking mode support.

        Preprocesses messages according to DeepSeek thinking mode requirements:
        - If last message is user (new turn), removes reasoning_content from history
        - If using deepseek-reasoner, ensures all assistant messages have reasoning_content field

        Also handles DSML function_calls format from deepseek-reasoner model,
        converting it to standard tool_calls when necessary.

        Args:
            messages: OpenAI format message list
            tools: Tool definition list, OpenAI format

        Yields:
            StreamChunk: Standardized streaming output chunk
        """
        # Preprocess messages for thinking mode
        processed_messages = self._preprocess_messages_for_thinking_mode(messages)

        # Track accumulated content and whether we have tool_calls
        accumulated_content: list[str] = []
        has_tool_calls = False

        # Call parent's arun_stream with processed messages
        async for chunk in super().arun_stream(processed_messages, tools=tools):
            if chunk.content:
                accumulated_content.append(chunk.content)
            if chunk.tool_calls:
                has_tool_calls = True

            yield chunk

        # Post-process: check if content contains DSML function_calls but no tool_calls
        if tools and not has_tool_calls and accumulated_content:
            full_content = "".join(accumulated_content)
            dsml_tool_calls = parse_dsml_function_calls(full_content)
            if dsml_tool_calls:
                # Yield an additional chunk with parsed tool_calls
                yield StreamChunk(tool_calls=dsml_tool_calls)


__all__ = ["DeepseekModel"]
