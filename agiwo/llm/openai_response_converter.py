import json

OpenAIMessage = dict[str, object]
OpenAITool = dict[str, object]
ResponsesInputItem = dict[str, object]


def split_system_instructions(
    messages: list[OpenAIMessage],
) -> tuple[str | None, list[OpenAIMessage]]:
    instruction_parts: list[str] = []
    remaining: list[OpenAIMessage] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            if content is None:
                continue
            if not isinstance(content, str):
                raise ValueError(
                    "Unsupported system content for openai-response provider"
                )
            instruction_parts.append(content)
            continue
        remaining.append(message)

    instructions = "\n\n".join(instruction_parts) or None
    return instructions, remaining


def _serialize_json_string(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True)


def _convert_user_message(message: OpenAIMessage) -> ResponsesInputItem:
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Unsupported user content for openai-response provider")
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": content}],
    }


def _convert_assistant_message(
    message: OpenAIMessage,
    *,
    message_index: int,
) -> list[ResponsesInputItem]:
    items: list[ResponsesInputItem] = []
    tool_calls = message.get("tool_calls") or []

    for tool_index, tool_call in enumerate(tool_calls):
        if tool_call.get("type") != "function":
            raise ValueError(
                "Unsupported assistant tool call type for openai-response provider"
            )
        function_payload = tool_call.get("function")
        if not isinstance(function_payload, dict):
            raise ValueError(
                "Unsupported assistant tool call payload for openai-response provider"
            )

        tool_call_id = (
            tool_call.get("id") or f"assistant_tool_call_{message_index}_{tool_index}"
        )
        tool_name = function_payload.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("Assistant tool call name missing for openai-response")

        items.append(
            {
                "type": "function_call",
                "call_id": tool_call_id,
                "name": tool_name,
                "arguments": _serialize_json_string(
                    function_payload.get("arguments", "")
                ),
            }
        )

    content = message.get("content")
    if content is not None:
        if not isinstance(content, str):
            raise ValueError("Unsupported assistant content for openai-response")
        items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                    }
                ],
            }
        )

    return items


def _convert_tool_message(message: OpenAIMessage) -> ResponsesInputItem:
    tool_call_id = message.get("tool_call_id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise ValueError("Tool message missing tool_call_id for openai-response")
    return {
        "type": "function_call_output",
        "call_id": tool_call_id,
        "output": _serialize_json_string(message.get("content")),
    }


def convert_messages_to_responses_input(
    messages: list[OpenAIMessage],
) -> list[ResponsesInputItem]:
    items: list[ResponsesInputItem] = []

    for message_index, message in enumerate(messages):
        role = message.get("role")
        if role == "user":
            items.append(_convert_user_message(message))
            continue
        if role == "assistant":
            items.extend(
                _convert_assistant_message(message, message_index=message_index)
            )
            continue
        if role == "tool":
            items.append(_convert_tool_message(message))
            continue
        raise ValueError(
            f"Unsupported message role for openai-response provider: {role}"
        )

    return items


def convert_tools_to_responses_tools(
    tools: list[OpenAITool] | None,
) -> list[ResponsesInputItem] | None:
    if not tools:
        return None

    converted: list[ResponsesInputItem] = []
    for tool in tools:
        if tool.get("type") != "function":
            raise ValueError("Unsupported tool type for openai-response provider")
        function_payload = tool.get("function")
        if not isinstance(function_payload, dict):
            raise ValueError("Unsupported tool payload for openai-response provider")
        converted.append(
            {
                "type": "function",
                "name": function_payload["name"],
                "description": function_payload.get("description", ""),
                "parameters": function_payload.get("parameters", {}),
            }
        )

    return converted


__all__ = [
    "convert_messages_to_responses_input",
    "convert_tools_to_responses_tools",
    "split_system_instructions",
]
