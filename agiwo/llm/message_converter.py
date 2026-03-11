"""OpenAI <-> Anthropic message/tool conversion helpers."""

import ast
import json
from typing import Any


def parse_json_tool_args(args: dict[str, Any] | str | None) -> dict[str, Any]:
    """Parse tool arguments from dict/JSON/Python-literal string formats."""
    if isinstance(args, dict):
        return args

    if not args or not isinstance(args, str):
        return {}

    try:
        return json.loads(args)
    except json.JSONDecodeError:
        pass

    try:
        if args.strip().startswith("{") and args.strip().endswith("}"):
            val = ast.literal_eval(args)
            if isinstance(val, dict):
                return val
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        pass

    return {"__raw_arguments__": args}


def convert_openai_tools_to_anthropic(
    tools: list[dict] | None,
) -> list[dict] | None:
    """Convert OpenAI-format tools to Anthropic tool definitions."""
    if not tools:
        return None

    anthropic_tools = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool["function"]
        anthropic_tools.append(
            {
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {}),
            }
        )
    return anthropic_tools if anthropic_tools else None


def _convert_user_message(
    content: str | list[dict[str, Any]] | None,
    wrap_user_text: bool,
) -> dict[str, Any]:
    if wrap_user_text:
        return {"role": "user", "content": [{"type": "text", "text": content}]}
    return {"role": "user", "content": content}


def _convert_assistant_message(
    msg: dict[str, Any],
    *,
    assistant_text_blocks: bool,
    include_reasoning: bool,
) -> dict[str, Any]:
    content = msg.get("content")
    reasoning = msg.get("reasoning_content") if include_reasoning else None
    has_tool_calls = "tool_calls" in msg

    if not (assistant_text_blocks or has_tool_calls or reasoning):
        return {"role": "assistant", "content": content}

    content_blocks: list[dict[str, Any]] = []
    if reasoning:
        content_blocks.append({"type": "thinking", "thinking": reasoning})
    if content:
        content_blocks.append({"type": "text", "text": content})

    if has_tool_calls:
        for tool_call in msg["tool_calls"]:
            func = tool_call["function"]
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": func["name"],
                    "input": parse_json_tool_args(func["arguments"]),
                }
            )

    return {"role": "assistant", "content": content_blocks}


def _convert_tool_message(msg: dict[str, Any]) -> dict[str, Any]:
    tool_result_content = msg.get("content")
    if not isinstance(tool_result_content, str):
        tool_result_content = json.dumps(tool_result_content)

    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id"),
                "content": tool_result_content,
            }
        ],
    }


def convert_openai_messages_to_anthropic(
    messages: list[dict],
    *,
    wrap_user_text: bool = False,
    assistant_text_blocks: bool = False,
    include_reasoning: bool = True,
) -> tuple[str | None, list[dict]]:
    """Convert OpenAI-format messages to Anthropic message payloads."""
    system_prompt = None
    anthropic_messages: list[dict] = []

    for msg in messages:
        role = msg.get("role")

        if role == "system":
            system_prompt = msg.get("content")
            continue

        if role == "user":
            anthropic_messages.append(
                _convert_user_message(msg.get("content"), wrap_user_text)
            )
            continue

        if role == "assistant":
            anthropic_messages.append(
                _convert_assistant_message(
                    msg,
                    assistant_text_blocks=assistant_text_blocks,
                    include_reasoning=include_reasoning,
                )
            )
            continue

        if role == "tool":
            anthropic_messages.append(_convert_tool_message(msg))

    return system_prompt, anthropic_messages


__all__ = [
    "parse_json_tool_args",
    "convert_openai_tools_to_anthropic",
    "convert_openai_messages_to_anthropic",
]
