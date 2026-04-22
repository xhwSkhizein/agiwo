from agiwo.llm.openai_response_converter import convert_messages_to_responses_input


def test_convert_messages_to_responses_input_falls_back_for_non_string_tool_call_id() -> (
    None
):
    items = convert_messages_to_responses_input(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": 123,
                        "type": "function",
                        "function": {
                            "name": "weather_lookup",
                            "arguments": '{"city":"Paris"}',
                        },
                    }
                ],
            }
        ]
    )

    assert items[0]["call_id"] == "assistant_tool_call_0_0"
