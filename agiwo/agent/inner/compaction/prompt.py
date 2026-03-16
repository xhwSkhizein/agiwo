"""Internal compact prompt template."""

DEFAULT_COMPACT_PROMPT = """**IMPORTANT: Context Compression Required**

The conversation context is approaching the limit. Please provide a comprehensive summary of the conversation so far.

## Instructions

1. Analyze the entire conversation history above
2. Extract and preserve:
   - Key decisions made by the user or assistant
   - Important facts, data, or conclusions
   - File paths, URLs, or references mentioned
   - Tool calls and their significant results
   - User preferences or explicit requests to remember
   - Current task state and progress

3. Output a JSON object with the following structure:
```json
{{
  "summary": "A comprehensive summary of the conversation...",
  "key_decisions": ["decision 1", "decision 2"],
  "important_refs": ["file/path/1", "https://url.com"],
  "tool_calls_summary": [
    {{"name": "tool_name", "result_summary": "brief result"}}
  ],
  "user_preferences": ["preference 1"],
  "current_task_state": "Description of where we are in the task"
}}
```

## Previous Compact Summary (if any)

{previous_summary}

## Output

Respond ONLY with the JSON object, no additional text.
"""

DEFAULT_ASSISTANT_RESPONSE = "Understood. I have the context from the summary. Continuing."

__all__ = ["DEFAULT_COMPACT_PROMPT", "DEFAULT_ASSISTANT_RESPONSE"]
