# OpenAI Responses Provider Design

- Date: 2026-04-15
- Status: Proposed
- Scope: SDK LLM provider layer, model factory, model docs, provider tests

## Context

The current SDK LLM boundary is streaming-first and intentionally narrow:

- `agiwo.llm.base.Model` exposes `arun_stream(messages, tools) -> AsyncIterator[StreamChunk]`
- agent runtime consumes only normalized `StreamChunk` fields
- `OpenAIModel` currently targets `chat.completions`

We want to support OpenAI's Responses API without changing the public `Model` contract and without changing agent/runtime message ledger semantics.

The requested scope is intentionally narrow:

- add a new provider: `openai-response`
- keep `openai` and `openai-compatible` on `chat.completions`
- keep the upper-layer behavior aligned with other providers by continuing to emit `StreamChunk`
- first version supports text streaming and function calling
- multi-turn behavior remains stateless from the SDK point of view
- do not add `previous_response_id`
- do not add encrypted reasoning preservation

## Goals

- Add `provider="openai-response"` as a first-class SDK provider.
- Keep the public `Model` contract unchanged.
- Preserve compatibility with the current agent/tool execution path.
- Support streamed text output.
- Support streamed function-call deltas and tool-result follow-up turns.
- Keep implementation localized to the OpenAI Responses provider boundary.

## Non-Goals

- Changing `agiwo.llm.base.Model` or `StreamChunk`.
- Changing agent runtime ledger or message storage format.
- Migrating `openai` or `openai-compatible` away from `chat.completions`.
- Introducing `previous_response_id`, `conversation`, or SDK-managed Responses state.
- Preserving encrypted reasoning items across turns.
- Adding built-in Responses tools such as web search, file search, or code interpreter.
- Refactoring all OpenAI-family providers into a new shared framework.

## Options Considered

### Option A: Extend `OpenAIModel` with a Responses branch

Reuse `agiwo/llm/openai.py` and switch endpoint behavior based on provider.

Pros:

- Lowest immediate code volume.
- Reuses existing client/retry configuration directly.

Cons:

- Mixes two distinct OpenAI protocols in one provider implementation.
- Makes testing and debugging harder because chat-completions and Responses events have different semantics.
- Increases regression risk for existing `openai` and `openai-compatible` behavior.

### Option B: Add a dedicated `OpenAIResponsesModel`

Create a new provider implementation dedicated to Responses and keep the existing OpenAI provider untouched.

Pros:

- Keeps protocol-specific logic isolated.
- Matches the repository's preference for explicit provider adapters.
- Minimizes risk to existing `chat.completions` behavior.
- Keeps the upper layer unchanged by normalizing to `StreamChunk`.

Cons:

- Requires a private message/tool conversion layer.
- Duplicates a small amount of client setup and retry wiring unless lightly shared.

### Option C: Introduce a broader OpenAI-family abstraction first

Extract common OpenAI-family infrastructure and then implement both chat-completions and Responses on top.

Pros:

- Potentially cleaner long-term architecture.
- Avoids duplicated config/client logic.

Cons:

- Expands scope beyond the requested feature.
- Raises implementation and review cost.
- Risks unrelated regressions and refactoring drift.

## Decision

Choose Option B.

Add a dedicated `OpenAIResponsesModel` and a new provider name `openai-response`. The provider is responsible for translating the SDK's canonical OpenAI-style message ledger into OpenAI Responses request items, then translating Responses stream events back into normalized `StreamChunk` values.

The rest of the SDK continues to treat this provider exactly like any other `Model`.

## 1. Public Surface and Factory Wiring

### Provider enum

Add `openai-response` to `ModelProvider` in `agiwo/config/settings.py` and include it in `ALL_MODEL_PROVIDERS`.

### Factory registration

Register the provider in `agiwo/llm/factory.py`:

- provider name: `openai-response`
- model class: `OpenAIResponsesModel`
- env fallback behavior: same as `openai`
- provider override: not needed

### Public exports

Export `OpenAIResponsesModel` from `agiwo/llm/__init__.py` if public provider classes are currently exported.

### Configuration semantics

`openai-response` uses the same core config fields as `openai`:

- `api_key`
- `base_url`
- `temperature`
- `top_p`
- `max_output_tokens`
- `frequency_penalty`
- `presence_penalty`

Provider-specific semantics stay internal. The factory and config layer should not introduce new public model params for the first version.

## 2. Provider Structure

Add a new provider module:

- `agiwo/llm/openai_response.py`

Optional private helper:

- `agiwo/llm/openai_response_converter.py`

Responsibilities:

- `OpenAIResponsesModel`
  - resolve API key and base URL using the same OpenAI env behavior as `OpenAIModel`
  - construct `AsyncOpenAI`
  - build Responses request params
  - stream raw Responses events
  - normalize events into `StreamChunk`

- converter helper
  - convert SDK canonical `messages` to Responses `input`
  - convert OpenAI chat-style tool schemas to Responses function tools
  - centralize any provider-specific validation

The converter remains provider-private. The canonical SDK conversation format stays as the existing OpenAI-style message list.

## 3. Request Mapping

The provider receives:

- `messages: list[dict]`
- `tools: list[dict] | None`

It must convert them into a Responses request while keeping the call stateless.

### 3.1 Stateless multi-turn policy

For every request:

- convert the full current message ledger into Responses `input`
- do not use `previous_response_id`
- do not use `conversation`
- do not depend on server-side retained state

This preserves current SDK semantics, where the agent runtime owns conversation state.

### 3.2 System prompt handling

`system` messages should be lifted into `instructions`.

Rules:

- if there is one system message, use its content as `instructions`
- if there are multiple system messages in the assembled ledger, concatenate them with clear separators in original order
- do not also duplicate those system messages into `input`

This keeps the provider aligned with the Responses API shape while preserving the SDK's existing prompt assembly.

### 3.3 User messages

Canonical input:

```python
{"role": "user", "content": "..."}
```

Responses input item:

```python
{
    "type": "message",
    "role": "user",
    "content": [{"type": "input_text", "text": "..."}],
}
```

If the current ledger already contains content structures that the first version does not support, the provider should fail fast with a clear `ValueError`.

### 3.4 Assistant text messages

Historical assistant text must be preserved in stateless replay.

Convert assistant text messages into Responses-compatible assistant output-message items. The conversion layer should preserve:

- assistant role
- text content
- item ordering in the conversation

The implementation may use typed SDK params or raw dict payloads, but it must produce valid Responses input items that represent prior assistant outputs.

### 3.5 Assistant tool calls

Canonical assistant tool calls today look like chat-completions tool calls:

```python
{
    "role": "assistant",
    "tool_calls": [
        {
            "id": "...",
            "type": "function",
            "function": {"name": "...", "arguments": "..."},
        }
    ],
}
```

Convert each tool call into a Responses `function_call` item:

- `call_id` should use the canonical tool-call id
- `name` should use the function name
- `arguments` should remain the raw JSON string

This is required so follow-up tool outputs can reference the same call id.

### 3.6 Tool result messages

Canonical tool-result messages:

```python
{
    "role": "tool",
    "tool_call_id": "...",
    "content": "...",
}
```

Convert them into Responses `function_call_output` items:

- `call_id = tool_call_id`
- `output = content` as string

If `content` is not a string, serialize it consistently with existing tool-message behavior before sending it to the provider.

### 3.7 Tool schema conversion

Current tool schemas are OpenAI chat-style function tools:

```python
{
    "type": "function",
    "function": {
        "name": "...",
        "description": "...",
        "parameters": {...},
    },
}
```

Responses function tools should be emitted as:

```python
{
    "type": "function",
    "name": "...",
    "description": "...",
    "parameters": {...},
}
```

Only function tools are in scope for the first version. Any other tool type should be rejected at the provider boundary.

## 4. Stream Event Normalization

The provider must translate Responses stream events into `StreamChunk` so `agiwo.agent.llm_caller.stream_assistant_step()` continues to work unchanged.

### 4.1 Text

Map text deltas:

- `response.output_text.delta` -> `StreamChunk(content=delta)`

The provider should emit text chunks incrementally as they arrive.

### 4.2 Function-call deltas

Responses streams function-call arguments separately from text. The provider must accumulate these into chat-compatible tool-call deltas.

Provider-internal state:

- per-output-index function call buffer
- stored fields: `id/call_id`, `name`, partial `arguments`

Normalized output shape:

```python
{
    "index": <output_index>,
    "id": <call_id>,
    "type": "function",
    "function": {
        "name": <name>,
        "arguments": <partial_json>,
    },
}
```

This shape must stay compatible with the accumulator in `agiwo.agent.llm_caller._accumulate_tool_calls()`.

### 4.3 Finish reason

Map Responses completion semantics onto current `StreamChunk.finish_reason` values:

- normal completion -> `stop`
- function-call completion -> `tool_calls`
- max-token/incomplete truncation -> `length`

If the exact Responses terminal shape varies by event payload, normalize it in one provider-local helper rather than leaking protocol differences upward.

### 4.4 Usage

Provider usage should be normalized through the existing usage normalization path when possible.

The provider should emit usage once it becomes available, typically on completion. The normalized shape must remain:

- `input_tokens`
- `output_tokens`
- `total_tokens`
- `cache_read_tokens`
- `cache_creation_tokens`

### 4.5 Reasoning content

First-version policy:

- if Responses emits stable reasoning text or summary content that can be surfaced safely, map it to `StreamChunk.reasoning_content`
- if a reasoning event cannot be mapped cleanly, ignore it
- do not preserve or replay encrypted reasoning items across turns

This keeps the public contract unchanged while allowing best-effort reasoning visibility.

## 5. Error Handling

### Retry behavior

Reuse the current OpenAI retry policy:

- `APIConnectionError`
- `RateLimitError`
- `InternalServerError`
- `APITimeoutError`

### Validation failures

If the converter encounters unsupported message or tool shapes, raise `ValueError` before making an API call. Error text should identify the unsupported input category.

### Empty stream

If the Responses stream yields no usable chunks, raise a runtime error equivalent in intent to the current `OpenAIModel` empty-stream failure.

### Unsupported output items

Ignore irrelevant protocol events that do not affect text, tool calls, usage, or completion.

Fail fast when the response contains output-item shapes that are required for correctness but unsupported by this provider version.

## 6. Compatibility Rules

The following invariants must hold:

- `Model.arun_stream()` signature does not change
- `StreamChunk` shape does not change
- agent/tool execution path does not branch on provider type
- existing `openai` and `openai-compatible` behavior does not change
- provider-specific Responses state does not leak into `RunContext`, `StepRecord`, or persisted messages

## 7. Testing

Add a dedicated provider test module:

- `tests/llm/test_openai_response.py`

Required coverage:

### Provider tests

- text streaming returns incremental `StreamChunk.content`
- completion usage is normalized
- function-call arguments stream into chat-compatible `tool_calls`
- follow-up requests convert prior assistant tool calls and tool outputs correctly
- empty stream raises a clear error
- unsupported message/tool inputs fail at the provider boundary
- OpenAI API exceptions still propagate through the existing retry policy

### Factory tests

- `create_model()` builds `OpenAIResponsesModel` for `provider="openai-response"`
- config/env resolution matches `openai`

### Contract tests

At least one agent-level or llm-caller-level contract test should verify that `stream_assistant_step()` can consume `openai-response` tool-call chunks without any caller changes.

## 8. Documentation

Update:

- `docs/api/model.md`
- `docs/concepts/model.md`

Document:

- new provider name `openai-response`
- that it uses OpenAI Responses API under the hood
- that upper-layer semantics remain `StreamChunk`-based
- that first-version support covers text and function calling only
- that multi-turn handling is stateless and does not use `previous_response_id`

## 9. Rollout Sequence

Implement in this order:

1. Add provider enum and factory wiring.
2. Add `OpenAIResponsesModel` with client setup and retry behavior.
3. Add message/tool conversion helper.
4. Add Responses stream-event normalization to `StreamChunk`.
5. Add provider and factory tests.
6. Add one upper-layer contract test.
7. Update model docs.

## 10. Risks and Mitigations

### Risk: assistant-history replay shape is subtly wrong

Mitigation:

- isolate conversion logic in one helper
- test realistic message sequences: user -> assistant tool call -> tool output -> assistant final text

### Risk: Responses function-call events do not align with current tool-call accumulator expectations

Mitigation:

- normalize all provider-emitted tool-call deltas into the exact shape currently consumed by `agiwo.agent.llm_caller`
- add explicit tests for partial argument streaming

### Risk: unsupported content types appear in current ledgers later

Mitigation:

- fail fast at the provider boundary for unsupported shapes
- keep the first version intentionally narrow

### Risk: provider starts duplicating too much `OpenAIModel` logic

Mitigation:

- allow light extraction of shared OpenAI client/setup helpers if it clearly reduces duplication
- avoid broader refactors unrelated to Responses support

## Acceptance Criteria

- A caller can construct a model with `provider="openai-response"`.
- The model streams text output as normalized `StreamChunk.content`.
- The model streams function-call deltas as normalized `StreamChunk.tool_calls`.
- Existing agent runtime code can execute tools against this provider without provider-specific branches.
- Existing `openai` and `openai-compatible` providers remain unchanged.
