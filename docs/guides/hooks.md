# Hooks

Hooks are registered through the phase-based `HookRegistry`. The old
single-slot `AgentHooks` dataclass is no longer part of the public API.

## Quick Start

```python
from agiwo.agent import (
    Agent,
    AgentConfig,
    HookPhase,
    HookRegistry,
    observe,
    transform,
)


async def add_prelude(payload: dict) -> dict:
    return {"prelude_text": "Please be concise."}


async def log_step(payload: dict) -> None:
    step = payload["step"]
    print(f"{step.role.value} step committed: {step.sequence}")


hooks = HookRegistry(
    [
        transform(HookPhase.PREPARE, "add_prelude", add_prelude),
        observe(HookPhase.AFTER_STEP_COMMIT, "log_step", log_step),
    ]
)

agent = Agent(
    AgentConfig(name="assistant", description="..."),
    model=model,
    hooks=hooks,
)
```

## Core Types

```python
from agiwo.agent import (
    HookCapability,
    HookGroup,
    HookPhase,
    HookRegistration,
    HookRegistry,
    decision_support,
    observe,
    transform,
)
```

- `HookPhase`: lifecycle phase enum
- `HookCapability`: `observe_only`, `transform`, `decision_support`
- `HookGroup`: execution group ordering (`system`, `runtime_adapter`, `user`)
- `HookRegistration`: one registered phase handler
- `HookRegistry`: ordered collection of registrations

## Ordering And Failure Rules

- Execution order is `group -> order -> registration order`.
- Only early phases may be marked `critical=True`:
  - `prepare`
  - `assemble_context`
  - `before_llm`
  - `before_tool_call`
- Non-critical hook failures are isolated, logged, and recorded as `HookFailed`
  run-log facts.
- Critical hook failures are re-raised and fail the run.

## Public Phases

```python
from agiwo.agent import HookPhase

HookPhase.PREPARE
HookPhase.ASSEMBLE_CONTEXT
HookPhase.BEFORE_LLM
HookPhase.AFTER_LLM
HookPhase.BEFORE_TOOL_CALL
HookPhase.AFTER_TOOL_CALL
HookPhase.BEFORE_COMPACTION
HookPhase.AFTER_COMPACTION
HookPhase.BEFORE_RETROSPECT
HookPhase.AFTER_RETROSPECT
HookPhase.BEFORE_TERMINATION
HookPhase.AFTER_TERMINATION
HookPhase.AFTER_STEP_COMMIT
HookPhase.RUN_FINALIZED
HookPhase.MEMORY_PERSIST
HookPhase.COMPACTION_FAILED
```

## Payload Contracts

Handlers always receive a single `payload: dict[str, Any]`.

### Transform Phases

These phases may return a partial dict that only changes allowlisted fields.

#### `prepare`

Input payload:

```python
{
    "user_input": user_input,
    "context": context,
    "prelude_text": None,
}
```

Allowlisted output fields:

```python
{"prelude_text": "Please answer in Chinese."}
```

#### `assemble_context`

Input payload:

```python
{
    "user_input": user_input,
    "context": context,
    "memories": [],
    "context_additions": [],
}
```

Allowlisted output fields:

```python
{"memories": memories, "context_additions": []}
```

#### `before_llm`

Input payload:

```python
{
    "messages": messages,
    "context": context,
    "model_settings_override": None,
    "llm_advice": None,
}
```

Allowlisted transform fields:

```python
{"messages": updated_messages, "model_settings_override": None}
```

#### `before_tool_call`

Input payload:

```python
{
    "tool_call_id": tool_call_id,
    "tool_name": tool_name,
    "parameters": parameters,
    "context": context,
    "tool_advice": None,
}
```

Allowlisted transform fields:

```python
{"parameters": updated_parameters}
```

### Decision-Support Phases

Decision-support handlers also return partial dicts, but may only write the
phase-specific `*_advice` field:

- `before_llm`: `llm_advice`
- `before_tool_call`: `tool_advice`
- `before_compaction`: `compaction_advice`
- `before_retrospect`: `retrospect_advice`
- `before_termination`: `termination_advice`

### Observe-Only Phases

These phases should return nothing useful; results are ignored.

- `after_llm`
- `after_tool_call`
- `after_compaction`
- `after_retrospect`
- `after_termination`
- `after_step_commit`
- `run_finalized`
- `memory_persist`
- `compaction_failed`

## Examples

### Rewrite Messages Before LLM

```python
from agiwo.agent import HookPhase, HookRegistry, transform


async def prepend_notice(payload: dict) -> dict:
    messages = list(payload["messages"])
    messages.append({"role": "user", "content": "Double-check file paths."})
    return {"messages": messages}


hooks = HookRegistry(
    [transform(HookPhase.BEFORE_LLM, "prepend_notice", prepend_notice)]
)
```

### Observe Each Tool Call

```python
from agiwo.agent import HookPhase, HookRegistry, observe


async def log_tool_call(payload: dict) -> None:
    print(
        payload["tool_call_id"],
        payload["tool_name"],
        payload["parameters"],
    )


hooks = HookRegistry(
    [observe(HookPhase.BEFORE_TOOL_CALL, "log_tool_call", log_tool_call)]
)
```

### Track Committed Steps

```python
from agiwo.agent import HookPhase, HookRegistry, observe


async def log_step(payload: dict) -> None:
    step = payload["step"]
    print(step.role.value, step.sequence, step.id)


hooks = HookRegistry(
    [observe(HookPhase.AFTER_STEP_COMMIT, "log_step", log_step)]
)
```

### Persist External Memory

```python
from agiwo.agent import HookPhase, HookRegistry, observe


async def write_external_memory(payload: dict) -> None:
    user_input = payload["user_input"]
    result = payload["result"]
    context = payload["context"]
    print(context.run_id, user_input, result.response)


hooks = HookRegistry(
    [observe(HookPhase.MEMORY_PERSIST, "write_external_memory", write_external_memory)]
)
```

## Notes

- Prefer `observe(...)`, `transform(...)`, and `decision_support(...)` over
  constructing `HookRegistration` directly.
- If a transform hook returns fields outside the phase allowlist, registration
  succeeds but dispatch fails with a validation error when the hook runs.
- `context` is included in runtime payloads so hooks can inspect run/session
  metadata without importing agent internals directly.
