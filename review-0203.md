# Project Review - 2026-02-03

## Scope Summary
AGIWO is an AI agent framework with a core execution loop (`agiwo/agent`), tool execution (`agiwo/tool`), session persistence (`agiwo/agent/session`), and observability (`agiwo/observability`). The main flow is: `AgiwoAgent.run()` → `RunLifecycle` + `SideEffectProcessor` → `AgentExecutor` loop → `LLMStreamHandler` streaming → `StepBuilder` + `StepFactory` → `SessionStore` + event stream via `Wire`. Tool execution is handled by `ToolExecutor`, while tracing is handled by `TraceCollector` with optional `TraceStore`/`SQLiteTraceStore`/OTLP export.

All items below are framed to preserve existing behavior unless explicitly flagged as “behavior decision” (i.e., may require product intent).

## Batch 1 - Blocking Mismatches (Runtime Errors)
1. `agiwo/agent/llm_handler.py` constructs `StepMetrics(exec_start_at=...)`, but `StepMetrics` defines `start_at`. This will raise `TypeError` on the first streamed call. Align field names.
2. `agiwo/agent/run_state.py` builds `RunMetrics(duration=...)`, but `RunMetrics` defines `duration_ms`. This will raise `TypeError` when building `RunOutput`. Align field names or add a compatibility alias.
3. `agiwo/agent/limit_checker.py` reads `self.config.run_timeout`, but `AgentConfigOptions` does not define it. This will raise `AttributeError`. Decide whether to add `run_timeout` or remove this branch.
4. `agiwo/agent/executor.py` constructs `ToolExecutor(..., permission_manager=..., default_timeout=...)` but `ToolExecutor.__init__` does not accept these keyword args. This will raise `TypeError`. Reconcile the signatures.
5. `agiwo/agent/base.py` uses a mutable default `tools: list[BaseTool] = []`. This can leak tools across instances. Use `None` + `[]` inside `__init__`.

## Batch 2 - Agent Loop & State Clarity
1. `agiwo/agent/run_state.py` has token accumulation logic in `MetricsTracker.track` that relies on truthy checks; use `is not None` and extract `_track_tokens` / `_track_assistant` / `_track_tool` helpers to reduce nesting without changing behavior.
2. `agiwo/agent/limit_checker.py` compares `state.tracker.total_tokens` against `max_output_tokens`. The name suggests output-only but the logic is total tokens. Either rename to `max_total_tokens` or document the behavior to prevent misunderstanding. Behavior decision.
3. `agiwo/agent/run_lifecycle.py` has a long `__aexit__` branch. Split into `_handle_success`, `_handle_failure`, and `_handle_incomplete` helpers to reduce nesting and keep the flow readable.
4. `agiwo/agent/side_effect_processor.py` returns sequence `1` when `session_store` is absent, which makes every step share the same sequence. If ordering still matters in memory-only mode, introduce an in-memory counter. Behavior decision.
5. `agiwo/agent/sequence_manager.py` mutates `context.metadata` while `ExecutionContext` is frozen. Consider moving `seq_start` into a dedicated field or copying the metadata dict before mutation to avoid surprising side effects.
6. `agiwo/agent/step_builder.py` mixes content accumulation and delta emission in a single method; extract small helpers for “append content”, “append reasoning”, and “accumulate tools” to reduce branching and improve testability.
7. `agiwo/agent/step_builder.py` returns tool calls in `dict` insertion order. If deterministic order by index matters, sort by index before returning. Behavior decision.

## Batch 3 - Tooling & Permissions
1. `agiwo/tool/executor.py` uses `asyncio.wait` and then iterates `tasks` rather than `done`. `asyncio.gather(return_exceptions=True)` would simplify result collection and reduce error-handling duplication without changing semantics.
2. `agiwo/tool/executor.py` has complex argument parsing inside `aexecute`. Extract `_parse_tool_args` to reduce nested try/except blocks and centralize error messaging.
3. `agiwo/tool/executor.py` does not use `BaseTool.is_concurrency_safe`, so unsafe tools still run in parallel. If the flag is intended, add serialization for unsafe tools. Behavior decision.
4. `agiwo/tool/cache.py` implements `ToolResultCache`, but `ToolExecutor` never reads/writes it. Either wire it in or remove the cache layer to reduce dead code.
5. `agiwo/tool/cache.py` uses `session_id[:16]` prefix in `clear_session`, but cache keys are SHA256 hashes, so this will almost never match. Fix the key scheme or store a reverse index per session.
6. `agiwo/tool/permission/manager.py` claims LRU eviction but deletes an arbitrary first key in a dict. Either switch to `OrderedDict` or rename the comment to avoid misleading behavior.
7. `agiwo/tool/permission/consent_waiter.py` performs cleanup in both `except` and `finally`; consolidate into a single cleanup block to reduce duplication.

## Batch 4 - Storage & Observability
1. `agiwo/agent/session/sqlite.py` repeats `from dataclasses import asdict, is_dataclass` and has duplicated JSON conversion paths. Extract a helper to serialize metrics and JSON fields consistently.
2. `agiwo/agent/session/sqlite.py` uses `if isinstance(..., str) is False`; prefer `not isinstance` for clarity and consistency.
3. `agiwo/observability/store.py` and `agiwo/observability/sqlite_store.py` share buffer/subscriber logic. Consider a small base class or mixin to reduce duplication and keep behavior aligned.
4. `agiwo/observability/collector.py` calls `asyncio.create_task` on every checkpoint event. Consider a buffered or debounced save to avoid task churn under heavy streaming. Behavior decision.
5. `agiwo/observability/collector.py` has a large `_process_event` method. Break it into per-event handlers to reduce branching depth.

## Batch 5 - LLM Provider Layer
1. `agiwo/llm/openai.py`, `agiwo/llm/anthropic.py`, `agiwo/llm/deepseek.py`, and `agiwo/llm/nvidia.py` duplicate request logging and parameter building. Extract a shared helper for request logging and param assembly to reduce drift.
2. `agiwo/agent/llm_handler.py` only records `temperature`, `max_tokens`, and `top_p` in `llm_request_params`. Consider capturing `frequency_penalty` and `presence_penalty` as well, since they are part of `Model`. Behavior decision.
3. `agiwo/llm/helper.py` mixes attribute access and dict access in a single function. Split into two small helpers (`_get_usage_attr` / `_get_usage_key`) to improve readability without changing output.

## Batch 6 - Skills & Project Hygiene
1. `agiwo/skills/manager.py` requires `initialize()` to populate metadata, but `AgentConfigOptions` never calls it. Consider auto-initialization on startup or explicit call in `AgiwoAgent` to avoid empty skills sections. Behavior decision.
2. `agiwo/agent/compact.py` is an empty placeholder. Either remove it or add a short comment explaining the future plan to avoid dead-file noise.
3. `AGENTS.md` is missing from the repo root. Per your requirement, add and maintain a development规范文档 in `AGENTS.md` and update it alongside refactors.
