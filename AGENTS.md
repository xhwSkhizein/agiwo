# AGENTS Development Guide

## Architecture Overview

Agiwo follows a **Main Agent + Agent as Tools** pattern. The SDK does NOT implement multi-agent orchestration; instead, agents can be nested via `AgentTool`.

Key components:
- `Agent` — concrete class (not ABC), the single entry point for execution
- `AgentExecutor` — internal loop coordinator (LLM calls + tool execution)
- `AgentHooks` — lifecycle hooks for extensibility (no subclassing needed)
- `Model` — abstract LLM provider; subclasses override `_resolve_api_key`/`_resolve_base_url`/`_create_client`
- `BaseTool` — abstract tool interface; `AgentTool` wraps an `Agent` as a tool
- `RunStepStorage` / `BaseTraceStorage` — injected via constructor, not global config

## Core Principles

- Clarity over cleverness. Prefer explicit, readable code even if slightly longer.
- SOLID + KISS. Single responsibility per class. No unnecessary abstractions.
- No backward compatibility unless explicitly requested. Delete legacy code promptly.
- No circular dependencies. If detected, refactor component coupling immediately.
- All imports at file top. No local/lazy imports.

## Code Style

- Python 3.11+. No `from __future__ import annotations`.
- Type hints on all public methods and core data structures.
- PascalCase for classes (`AgiwoSettings`), snake_case for functions/variables.
- Prefer `is not None` for sentinel checks; truthy checks only when intentional.
- Avoid mutable default arguments; use `None` + initialization in `__init__`.
- Keep async code explicit: do not hide awaits behind implicit helpers.
- Prefer small helpers over large methods with deep nesting.
- Name by intent: `*_handler`, `*_builder`, `*_store`, `*_hook`.

## Adding New LLM Providers

Subclass `Model` and override three methods:
- `_resolve_api_key() -> str | None`
- `_resolve_base_url() -> str | None`
- `_create_client() -> <ClientType>`

For OpenAI-compatible APIs, subclass `OpenAIModel` instead (only override resolve methods).

## Adding New Hooks

Add the hook type alias and field to `agiwo/agent/hooks.py`. Call the hook at the appropriate point in `Agent._execute_workflow` or `AgentExecutor._run_loop` / `_execute_tools`.

## Observability

- Logs must be structured and actionable (event name + key fields).
- Never log sensitive values (API keys, tokens, credentials).
- Trace/span models must stay serializable and consistent across stores.
- OTLP export is optional; configure via environment variables.

## Testing

- Unit tests in `tests/` use mocks (no real API calls).
- Integration tests (`test_real_agent.py`, `test_real_api.py`) require `.env` API keys.
- Run: `uv run pytest tests/ -v`

## Dependency Injection

Stores (`RunStepStorage`, `BaseTraceStorage`) are injected via `Agent` constructor. The SDK does NOT auto-create stores from global settings. Users choose their persistence layer explicitly.
