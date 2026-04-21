# Logging Guidelines

## Format

```python
logger.{level}("event_name", key=value, ...)
```

## Event Name Rules

- Use `snake_case`, verb + noun
- Examples: `llm_stream_started`, `tool_execution_failed`, `config_rebind_failed`
- **Forbidden**: Chinese characters, free-form strings (e.g., `"compaction failed"`)

## Log Level Guidelines

- `debug`: Debug information (detailed execution paths)
- `info`: Key business events (request start/end, state changes)
- `warning`: Recoverable exceptions (retry, degradation)
- `error`: Errors needing attention (but not crashing)
- `exception`: Uncaught exceptions (with stack trace)

## Parameter Guidelines

- Use `key=value` format for structured logging
- Example: `logger.info("tool_execution_started", tool_name="bash", step_id=123)`

## Exception Handling Guidelines

- **Forbidden**: Abuse of `# noqa: BLE001`
- Use specific exception catching: `except ValueError as exc`
- Only use `except Exception as exc` for unknown exceptions and log detailed error

## Examples

### Good

```python
logger.info("llm_request", model=model, messages_count=len(messages))
logger.warning("tool_execution_failed", tool_name="bash", error=str(exc))
logger.exception("scheduler_tick_error")
```

### Bad

```python
logger.info("LLM started")  # Free-form string
logger.warning("工具执行失败")  # Chinese
logger.error("Error", exc)  # Vague message
except Exception:  # noqa: BLE001  # Silent exception swallowing
```

## Migration Plan

1. **Phase 1**: Create this document
2. **Phase 2**: Fix critical paths (run_loop, llm_caller, scheduler/engine, scheduler/runner)
3. **Phase 3**: Fix other modules (tools, agent)
4. **Phase 4**: Fix low-frequency modules (utils, observability)
5. **Phase 5**: Add pre-commit hook for enforcement
