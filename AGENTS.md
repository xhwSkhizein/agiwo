# AGENTS Development Guide

## Core Principles
- Preserve behavior. Refactors must not change outputs, side effects, or execution flow.
- Clarity over cleverness. Prefer explicit, readable code even if slightly longer.
- Avoid nested ternaries. Use `if/elif/else` or `match` where appropriate.
- Reduce incidental complexity. Remove redundant abstractions and dead code.
- Keep responsibilities focused. Do not overload single functions or classes.

## Refactor Rules
- Do not change public interfaces unless required to fix a bug.
- Keep error messages and logging stable unless they are incorrect.
- Prefer small helpers over large methods with deep nesting.
- Name things by intent: `*_factory`, `*_handler`, `*_tracker`, etc.

## Python Style
- Use type hints for public methods and core data structures.
- Prefer `is not None` for sentinel checks; use truthy checks only when intentional.
- Avoid mutable default arguments; use `None` + initialization inside `__init__`.
- Keep async code explicit: do not hide awaits behind implicit helpers.

## Observability
- Logs must be structured and actionable (event name + key fields).
- Avoid logging sensitive values (API keys, tokens, credentials).
- Trace/span models should stay serializable and consistent across stores.

## Documentation
- Keep README up to date for environment and run instructions.
- Document any behavior decision before implementing it.
