# Configuration Refactor Plan

This document records the current configuration ownership split and the rules for
adding new config.

## Ownership

- SDK runtime config lives in `agiwo/config/settings.py` and uses `AGIWO_*`.
- Console deployment config lives in `console/server/config.py` and uses
  `AGIWO_CONSOLE_*`.
- External provider credentials keep their canonical names such as
  `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`, and
  `NVIDIA_BUILD_API_KEY`.

## Current Tool Model Pattern

- Builtin tools that need a `Model` must construct it from config rather than
  receiving a live model from `Agent` or Console builder.
- Global defaults for tool-created models live under
  `AGIWO_TOOL_DEFAULT_MODEL_*`.
- Per-tool overrides use `AGIWO_TOOL_<TOOL_NAME>_MODEL_*`.
- `web_reader` currently consumes this configuration.
- Compatible providers (`openai-compatible`, `anthropic-compatible`) are protocol adapters,
  not credential namespaces. They must carry explicit `base_url` plus their own
  credential reference/config, and must not fall back to `OPENAI_*` or
  `ANTHROPIC_*`.

## Rules

- Read environment variables only in configuration modules.
- Prefer one shared abstraction for model construction; do not add tool-specific
  ad hoc HTTP clients that bypass `agiwo.llm`.
- When adding or renaming config, update:
  - `.env.example`
  - `console/.env.example`
  - `README.md`
  - this file
