# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed
- Scheduler upgrade note: `PendingEvent.USER_HINT` payloads are now stored as `{"user_input": UserMessage.to_storage_value(...)}` instead of the legacy plain-text shape.

### Upgrade Notes
- Scheduler SQLite users must recreate `scheduler.db` before upgrading to builds that add `PendingEvent.urgent` and the structured `PendingEvent.USER_HINT` payload. Existing `pending_events` tables created by older versions do not have the new `urgent` column, so the first scheduler `save_event()` write will fail with `no column named urgent`. Delete `scheduler.db`, or back it up and recreate it, before starting the upgraded build.

### Added
- Comprehensive English documentation (`docs/`)
- MIT License
- Contributing guide (`CONTRIBUTING.md`)
- CI pipeline (lint + test on PR)
- GitHub issue and PR templates
- Release packaging and documentation cleanup for the first public release

## [0.1.0] - 2026-04-16

### Added
- Streaming-first Agent SDK with `run()`, `run_stream()`, and `start()` API
- Tool system with `BaseTool`, `ToolResult`, session caching, and builtin tools (bash, web_search, web_reader, memory_retrieval)
- Agent-as-tool composition via `as_tool()` / `AgentTool`
- Scheduler orchestration layer with spawn, sleep/wake, steer, cancel
- LLM providers: OpenAI, Anthropic, DeepSeek, NVIDIA, Bedrock, OpenAI-compatible, Anthropic-compatible
- Hook system for run, tool, LLM, step, and memory lifecycle events
- Run/step/session/trace storage backends (memory, SQLite, MongoDB)
- Hybrid BM25 + vector memory retrieval with temporal decay and MMR
- File-based skill discovery and loading
- Console: FastAPI backend with REST + SSE APIs
- Console: Next.js web UI for agent management, chat, scheduler, and traces
- Console: Feishu channel integration
