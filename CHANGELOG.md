# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

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
