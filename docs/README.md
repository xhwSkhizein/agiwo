# Documentation

Welcome to the Agiwo documentation. Agiwo is a streaming-first AI Agent SDK and Console for Python.

## Quick Links

- **[Getting Started](./getting-started.md)** — Installation, first agent, and configuration
- **[Core Concepts](./concepts/model.md)** — Understand Model, Tool, and Scheduler

## Guides

- **[Custom Tools](./guides/custom-tools.md)** — Build your own tools
- **[Multi-Agent & Composition](./guides/multi-agent.md)** — Agent-as-tool and scheduler orchestration
- **[Streaming](./guides/streaming.md)** — Real-time streaming responses
- **[Hooks](./guides/hooks.md)** — Observe and intercept agent lifecycle events
- **[Storage & Observability](./guides/storage.md)** — Persist runs, sessions, and traces
- **[Skills](./guides/skills.md)** — File-based skill discovery and loading

## Console

- **[Console Overview](./console/overview.md)** — FastAPI control plane and Next.js web UI
- **[Console API](./console/api.md)** — REST and SSE endpoints
- **[Feishu Integration](./console/feishu.md)** — Channel runtime for Feishu

## Architecture

- **[Architecture Overview](./architecture/overview.md)** — High-level design and module boundaries
- **[Memory System](./architecture/memory.md)** — Hybrid retrieval with BM25 + vector search
- **[Scheduler / Console Runtime Refactor Proposal](./architecture/scheduler-console-runtime-refactor.md)** — Root/child runtime boundary and materialization refactor proposal

## API Reference

- **[Model API](./api/model.md)** — `Model`, provider implementations
- **[Tool API](./api/tool.md)** — `BaseTool`, `ToolResult`, `ToolContext`
- **[Scheduler API](./api/scheduler.md)** — `Scheduler`, orchestration methods

## Public Website

The public documentation site is intended to live at `https://docs.agiwo.o-ai.tech`.

The source for the public site lives under `website/`. The Markdown files in `docs/` remain the repository-native documentation source and design/archive materials.
