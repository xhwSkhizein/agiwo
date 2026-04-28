# Documentation

Welcome to the Agiwo documentation. Agiwo is a streaming-first AI agent SDK and Console for Python.

## Quick Links

- **[Getting Started](./getting-started.md)** — installation, first agent, streaming, and builtin tools
- **[Agent](./concepts/agent.md)** — public `Agent` API, config, and execution handles
- **[Core Concepts](./concepts/model.md)** — understand Model, Tool, and Scheduler

## Guides

- **[Custom Tools](./guides/custom-tools.md)** — build your own tools
- **[Multi-Agent & Composition](./guides/multi-agent.md)** — `Agent.as_tool()` and scheduler orchestration
- **[Streaming](./guides/streaming.md)** — real-time streaming responses
- **[Hooks](./guides/hooks.md)** — observe and intercept agent lifecycle events
- **[Storage & Observability](./guides/storage.md)** — persist runs, traces, and metrics
- **[Context Optimization](./guides/context-optimization.md)** — context rollback, goal-directed review, and introspection repair
- **[Skills](./guides/skills.md)** — file-based skill discovery and allowlisting
- **[Release Publishing](./release.md)** — maintainer runbook for GitHub Release driven PyPI publishing

## Console

- **[Console Overview](./console/overview.md)** — session-first FastAPI control plane and web UI
- **[Console API](./console/api.md)** — REST and SSE endpoints for agents, sessions, scheduler, traces, and runtime config
- **[Console Docker Deployment](./console/docker.md)** — managed single-container deployment
- **[Feishu Integration](./console/feishu.md)** — channel runtime for Feishu

## Architecture

- **[Architecture Overview](./architecture/overview.md)** — high-level design and module boundaries
- **[Memory System](./architecture/memory.md)** — hybrid retrieval with BM25 + vector search
- **[Scheduler / Console Runtime Refactor Proposal](./architecture/scheduler-console-runtime-refactor.md)** — root/child runtime boundary and materialization refactor proposal

## API Reference

- **[Model API](./api/model.md)** — `Model`, `ModelSpec`, and provider implementations
- **[Tool API](./api/tool.md)** — `BaseTool`, `ToolResult`, and `ToolContext`
- **[Scheduler API](./api/scheduler.md)** — `Scheduler`, orchestration methods, and state models

## Public Website

The public documentation site is intended to live at `https://docs.agiwo.o-ai.tech`.

The source for the public site lives under `website/`. The Markdown files in `docs/` remain the repository-native documentation source and design/archive materials.
