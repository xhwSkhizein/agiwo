# Documentation

Welcome to the Agiwo documentation. Agiwo is a streaming-first AI Agent SDK and Console for Python.

## Quick Links

- **[Getting Started](./getting-started.md)** — Installation, first agent, and configuration
- **[Core Concepts](./concepts/agent.md)** — Understand Agent, Model, Tool, and Scheduler

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

## API Reference

- **[Agent API](./api/agent.md)** — `Agent`, `AgentConfig`, `RunOutput`, `AgentStreamItem`
- **[Tool API](./api/tool.md)** — `BaseTool`, `ToolResult`, `ToolContext`
- **[Model API](./api/model.md)** — `Model`, provider implementations
- **[Scheduler API](./api/scheduler.md)** — `Scheduler`, orchestration methods
