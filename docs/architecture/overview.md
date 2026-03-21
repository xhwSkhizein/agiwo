# Architecture Overview

Agiwo is designed as a layered system where each layer has clear responsibilities and dependency boundaries.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Console                              │
│  FastAPI Server + Next.js UI + Channel Integrations         │
└─────────────────────────┬───────────────────────────────────┘
                          │ uses
┌─────────────────────────▼───────────────────────────────────┐
│                      Scheduler                              │
│  Orchestrate, spawn, sleep/wake, steer, cancel              │
└─────────────────────────┬───────────────────────────────────┘
                          │ uses
┌─────────────────────────▼───────────────────────────────────┘
│                       Agent                                 │
│  Run, stream, hooks, tools, prompt, storage, trace          │
└─────────┬──────────────┬──────────────┬─────────────────────┘
          │              │              │
   ┌──────▼──────┐ ┌─────▼─────┐ ┌─────▼──────┐
   │    Tool     │ │   Model   │ │ Observability│
   │  Execute    │ │  LLM API  │ │   Traces     │
   └─────────────┘ └───────────┘ └──────────────┘
```

**Dependency rule**: `Console → Scheduler → Agent → {Tool, Model, Observability}`. Never reverse.

## Module Map

### `agiwo/agent/` — Agent Runtime

The core execution engine:

| Module | Responsibility |
|--------|---------------|
| `agent.py` | Public facade — `Agent` class with `run()`, `run_stream()`, `start()` |
| `config.py` | `AgentConfig` — pure configuration, no live objects |
| `execution.py` | `AgentExecutionHandle` — per-run control surface |
| `hooks.py` | `AgentHooks` — lifecycle callbacks |
| `input.py` | `UserInput` — input normalization |
| `runtime/` | `AgentStreamItem`, `RunOutput`, `StepRecord`, `RunMetrics` — runtime domain types |
| `lifecycle/` | **Internal** — definition/resource/session/orchestrator owners |
| `engine/` | **Internal** — single-run state, recorder, llm/tool/termination/compaction |
| `storage/` | Run/step/session persistence |
| `trace/` | Agent-to-trace adapter |
| `runtime_tools/` | Agent-as-tool adapter, runtime tool contracts |
| `prompt/` | System prompt construction |
| `streaming.py` | Stream consumption helpers |

**Do not import from `agiwo.agent.lifecycle` or `agiwo.agent.engine` outside the `agiwo.agent` package.**

### `agiwo/llm/` — Model Providers

| Module | Responsibility |
|--------|---------------|
| `base.py` | `Model` ABC and `StreamChunk` |
| `openai.py` | OpenAI provider |
| `anthropic.py` | Anthropic provider |
| `deepseek.py` | DeepSeek provider |
| `nvidia.py` | NVIDIA provider |
| `bedrock_anthropic.py` | AWS Bedrock (Anthropic models) |
| `factory.py` | Model construction from config |
| `message_converter.py` | Message format normalization |
| `event_normalizer.py` | Provider event → `StreamChunk` normalization |

### `agiwo/tool/` — Tool System

| Module | Responsibility |
|--------|---------------|
| `base.py` | `BaseTool`, `ToolResult`, `ToolDefinition` |
| `context.py` | `ToolContext` — runtime info for tool execution |
| `cache.py` | Session-scoped result caching |
| `builtin/` | Built-in tools (bash, web, memory) |
| `authz/` | Tool authorization domain types |
| `process/` | Background process registry |
| `storage/citation/` | Citation storage for web tools |

### `agiwo/scheduler/` — Orchestration

| Module | Responsibility |
|--------|---------------|
| `scheduler.py` | Public facade — `Scheduler` class |
| `engine.py` | `SchedulerEngine` — orchestration API |
| `runner.py` | Single agent execution cycle |
| `coordinator.py` | In-process state (agents, handles, tasks) |
| `control.py` | Interface for scheduler tools |
| `state_ops.py` | State transitions |
| `tick_ops.py` | Scheduler tick phases |
| `tree_ops.py` | Tree cancel/shutdown |
| `guard.py` | Task spawn/wake guardrails |
| `store/` | State persistence (memory/sqlite) |
| `runtime_tools.py` | Scheduler-specific tools (spawn, sleep, etc.) |

### `agiwo/observability/` — Tracing

| Module | Responsibility |
|--------|---------------|
| `base.py` | `BaseTraceStorage` interface |
| `trace.py` | Trace and Span models |
| `store.py` | Storage implementations |
| `factory.py` | Trace storage construction |

### Other Modules

| Module | Responsibility |
|--------|---------------|
| `agiwo/config/` | SDK configuration, provider enums |
| `agiwo/embedding/` | Embedding abstractions (local, OpenAI) |
| `agiwo/memory/` | MEMORY indexing and search |
| `agiwo/skill/` | Skill discovery, loading, registry |
| `agiwo/workspace/` | Workspace layout and bootstrap |
| `agiwo/utils/` | Cross-cutting utilities (logging, pools, retry) |

## Design Principles

1. **Streaming-first**: All LLM interaction flows through `AsyncIterator[StreamChunk]`
2. **Explicit wiring**: No global singletons; dependencies are passed explicitly
3. **Facade pattern**: Complex internals hidden behind thin public facades (`Agent`, `Scheduler`)
4. **Separation of concerns**: Definition (config/tools) vs. resource (storage/runs) vs. execution (runner)
5. **No background daemons**: Indexing and sync happen on-demand, not via watchers
6. **Graceful degradation**: Missing embedding provider → BM25-only; missing sqlite-vec → memory computation

## Data Flow: A Single Agent Run

```
User Input
  │
  ▼
Agent.run("Hello")
  │
  ▼
AgentExecutionHandle (start)
  │
  ▼
ExecutionOrchestrator.start_root()
  │
  ├── Build system prompt (AgentDefinitionRuntime.prompt_runtime)
  │     └── Inject memories, skills, environment
  │
  ├── ExecutionEngine (single-run pipeline)
  │     │
  │     ├─ 1. Prepare state + load session history
  │     │
  │     ├─ 2. Run before_run hooks + memory retrieval
  │     │
  │     ├─ 3. LLM call → Model.arun_stream()
  │     │     └── StreamChunk → AgentStreamItem
  │     │
  │     ├─ 4. Parse response
  │     │     ├── Text → done?
  │     │     └── Tool calls → execute
  │     │
  │     ├─ 5. Execute tools (concurrent-safe batched)
  │     │     └── BaseTool.execute() → ToolResult
  │     │
  │     └─ 6. Compact / terminate / finalize
  │
  ├── RunRecorder
  │     ├── Write step records
  │     ├── Update trace spans
  │     └── Fan out to stream subscribers
  │
  └── RunOutput (final result)
```
