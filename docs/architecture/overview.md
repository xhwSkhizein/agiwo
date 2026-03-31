# Architecture Overview

## Console Interaction Path

```
Console / Feishu → Session application layer → Scheduler → Agent
```

The Console and Feishu adapters share the same session-first conversation semantics:

- **Entry adapters** (Console Web, Feishu) handle transport-specific concerns (SSE, long-connection, message parsing)
- **Session application layer** (`SessionManager`, `SessionContextService` in `console/server/channels/session/`) owns session lifecycle, implicit task management, and fork semantics
- **Scheduler** mediates all execution — both Console and Feishu route through `scheduler.route_root_input()`
- **Agent** executes the actual work, with results projected back as SDK execution facts

## Key Domain Objects

| Object | Description |
|--------|-------------|
| Session | Primary conversation container. Users create, switch, and resume sessions across entrypoints. |
| Task | Unit of work inside a Session. Created implicitly when the first message arrives. |
| Run | Execution-level realization of task work, derived from SDK RunStep records. |

## Design Principles

1. **Console is a projection layer** — it views SDK execution facts, never creates a second execution truth
2. **One session = one task by default** — keeps the mental model simple
3. **Fork for branching** — when work diverges, fork to a new session rather than overloading one session
4. **Scheduler-mediated execution** — Console no longer calls `agent.start()` directly
5. **RunStep-first projections** — task/run views are built from SDK-provided execution records
