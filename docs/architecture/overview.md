# Architecture Overview

## Console Interaction Path

```
Console / Feishu → SessionGateway → MainAgent.accept → Agent Loop / RunLog
```

The Console and Feishu adapters share the same session-first conversation semantics:

- **Entry adapters** (Console Web, Feishu) handle transport-specific concerns (SSE, long-connection, message parsing)
- **SessionGateway / SessionTurnService** accept user input via `MainAgent.accept` and project history from RunLog
- **Scheduler** owns waitset, cancel, and Worker parent registration—not Session chat entry
- **Agent / MainAgent** execute the work; results are projected from RunLog facts

## Key Domain Objects

| Object | Description |
|--------|-------------|
| Session | Primary conversation container. Users create, switch, and resume sessions across entrypoints. |
| MainAgent | Session-scoped live executor; `accept` is the user-input API. |
| Run | One MainAgent (or Worker) execution cycle, recorded in RunLog. |
| Scheduler | Waitset + cancel + persistent-root `enqueue_input` for orchestration trees. |

## Design Principles

1. **Console is a projection layer** — it views SDK execution facts, never creates a second execution truth
2. **One session entry** — `MainAgent.accept`; no Scheduler Session facade
3. **Fork for branching** — when work diverges, fork to a new session rather than overloading one session
4. **RunLog-first projections** — session history, stream, and trace views are built from committed RunLog entries
