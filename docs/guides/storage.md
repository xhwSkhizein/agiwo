# Storage & Observability

Agiwo separates storage into two independent concerns: run-log persistence and
trace collection.

## Storage Layers

| Layer | Purpose | Interface |
|-------|---------|-----------|
| Run Log Storage | Persist canonical runtime facts and replayable views | `RunLogStorage` |
| Trace Storage | Persist trace snapshots built from committed run-log facts | `BaseTraceStorage` |

## Run Log Storage

Records canonical runtime facts as append-only `RunLog` entries:

```python
from agiwo.agent import RunLogStorageConfig
from agiwo.agent.storage.factory import create_run_log_storage

# SQLite backend
storage = create_run_log_storage(
    RunLogStorageConfig(storage_type="sqlite", config={"db_path": "runs.db"})
)

# Memory backend (for testing)
storage = create_run_log_storage(RunLogStorageConfig(storage_type="memory"))
```

### What gets stored

- **Run facts**: `RunStarted`, `RunFinished`, `RunFailed`, `TerminationDecided`
- **LLM facts**: `LLMCallStarted`, `LLMCallCompleted`
- **Committed step facts**: user, assistant, and tool step commits
- **Runtime decisions**: compaction, step-back, rollback, hook failures

Replay/query helpers such as `list_step_views(...)`, `list_run_views(...)`, and
`get_runtime_decision_state(...)` are rebuilt from the stored `RunLog`.

## Session Storage

The SDK does not expose a standalone session storage abstraction. Session lifecycle for the Console is owned by `console/server/services/session_store/` plus the runtime services under `console/server/services/runtime/`.

## Trace Storage

Collects trace snapshots for debugging and monitoring:

```python
from agiwo.agent import TraceStorageConfig
from agiwo.observability import create_trace_storage

# SQLite
trace_storage = create_trace_storage(
    TraceStorageConfig(storage_type="sqlite", config={"db_path": "traces.db"})
)

# Memory
trace_storage = create_trace_storage(TraceStorageConfig(storage_type="memory"))
```

Pass storage configuration through `AgentOptions.storage`:

```python
from agiwo.agent import (
    Agent,
    AgentConfig,
    AgentOptions,
    AgentStorageOptions,
    TraceStorageConfig,
)

agent = Agent(
    AgentConfig(
        name="assistant",
        description="...",
        system_prompt="...",
        options=AgentOptions(
            storage=AgentStorageOptions(
                trace_storage=TraceStorageConfig(
                    storage_type="sqlite",
                    config={"db_path": "traces.db"},
                )
            )
        ),
    ),
    model=model,
)
```

### Trace Structure

```
Trace (one per session/runtime trace id)
├── Span: Agent Execution
│   ├── Span: LLM Call (with token usage)
│   ├── Span: Tool Execution (with duration)
│   └── Span: Tool Execution
├── Span: LLM Call
└── Span: Child Agent (if spawned)
```

### Querying Traces

```python
# Get a specific trace
trace = await trace_storage.get_trace(trace_id)

# Query traces with filters (agent_id, session_id, status, time range, etc.)
from agiwo.observability.base import TraceQuery
traces = await trace_storage.query_traces(
    TraceQuery(agent_id="agent-123", limit=50)
)
```

## Choosing Backends

| Backend | Use Case | Persistence |
|---------|----------|-------------|
| `memory` | Testing, ephemeral workloads | No (lost on restart) |
| `sqlite` | Single-node deployments, development | Yes |

## Configuration

Storage constructors are config-driven:

```python
from agiwo.agent import RunLogStorageConfig, TraceStorageConfig
from agiwo.agent.storage.factory import create_run_log_storage
from agiwo.observability import create_trace_storage

run_log_storage = create_run_log_storage(
    RunLogStorageConfig(
        storage_type="sqlite",
        config={"db_path": "./data/runs.db"},
    )
)

trace_storage = create_trace_storage(
    TraceStorageConfig(
        storage_type="sqlite",
        config={
            "db_path": "./data/traces.db",
        },
    )
)
```

Or configure storage through `AgentOptions.storage`:

```python
from agiwo.agent import (
    AgentConfig,
    AgentOptions,
    AgentStorageOptions,
    RunLogStorageConfig,
    TraceStorageConfig,
)

config = AgentConfig(
    name="assistant",
    options=AgentOptions(
        storage=AgentStorageOptions(
            run_log_storage=RunLogStorageConfig(
                storage_type="sqlite",
                config={"db_path": "./data/runs.db"},
            ),
            trace_storage=TraceStorageConfig(
                storage_type="sqlite",
                config={"db_path": "./data/traces.db"},
            ),
        )
    ),
)
```

## Cost Tracking

When `input_price`, `output_price`, and `cache_hit_price` are set on the Model, the SDK automatically tracks costs:

```python
from agiwo.llm import OpenAIModel

model = OpenAIModel(
    name="gpt-5.4",
    input_price=0.005,     # per 1K tokens
    output_price=0.015,    # per 1K tokens
    cache_hit_price=0.001, # per 1K tokens (cached)
)

# After run:
result = await agent.run("Hello")
print(result.metrics)  # Token counts and cost
# Cost is reflected in committed step/run facts
```
