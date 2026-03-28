# Storage & Observability

Agiwo separates storage into three independent concerns: run/step persistence, session metadata, and trace collection.

## Storage Layers

| Layer | Purpose | Interface |
|-------|---------|-----------|
| Run/Step Storage | Persist agent run and step records | `RunStepStorage` |
| Session Storage | Session metadata and compaction state | `SessionStorage` |
| Trace Storage | Distributed traces for observability | `BaseTraceStorage` |

## Run/Step Storage

Records every agent execution with full step details:

```python
from agiwo.agent import Agent, AgentConfig, AgentOptions, RunStepStorageConfig
from agiwo.agent.storage.factory import create_run_step_storage

# SQLite backend (default)
storage = create_run_step_storage(
    RunStepStorageConfig(storage_type="sqlite", config={"db_path": "runs.db"})
)

# Memory backend (for testing)
storage = create_run_step_storage(RunStepStorageConfig(storage_type="memory"))

# MongoDB backend
storage = create_run_step_storage(
    RunStepStorageConfig(
        storage_type="mongodb",
        config={"mongo_uri": "mongodb://...", "db_name": "agiwo"},
    )
)
```

Pass storage configuration through `AgentConfig.options`:

```python
agent = Agent(
    AgentConfig(
        name="assistant",
        description="...",
        system_prompt="...",
        options=AgentOptions(
            run_step_storage=RunStepStorageConfig(
                storage_type="sqlite",
                config={"db_path": "runs.db"},
            )
        ),
    ),
    model=model,
)
```

### What gets stored

- **Run records**: input, output, timestamps, token usage, status
- **Step records**: LLM messages, tool calls, tool results, intermediate reasoning
- **Cost tracking**: per-step and per-run token counts and estimated costs

## Session Storage

Manages session-level metadata:

```python
from agiwo.agent import RunStepStorageConfig
from agiwo.agent.storage.factory import create_session_storage

storage = create_session_storage(
    RunStepStorageConfig(storage_type="sqlite", config={"db_path": "sessions.db"})
)
```

Sessions track:
- Conversation history metadata
- Compaction state (when context gets too long)
- User identity and preferences

## Trace Storage

Collects distributed traces for debugging and monitoring:

```python
from agiwo.agent import Agent, AgentConfig, AgentOptions, TraceStorageConfig
from agiwo.observability import create_trace_storage

# SQLite
trace_storage = create_trace_storage(
    TraceStorageConfig(storage_type="sqlite", config={"db_path": "traces.db"})
)

# Memory
trace_storage = create_trace_storage(TraceStorageConfig(storage_type="memory"))

# MongoDB
trace_storage = create_trace_storage(
    TraceStorageConfig(
        storage_type="mongodb",
        config={"mongo_uri": "mongodb://...", "db_name": "agiwo"},
    )
)
```

Pass trace configuration through `AgentConfig.options`:

```python
agent = Agent(
    AgentConfig(
        name="assistant",
        description="...",
        system_prompt="...",
        options=AgentOptions(
            trace_storage=TraceStorageConfig(
                storage_type="sqlite",
                config={"db_path": "traces.db"},
            )
        ),
    ),
    model=model,
)
```

### Trace Structure

```
Trace (one per run)
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

# List recent traces
traces = await trace_storage.list_traces(limit=50)

# Subscribe to live traces (for SSE/streaming)
async for trace_update in trace_storage.subscribe():
    process(trace_update)
```

The `subscribe()` method enables real-time trace streaming to the Console UI.

## Choosing Backends

| Backend | Use Case | Persistence |
|---------|----------|-------------|
| `memory` | Testing, ephemeral workloads | No (lost on restart) |
| `sqlite` | Single-node deployments, development | Yes |
| `mongo` | Production, multi-node, high volume | Yes |

## Configuration

Storage constructors are config-driven:

```python
run_step_storage = create_run_step_storage(
    RunStepStorageConfig(
        storage_type="sqlite",
        config={"db_path": "./data/runs.db"},
    )
)

trace_storage = create_trace_storage(
    TraceStorageConfig(
        storage_type="mongodb",
        config={
            "mongo_uri": "mongodb://localhost:27017",
            "db_name": "agiwo",
        },
    )
)
```

## Cost Tracking

When `input_price`, `output_price`, and `cache_hit_price` are set on the Model, the SDK automatically tracks costs:

```python
model = OpenAIModel(
    id="gpt-4o",
    name="gpt-4o",
    input_price=0.005,     # per 1K tokens
    output_price=0.015,    # per 1K tokens
    cache_hit_price=0.001, # per 1K tokens (cached)
)

# After run:
result = await agent.run("Hello")
print(result.metrics)  # Token counts and cost
# Cost is recorded in run/step storage
```
