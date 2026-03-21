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
from agiwo.agent.storage import create_run_step_storage

# SQLite backend (default)
storage = create_run_step_storage("sqlite", db_path="runs.db")

# Memory backend (for testing)
storage = create_run_step_storage("memory")

# MongoDB backend
storage = create_run_step_storage("mongo", connection_string="mongodb://...")
```

Pass it to the Agent:

```python
from agiwo.agent.options import AgentOptions

agent = Agent(
    config,
    model=model,
    options=AgentOptions(run_step_storage=my_storage),
)
```

### What gets stored

- **Run records**: input, output, timestamps, token usage, status
- **Step records**: LLM messages, tool calls, tool results, intermediate reasoning
- **Cost tracking**: per-step and per-run token counts and estimated costs

## Session Storage

Manages session-level metadata:

```python
from agiwo.agent.storage import create_session_storage

storage = create_session_storage("sqlite", db_path="sessions.db")
```

Sessions track:
- Conversation history metadata
- Compaction state (when context gets too long)
- User identity and preferences

## Trace Storage

Collects distributed traces for debugging and monitoring:

```python
from agiwo.observability import create_trace_storage

# SQLite
trace_storage = create_trace_storage("sqlite", db_path="traces.db")

# Memory
trace_storage = create_trace_storage("memory")

# MongoDB
trace_storage = create_trace_storage("mongo", connection_string="mongodb://...")
```

Pass it to the Agent:

```python
agent = Agent(
    config,
    model=model,
    options=AgentOptions(trace_storage=my_trace_storage),
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

All storage backends support these common options:

```python
# SQLite
storage = create_run_step_storage(
    "sqlite",
    db_path="./data/runs.db",
)

# MongoDB
storage = create_run_step_storage(
    "mongo",
    connection_string="mongodb://localhost:27017",
    database="agiwo",
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
