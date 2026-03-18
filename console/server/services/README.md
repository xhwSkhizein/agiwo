# Services 业务服务层

Console Server 的核心业务逻辑层，封装 Agent 生命周期管理、存储配置、对话 SSE 和指标聚合。

---

## 一句话概括

Services 是 **Agent 的"后勤部门"** —— 负责组装 Agent、配置存储、处理实时对话流，以及收集运行指标。

---

## 架构定位

```mermaid
flowchart TB
    subgraph Routers["Routers 层"]
        Agents["/api/agents"]
        Chat["/api/chat"]
        Scheduler["/api/scheduler"]
    end

    subgraph Services["Services 层"]
        Lifecycle["| `agent_lifecycle.py` | Agent 构建、重新水化、持久 Agent 恢复 |
| `agent_registry/` | Agent 配置的持久化 CRUD |
| `storage_wiring.py` | 存储配置的构建器 |
| `chat_sse.py` | Chat SSE helpers and strategies |
| `metrics.py` | Run metrics aggregation |"]
        Registry["agent_registry/<br/>配置注册表"]
        Storage["storage_wiring.py<br/>存储配置"]
        ChatSSE["chat_sse.py<br/>对话 SSE"]
        Metrics["metrics.py<br/>指标聚合"]
    end

    subgraph SDK["Agiwo SDK"]
        Agent["Agent"]
        Model["Model"]
        Tools["Tools"]
    end

    Agents --> Registry
    Chat --> ChatSSE
    Chat --> Lifecycle
    Scheduler --> Lifecycle

    Lifecycle -->|build| Agent
    Lifecycle -->|create| Model
    Registry -->|提供配置| Lifecycle
    Storage -->|提供配置| Lifecycle

    ChatSSE -->|流式输出| Agent
    Metrics -->|查询| SDK
```

---

## 核心流程

### Agent 构建流程

```mermaid
sequenceDiagram
    actor Router as API Router
    participant Lifecycle as agent_lifecycle
    participant Registry as AgentRegistry
    participant Tools as tools.py
    participant SDK as Agiwo SDK

    Router->>Lifecycle: build_agent(config, console_config, registry)

    Lifecycle->>Lifecycle: build_model(config)
    Lifecycle->>SDK: create_model_from_dict(...)
    SDK-->>Lifecycle: Model 实例

    Lifecycle->>Lifecycle: build_agent_options(config, console_config)
    Lifecycle->>Lifecycle: 注入 storage_config

    Lifecycle->>Tools: build_tools(tool_refs, ...)
    Tools->>Tools: 解析工具引用
    loop 每个 Agent 工具引用
        Tools->>Registry: get_agent(agent_id)
        Tools->>Lifecycle: build_agent(子配置)
        Lifecycle-->>Tools: 子 Agent
        Tools->>Tools: build_agent_tool
    end
    Tools-->>Lifecycle: list[RuntimeToolLike]

    Lifecycle->>SDK: Agent(agent_config, model, tools, id)
    SDK-->>Lifecycle: Agent 实例
    Lifecycle-->>Router: Agent
```

### Scheduler Chat 对话流程

```mermaid
sequenceDiagram
    actor Client as 前端客户端
    participant Router as scheduler.router
    participant ChatSSE as chat_sse.py
    participant Lifecycle as agent_lifecycle
    participant Scheduler as Scheduler
    participant SDK as Agent

    Client->>Router: POST /api/scheduler/chat/{id}
    Router->>ChatSSE: create_conversation_response(...)

    ChatSSE->>Lifecycle: prepare_conversation(...)
    Lifecycle->>Registry: get_agent(agent_id)
    Lifecycle->>Lifecycle: build_agent(...)
    Lifecycle-->>ChatSSE: PreparedConversation

    ChatSSE->>ChatSSE: event_generator()
    ChatSSE->>Scheduler: scheduler.stream(...)
    Scheduler->>SDK: 调度执行
    SDK-->>Scheduler: 流式事件
    Scheduler-->>ChatSSE: AgentStreamItem
    ChatSSE-->>Client: SSE 数据
```

---

## 模块详解

### 1. agent_lifecycle.py — Agent 生命周期管理

**职责**: 构建、重新水化、恢复 Agent 实例

#### 核心函数

| 函数 | 职责 | 调用场景 |
|------|------|----------|
| `build_agent` | 从配置构建 Agent | 新建对话、创建子 Agent |
| `build_model` | 构建 LLM Model | build_agent 内部使用 |
| `build_agent_options` | 构建 AgentOptions | build_agent 内部使用 |
| `rehydrate_agent` | 从 State 恢复 Agent | 服务重启后恢复持久 Agent |
| `resume_persistent_agent` | 恢复并继续执行 | Scheduler Resume API |

#### Agent 构建细节

```mermaid
flowchart TD
    A[AgentConfigRecord] --> B{工具引用}

    B -->|内置工具| C[直接从 BUILTIN_TOOLS 获取]
    B -->|Agent 工具| D[递归 build_agent]

    C --> E[组装工具列表]
    D -->|循环引用检查| F{_building 集合检查}
    F -->|检测到循环| G[抛出 ValueError]
    F -->|无循环| D
    D --> E

    E --> H[创建 AgentConfig]
    H --> I[实例化 Agent]
    I --> J[返回 Agent]
```

**循环引用检测**:
```python
if config.id in _building:
    raise ValueError(f"Circular agent reference detected: {config.id}")
_building.add(config.id)
# 递归构建子 Agent 时传递 _building.copy()
```

#### 默认配置

```python
def build_default_agent_options() -> dict[str, Any]:
    """Return the canonical default agent options payload."""
    return AgentOptionsInput.model_validate({}).model_dump(exclude_none=True)
```

### 2. agent_registry/ — Agent 配置注册表

**目录**: `agent_registry/`

Agent 配置的持久化 CRUD，支持 SQLite 和 MongoDB。

#### 架构

```mermaid
graph TB
    subgraph API["API 层"]
        Router["agents.router"]
    end

    subgraph Service["服务层"]
        Registry["AgentRegistry"]
    end

    subgraph Store["存储层"]
        Protocol["AgentRegistryStore<br/>(Protocol)"]
        Memory["InMemoryAgentRegistryStore"]
        SQLite["SqliteAgentRegistryStore"]
        Mongo["MongoAgentRegistryStore"]
    end

    Router --> Registry
    Registry --> Protocol
    Memory -.->|实现| Protocol
    SQLite -.->|实现| Protocol
    Mongo -.->|实现| Protocol
```

#### AgentRegistry API

| 方法 | 职责 |
|------|------|
| `list_agents` | 列出所有 Agent 配置 |
| `get_agent` | 根据 ID 获取配置 |
| `get_agent_by_name` | 根据名称获取配置 |
| `create_agent` | 创建新配置（验证 + 保存） |
| `replace_agent` | 全量替换配置 |
| `delete_agent` | 删除配置 |

#### AgentConfigRecord 模型

```python
class AgentConfigRecord(BaseModel):
    id: str                    # UUID
    name: str                  # 显示名称
    description: str           # 描述
    model_provider: str        # openai-compatible / anthropic / ...
    model_name: str            # 模型名称
    system_prompt: str         # 系统提示词
    tools: list[str]           # 工具引用列表
    options: dict              # AgentOptions
    model_params: dict         # LLM 参数 (temperature 等)
    created_at: datetime
    updated_at: datetime
```

### 3. storage_wiring.py — 存储配置

**职责**: 构建存储配置

#### 配置构建器

| 函数 | 返回 | 说明 |
|------|------|------|
| `build_run_step_storage_config` | `RunStepStorageConfig` | Run/Step 存储配置 |
| `build_trace_storage_config` | `TraceStorageConfig` | Trace 存储配置 |
| `build_agent_state_storage_config` | `AgentStateStorageConfig` | Scheduler 状态存储 |
| `build_citation_store_config` | `CitationStoreConfig` | 引用存储配置 |


### 4. chat_sse.py — 对话 SSE 工具

**职责**: 封装 Chat 对话的 SSE 流式响应

#### 核心函数

| 函数 | 职责 |
|------|------|
| `create_conversation_response` | 创建 SSE 响应 |
| `prepare_conversation` | 准备 Agent 和会话 |
| `stream_event_message` | 序列化流式事件 |
| `stream_scheduler_events` | Scheduler 对话策略 |

#### 对话策略

```mermaid
flowchart LR
    A[用户请求] --> B{选择策略}

    B -->|直接对话| C[_stream_chat_events<br/>直接调用 Agent.start]
    B -->|Scheduler 对话| D[stream_scheduler_events<br/>通过 Scheduler.stream]

    C --> E[返回 EventSourceResponse]
    D --> E
```

#### 错误处理

```python
async def event_generator() -> AsyncIterator[SseMessage]:
    try:
        async for sse_msg in strategy(...):
            yield sse_msg
    except Exception as exc:
        yield error_event_message(exc)  # 异常封装为 SSE 事件
    finally:
        await prepared.agent.close()    # 确保资源释放
```

### 5. metrics.py — 指标聚合

**职责**: 聚合 Run/Session/State 的运行指标

#### 核心函数

| 函数 | 职责 |
|------|------|
| `collect_session_aggregates` | 按 Session 聚合 Run 指标 |
| `summarize_runs_paginated` | 分页统计 Run 指标 |
| `build_metrics_by_state` | 为 Scheduler State 构建指标 |

#### 指标聚合流程

```mermaid
sequenceDiagram
    participant Router as Router
    participant Metrics as metrics.py
    participant Storage as RunStepStorage

    Router->>Metrics: collect_session_aggregates(storage)

    loop 分页查询 (每页 500)
        Metrics->>Storage: list_runs(limit=500, offset=N)
        Storage-->>Metrics: list[Run]
        Metrics->>Metrics: _merge_run_into_session
    end

    Metrics->>Metrics: 按更新时间排序
    Metrics-->>Router: list[SessionAggregate]
```

#### RunMetricsSummary 字段

| 字段 | 说明 |
|------|------|
| `run_count` | Run 总数 |
| `completed_run_count` | 完成数 |
| `step_count` | Step 总数 |
| `tool_calls_count` | 工具调用次数 |
| `input_tokens` | 输入 Token 数 |
| `output_tokens` | 输出 Token 数 |
| `total_tokens` | 总 Token 数 |
| `cache_read_tokens` | 缓存读取 Token |
| `cache_creation_tokens` | 缓存创建 Token |
| `duration_ms` | 总耗时 |
| `token_cost` | Token 成本 |

---

## 数据流详解

### Agent 配置 CRUD 数据流

```mermaid
sequenceDiagram
    actor Client as 前端/客户端
    participant Router as agents.router
    participant Registry as AgentRegistry
    participant Models as AgentConfigRecord
    participant Store as AgentRegistryStore

    Client->>Router: POST /api/agents (创建)
    Router->>Models: AgentConfigPayload
    Router->>Router: _body_to_record
    Router->>Registry: create_agent(record)
    Registry->>Registry: _validate_agent_config_record
    Registry->>Store: upsert_agent(normalized)
    Store-->>Registry: 保存完成
    Registry-->>Router: 返回带 ID 的 record
    Router->>Router: _record_to_response
    Router-->>Client: AgentConfigResponse (201)

    Client->>Router: GET /api/agents (列表)
    Router->>Registry: list_agents()
    Registry->>Store: list_agents()
    Store-->>Registry: list[AgentConfigRecord]
    Registry-->>Router: records
    Router->>Router: [_record_to_response(r) for r in records]
    Router-->>Client: list[AgentConfigResponse]
```

### 指标查询数据流

```mermaid
sequenceDiagram
    actor Client as 前端
    participant Router as scheduler.router
    participant Metrics as metrics.py
    participant Storage as RunStepStorage

    Client->>Router: GET /api/scheduler/states
    Router->>Router: list_all states
    Router->>Metrics: build_metrics_by_state(states, run_storage)

    Metrics->>Metrics: 按 session_id 分组 agent_ids

    loop 每个 session_id
        Metrics->>Storage: list_runs(session_id=xxx)
        Storage-->>Metrics: list[Run]
        loop 每个 Run
            Metrics->>Metrics: 匹配 agent_id
            Metrics->>Metrics: add_run_to_summary
        end
    end

    Metrics-->>Router: dict[(session_id, agent_id), RunMetricsSummary]
    Router->>Router: 组装到响应
    Router-->>Client: list[AgentStateListItem]
```

---

## 接口定义

### AgentRegistryStore (Protocol)

```python
class AgentRegistryStore(Protocol):
    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def list_agents(self, limit: int, offset: int) -> list[AgentConfigRecord]: ...
    async def get_agent(self, agent_id: str) -> AgentConfigRecord | None: ...
    async def get_agent_by_name(self, name: str) -> AgentConfigRecord | None: ...
    async def upsert_agent(self, record: AgentConfigRecord) -> None: ...
    async def delete_agent(self, agent_id: str) -> bool: ...
```

### RunStoragePort (Protocol)

```python
class RunStoragePort(Protocol):
    async def list_runs(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Run]: ...
```

---

## 异常体系

### agent_lifecycle 异常

```python
class PersistentAgentResumeError(RuntimeError):
    """Base error for persistent-agent resume failures."""

class PersistentAgentNotFoundError(PersistentAgentResumeError):
    """Raised when the target scheduler state does not exist."""

class PersistentAgentValidationError(PersistentAgentResumeError):
    """Raised when the target state cannot be resumed."""
```

---

## 配置说明

Services 层依赖 ConsoleConfig 的配置项：

| 配置项 | 用途 | 影响模块 |
|--------|------|----------|
| `run_step_storage_type` | Run/Step 存储后端 | storage_wiring |
| `trace_storage_type` | Trace 存储后端 | storage_wiring |
| `metadata_storage_type` | 元数据存储后端 | storage_wiring, agent_registry |
| `sqlite_db_path` | SQLite 数据库路径 | 所有存储配置 |
| `mongodb_uri` | MongoDB 连接串 | 所有存储配置 |
| `default_agent_*` | 默认 Agent 配置 | agent_lifecycle (fallback) |
