# Console Server

Agiwo Console 的控制平面后端，基于 FastAPI 构建，为 Agent SDK 提供可视化管理界面和第三方渠道接入能力。

## 一句话概括

Console Server 是 **Agent 的控制中枢** —— 它把底层的 Agent SDK 包装成 REST API，让前端可以管理 Agent 配置、查看执行记录，同时通过渠道系统把 Agent 接入飞书等 IM 平台。

---

## 整体架构

```mermaid
flowchart TB
    subgraph Frontend["前端 (Next.js)"]
        UI["管理界面"]
        ChatUI["对话界面"]
    end

    subgraph Console["Console Server (FastAPI)"]
        Routers["API 路由层<br/>routers/"]
        Services["业务服务层<br/>services/"]
        Channels["渠道接入层<br/>channels/"]
        Domain["领域模型<br/>domain/"]
    end

    subgraph SDK["Agiwo SDK"]
        Agent["Agent"]
        Scheduler["Scheduler"]
        Storage["Storage"]
    end

    subgraph External["外部系统"]
        Feishu["飞书 IM"]
        LLM["LLM Provider"]
    end

    UI -->|HTTP| Routers
    ChatUI -->|SSE| Routers
    Routers --> Services
    Services -->|调用| SDK
    Channels -->|封装| SDK
    Channels <-->|WebSocket| Feishu
    SDK <-->|API| LLM
```

---

## 核心主流程

### 1. Agent 配置管理流程

```mermaid
sequenceDiagram
    actor User as 用户
    participant API as /api/agents
    participant Registry as AgentRegistry
    participant Store as RegistryStore
    participant SDK as Agent SDK

    User->>API: 创建 Agent 配置
    API->>Registry: create_agent(record)
    Registry->>Registry: 验证配置合法性
    Registry->>Store: upsert_agent(record)
    Store-->>Registry: 保存成功
    Registry-->>API: 返回 AgentConfigRecord
    API-->>User: 201 Created

    User->>API: 启动对话 (POST /api/chat/{id})
    API->>SDK: build_agent(config)
    SDK-->>API: Agent 实例
    API->>SDK: agent.start(message)
    SDK-->>API: SSE 流式响应
    API-->>User: 实时返回 LLM 输出
```

### 2. Scheduler 调度流程

```mermaid
sequenceDiagram
    actor User as 用户
    participant API as /api/scheduler
    participant Scheduler as Scheduler
    participant Engine as SchedulerEngine
    participant Store as StateStorage
    participant SDK as Agent SDK

    User->>API: 提交任务 (POST /states/create)
    API->>Scheduler: submit(agent, task)
    Scheduler->>Engine: 调度执行
    Engine->>Store: 创建 State 记录
    Engine->>SDK: agent.start(task)
    SDK-->>Engine: 流式事件
    Engine-->>Scheduler: 状态更新
    Scheduler-->>API: state_id
    API-->>User: {state_id, ok: true}

    User->>API: 查询状态 (GET /states/{id})
    API->>Scheduler: get_state(id)
    Scheduler->>Store: 查询最新状态
    Store-->>Scheduler: AgentState
    Scheduler-->>API: 状态详情
    API-->>User: AgentStateResponse

    User->>API: 发送指令 (POST /states/{id}/steer)
    API->>Scheduler: steer(id, message)
    Scheduler->>Engine: 注入消息
    Engine->>SDK: 当前执行上下文接收消息
    SDK-->>Engine: 继续执行
    Engine-->>Scheduler: steer 成功
    Scheduler-->>API: {ok: true}
```

### 3. 飞书渠道消息处理流程

```mermaid
sequenceDiagram
    actor User as 飞书用户
    participant Feishu as 飞书服务器
    participant WS as WebSocket 长连接<br/>connection.py
    participant Handler as 消息处理器<br/>inbound_handler.py
    participant Parser as 消息解析器<br/>message_parser.py
    participant Session as 会话管理<br/>session/
    participant Agent as Agent 执行器<br/>agent_executor.py

    User->>Feishu: 发送消息 @机器人
    Feishu->>WS: 推送事件
    WS->>WS: 解密 + 验证
    WS->>Handler: 转发消息信封
    Handler->>Parser: 解析消息内容
    Parser->>Parser: 提取文本/附件/艾特信息
    Parser-->>Handler: InboundMessage

    Handler->>Handler: 去重检查
    Handler->>Handler: 触发规则判断<br/>(白名单/群聊需@/私聊)

    Handler->>Session: 获取或创建会话
    Session->>Session: ChatContext + Session 绑定
    Session-->>Handler: Session 上下文

    Handler->>Handler: 命令拦截检查
    alt 是命令
        Handler->>Handler: 执行命令
        Handler-->>Feishu: 回复命令结果
    else 普通消息
        Handler->>Session: 消息入队
        Session->>Session: 防抖 + 批处理
        Session->>Agent: 触发 Agent 执行
        Agent->>Agent: 提交到 Scheduler
        Agent-->>Handler: 流式输出
        Handler-->>Feishu: 回复 Agent 结果
        Feishu-->>User: 显示回复
    end
```

---

## 目录结构详解

```
console/server/
├── app.py                    # FastAPI 入口 + lifespan 管理
├── config.py                 # 配置定义 (ConsoleConfig)
├── dependencies.py           # 依赖注入容器
├── schemas.py                # API 请求/响应 Pydantic 模型
├── response_serialization.py # 响应序列化工具
├── tools.py                  # 工具目录 + 工具组装
├── routers/                  # API 路由层
│   ├── sessions.py           # Session/Run/Step 查询
│   ├── agents.py             # Agent CRUD
│   ├── chat.py               # 直接对话 SSE
│   ├── scheduler.py          # Scheduler 状态/控制
│   ├── traces.py             # Trace 查询 + SSE
│   └── feishu.py             # 飞书渠道状态
├── services/                 # 业务服务层
│   ├── agent_lifecycle.py    # Agent 构建/恢复/复用
│   ├── agent_registry/       # Agent 配置持久化
│   ├── storage_wiring.py     # 存储配置构建
│   ├── chat_sse.py           # Chat SSE 工具
│   └── metrics.py            # 指标聚合
├── channels/                 # 渠道接入层
│   ├── base.py               # 渠道抽象基类
│   ├── agent_executor.py     # Agent 执行器
│   ├── runtime_agent_pool.py # Agent 运行时池
│   ├── session/              # 会话管理子包
│   └── feishu/               # 飞书渠道实现
└── domain/                   # 领域模型
    ├── agent_configs.py
    ├── run_metrics.py
    ├── sessions.py
    └── tool_references.py
```

---

## 三大子系统

### 1. API 层 (`routers/`)

REST API + SSE 端点，供前端调用。

| 路由 | 职责 |
|------|------|
| `/api/agents` | Agent 配置的 CRUD |
| `/api/chat/{id}` | 直接对话 SSE |
| `/api/scheduler/*` | Scheduler 状态查询、控制、调度对话 |
| `/api/sessions/*` | Session/Run/Step 查询 |
| `/api/traces/*` | Trace 查询 + 实时 SSE |
| `/api/channels/feishu/*` | 飞书渠道状态 |

### 2. 服务层 (`services/`)

核心业务逻辑，封装 SDK 调用。

| 模块 | 职责 |
|------|------|
| `agent_lifecycle.py` | Agent 构建、重新水化、持久 Agent 恢复 |
| `agent_registry/` | Agent 配置的持久化 CRUD |
| `storage_wiring.py` | 存储配置的构建器 |
| `chat_sse.py` | Chat 对话的 SSE 流式响应封装 |
| `metrics.py` | Run/Session/State 的指标聚合 |

### 3. 渠道层 (`channels/`)

第三方 IM 接入，当前只有飞书，设计为可扩展。

| 模块 | 职责 |
|------|------|
| `base.py` | 渠道抽象基类，定义通用流程 |
| `agent_executor.py` | 封装 Agent 执行，对接 Scheduler |
| `runtime_agent_pool.py` | Agent 实例缓存 + 配置指纹刷新 |
| `session/` | 会话生命周期管理 |
| `feishu/` | 飞书渠道具体实现 |

---

## 关键概念图解

### Session 绑定模型

```mermaid
graph LR
    subgraph "飞书消息上下文"
        A[用户A] -->|私聊| P2P[p2p:用户A-ID]
        A -->|群聊at机器人| GROUP[group:群ID + user:用户A]
    end

    subgraph "ChannelChatContext"
        C1[ChatContext P2P]
        C2[ChatContext Group]
    end

    subgraph "Session"
        S1[Session A-1]
        S2[Session A-2]
        S3[Session B-1]
    end

    P2P --> C1
    GROUP --> C2
    C1 --> S1
    C1 -.->|切换会话| S2
    C2 --> S3
```

### Agent 运行时关系

```mermaid
graph TB
    subgraph "配置层"
        CONFIG[AgentConfigRecord<br/>存在 Registry]
    end

    subgraph "运行时层"
        AGENT[Agent 实例<br/>agent.id = 随机ID]
        STATE[Scheduler State<br/>state.id = agent.id]
    end

    subgraph "存储层"
        RUN[Run 记录]
        STEP[Step 记录]
        TRACE[Trace 记录]
    end

    CONFIG -->|build_agent| AGENT
    AGENT -->|submit| STATE
    AGENT -->|产生| RUN
    AGENT -->|产生| STEP
    AGENT -->|产生| TRACE
```

---

## 数据流总览

```mermaid
flowchart LR
    subgraph 输入
        HTTP["HTTP 请求"]
        WS["飞书 WebSocket"]
    end

    subgraph 路由层
        R1["agents.router"]
        R2["chat.router"]
        R3["scheduler.router"]
        R4["feishu.router"]
    end

    subgraph 服务层
        S1["AgentRegistry"]
        S2["build_agent"]
        S3["Scheduler"]
        S4["FeishuChannelService"]
    end

    subgraph SDK
        SDK1["Agent"]
        SDK2["Model"]
        SDK3["Tools"]
    end

    HTTP --> R1
    HTTP --> R2
    HTTP --> R3
    WS --> R4

    R1 --> S1
    R2 --> S2
    R3 --> S3
    R4 --> S4

    S2 --> SDK1
    S4 --> S3
    S3 --> SDK1
    SDK1 --> SDK2
    SDK1 --> SDK3
```

---

## 子包文档

- [`channels/`](./channels/README.md) — 渠道接入层详解
- [`channels/feishu/`](./channels/feishu/README.md) — 飞书渠道实现
- [`services/`](./services/README.md) — 业务服务层详解

---

## 启动流程

```python
# app.py 的 lifespan

1. 加载 ConsoleConfig
2. 创建 run_step_storage (SQLite/Mongo/Memory)
3. 创建 trace_storage (BaseTraceStorage)
4. 初始化 AgentRegistry
5. 创建 Scheduler (使用 state_storage)
6. 如启用飞书:
   - 获取/创建默认 Agent 配置
   - 创建 FeishuChannelService
   - 启动 WebSocket 长连接
7. 组装 ConsoleRuntime 绑定到 app.state
8. 启动完成，开始服务请求
```

---

## 开发指南

### 添加新的 API 端点

1. 在 `schemas.py` 定义请求/响应模型
2. 在 `routers/` 创建或修改路由文件
3. 使用 `ConsoleRuntimeDep` 获取依赖
4. 在 `app.py` 的 `create_app()` 中注册路由

### 添加新的渠道

1. 继承 `BaseChannelService`
2. 实现抽象方法 (`_build_user_message`, `_deliver_reply`, 等)
3. 实现渠道特定的消息解析、存储
4. 在 `app.py` lifespan 中初始化和启动

### 存储后端切换

通过环境变量控制：
- `AGIWO_CONSOLE_RUN_STEP_STORAGE_TYPE=sqlite|mongodb|memory`
- `AGIWO_CONSOLE_TRACE_STORAGE_TYPE=sqlite|mongodb|memory`
- `AGIWO_CONSOLE_METADATA_STORAGE_TYPE=sqlite|mongodb|memory`
