# Channels 渠道接入层

Channels 包负责把 Agent 能力接入到第三方 IM 平台（如飞书）。它提供了一套通用的渠道抽象，让不同 IM 平台的接入逻辑可以复用相同的基础设施。

---

## 一句话概括

Channels 是 **Agent 的"翻译官"** —— 它把飞书、钉钉等 IM 平台的消息格式转换成 Agent 能理解的输入，再把 Agent 的输出发送回 IM 平台。

---

## 架构定位

```mermaid
flowchart TB
    subgraph IM["IM 平台"]
        Feishu["飞书"]
        Slack["Slack (未来)"]
        DingTalk["钉钉 (未来)"]
    end

    subgraph Channels["Channels 层"]
        Base["BaseChannelService<br/>通用流程抽象"]
        Session["Session 管理<br/>会话生命周期"]
        Executor["AgentExecutor<br/>执行器"]
        Pool["RuntimeAgentPool<br/>实例缓存"]
    end

    subgraph SDK["Agiwo SDK"]
        Agent["Agent"]
        Scheduler["Scheduler"]
    end

    Feishu -->|WebSocket 事件| Base
    Base -->|调用| Executor
    Executor -->|提交任务| Scheduler
    Scheduler -->|调度| Agent
    Pool -->|提供实例| Executor
    Session -->|管理状态| Base
```

---

## 核心流程

### 消息处理总流程

```mermaid
sequenceDiagram
    actor User as IM 用户
    participant IM as IM 服务器
    participant Channel as ChannelService
    participant Handler as InboundHandler
    participant Session as SessionManager
    participant Executor as AgentExecutor
    participant SDK as Agent/Scheduler

    User->>IM: 发送消息
    IM->>Channel: 推送事件
    Channel->>Handler: 解析 + 去重
    Handler->>Handler: 触发规则检查<br/>白名单/艾特/私聊

    alt 不触发
        Handler-->>IM: 忽略
    else 触发
        Handler->>Session: 入队消息
        Session->>Session: 防抖/批处理
        Session->>Executor: 批量执行

        Executor->>Session: 获取/创建会话
        Executor->>Executor: 获取 runtime agent
        Executor->>SDK: 提交到 Scheduler
        SDK-->>Executor: 流式输出
        Executor-->>Channel: 输出文本
        Channel-->>IM: 回复用户
    end
```

---

## 模块详解

### 1. BaseChannelService — 渠道抽象基类

**文件**: `base.py`

定义了所有渠道必须实现的通用流程。

```mermaid
classDiagram
    class BaseChannelService {
        <<abstract>>
        -_session_service: SessionContextService
        -_agent_pool: RuntimeAgentPool
        -_executor: AgentExecutor
        -_session_mgr: SessionManager
        +close_base()
        +session_manager
        +session_service
        +agent_pool
        +executor
        #_on_batch_ready()
        #_execute_batch()
        #_build_user_message()* 
        #_deliver_reply()*
        #_deliver_message()*
        #_to_user_facing_error()*
    }

    class FeishuChannelService {
        +initialize()
        +close()
        +get_status()
        #_build_user_message()
        #_deliver_reply()
        #_deliver_message()
        #_to_user_facing_error()
    }

    BaseChannelService <|-- FeishuChannelService
```

**抽象方法说明**:

| 方法 | 职责 | 飞书实现示例 |
|------|------|-------------|
| `_build_user_message` | 把渠道消息转成 `UserMessage` | `FeishuUserMessageBuilder` |
| `_deliver_reply` | 发送首条回复 | 回复消息或添加表情 |
| `_deliver_message` | 发送后续消息 | 创建新消息 |
| `_to_user_facing_error` | 错误本地化 | "上一条任务仍在处理中" |

### 2. Session 会话管理子包

**目录**: `session/`

管理 IM 会话的生命周期，处理多会话切换。

```mermaid
graph TB
    subgraph "概念层次"
        ChatContext["ChannelChatContext<br/>聊天上下文"]
        Session["Session<br/>会话实例"]
        Agent["runtime_agent_id<br/>运行时 Agent ID"]
        State["scheduler_state_id<br/>调度器状态 ID"]
    end

    ChatContext -->|包含多个| Session
    Session -->|绑定| Agent
    Session -->|关联| State
```

**核心模型**:

| 模型 | 含义 | 示例 |
|------|------|------|
| `ChannelChatContext` | 聊天上下文（群/私聊） | `feishu:main:p2p:ou_xxx` |
| `Session` | 一次连续对话 | UUID |
| `BatchContext` | 批量消息上下文 | 包含触发用户、消息 ID 等 |
| `InboundMessage` | 入站消息 | 解析后的飞书消息 |

**核心服务**:

| 模块 | 职责 |
|------|------|
| `SessionContextService` | 会话协调（获取/创建/切换） |
| `SessionManager` | 消息批处理 + 防抖 |
| `binding.py` | 会话操作的原子领域逻辑 |

### 3. AgentExecutor — Agent 执行器

**文件**: `agent_executor.py`

封装 Agent 执行的细节，对接 Scheduler。

**状态路由逻辑**:

```
当前状态 → 动作
─────────────────────────────
None/COMPLETED/FAILED    → submit (新建持久 Agent)
IDLE/FAILED(persistent)  → enqueue (复用现有 Agent)
RUNNING/WAITING/QUEUED   → steer (注入消息)
PENDING                  → wait → submit/enqueue
```

```mermaid
flowchart TD
    A[用户消息到来] --> B{查询 State 状态}

    B -->|None/Completed/Failed| C[submit_and_stream]
    B -->|Idle/Failed-persistent| D[enqueue_and_stream]
    B -->|Running/Waiting/Queued| E[steer]
    B -->|Pending| F[wait_for 状态变更]
    F --> G{新状态判断}
    G -->|Idle| D
    G -->|其他| C

    C --> H[创建新 State]
    D --> I[复用现有 State]
    E --> J[向执行上下文注入消息]
    H --> K[stream 输出]
    I --> K
    K --> L[返回文本给用户]
```

### 4. RuntimeAgentPool — Agent 运行时池

**文件**: `runtime_agent_pool.py`

缓存 Agent 实例，避免重复构建，支持配置热更新。

```mermaid
sequenceDiagram
    participant Executor as AgentExecutor
    participant Pool as RuntimeAgentPool
    participant Registry as AgentRegistry
    participant Cache as 内存缓存

    Executor->>Pool: get_or_create_runtime_agent(session)
    Pool->>Registry: 获取 base_agent 配置
    Pool->>Pool: 计算配置指纹 SHA1

    alt 缓存命中且指纹匹配
        Pool-->>Executor: 返回缓存的 Agent
    else 缓存命中但指纹不匹配
        Pool->>Pool: 检查 State 是否 RUNNING
        alt RUNNING
            Pool-->>Executor: 返回旧 Agent（延迟刷新）
        else 非 RUNNING
            Pool->>Pool: 关闭旧 Agent
            Pool->>Pool: 构建新 Agent
            Pool->>Cache: 更新缓存
            Pool-->>Executor: 返回新 Agent
        end
    else 缓存未命中
        Pool->>Pool: build_agent
        Pool->>Cache: 存入缓存
        Pool-->>Executor: 返回新 Agent
    end
```

**配置指纹计算**:
```python
# 包含字段：name, description, model_provider, model_name,
#          system_prompt, tools, options, model_params
# 算法：SHA1(sorted JSON)
```

---

## Session 管理详解

### 会话绑定流程

```mermaid
sequenceDiagram
    actor User as 用户
    participant IM as IM 平台
    participant Handler as InboundHandler
    participant Service as SessionContextService
    participant Store as ChannelStore
    participant Binding as binding.py

    User->>IM: 首次发送消息
    IM->>Handler: 消息事件
    Handler->>Service: get_or_create_current_session

    Service->>Store: get_chat_context(scope_id)
    Store-->>Service: None（首次）

    Service->>Binding: open_initial_session(...)
    Binding->>Binding: 创建 ChatContext + Session
    Binding-->>Service: SessionMutationPlan
    Service->>Store: apply_session_mutation
    Store-->>Service: 保存成功
    Service-->>Handler: SessionContextResolution
```

### 会话切换流程

```mermaid
sequenceDiagram
    actor User as 用户
    participant Handler as InboundHandler
    participant Service as SessionContextService
    participant Binding as binding.py
    participant Store as ChannelStore

    User->>Handler: 发送 /switch 命令
    Handler->>Service: switch_session(...)

    Service->>Store: 获取 ChatContext
    Service->>Store: 获取目标 Session
    Service->>Binding: switch_session(chat_ctx, prev, target)
    Binding->>Binding: 更新 chat_ctx.current_session_id
    Binding-->>Service: SessionMutationPlan
    Service->>Store: apply_session_mutation
    Service-->>Handler: SessionSwitchResult
    Handler-->>User: 切换成功通知
```

---

## 消息批处理机制

SessionManager 实现了防抖 + 最大等待窗口的消息批处理：

```mermaid
graph TB
    A[消息 A 到达] -->|first_pending_at = t1| B[启动定时器]
    C[消息 B 到达] -->|重置定时器| B
    D[消息 C 到达] -->|重置定时器| B

    B -->|debounce_ms 内无新消息| E[触发 _on_batch_ready]
    B -->|max_batch_window_ms 到达| E

    E --> F[批量执行]
    F -->|执行完成| G[running = False]
    G -->|pending_messages 不为空| B
```

**参数配置** (config.py):
- `feishu_debounce_ms`: 防抖等待时间 (默认 3000ms)
- `feishu_max_batch_window_ms`: 最大批处理窗口 (默认 15000ms)

---

## 接口定义

### ChannelChatSessionStore (Protocol)

```python
class ChannelChatSessionStore(Protocol):
    async def get_chat_context(self, scope_id: str) -> ChannelChatContext | None: ...
    async def upsert_chat_context(self, chat_context: ChannelChatContext) -> None: ...
    async def get_session(self, session_id: str) -> Session | None: ...
    async def upsert_session(self, session: Session) -> None: ...
    async def apply_session_mutation(self, mutation: SessionMutationPlan) -> None: ...
    async def list_sessions_by_user(self, user_open_id: str) -> list[SessionWithContext]: ...
```

### 存储实现

| 实现 | 文件 | 适用场景 |
|------|------|----------|
| `InMemoryFeishuChannelStore` | `feishu/store/memory.py` | 开发/测试 |
| `SqliteFeishuChannelStore` | `feishu/store/sqlite.py` | 生产环境 |

---

## 扩展指南

### 接入新的 IM 平台 (如 Slack)

1. **创建渠道目录** `channels/slack/`
2. **实现消息解析**
   ```python
   # slack/message_parser.py
   class SlackInboundEnvelope: ...
   class SlackMessageParser: ...
   ```
3. **实现内容提取**
   ```python
   # slack/content_extractor.py
   class SlackContentExtractor: ...
   ```
4. **实现发送服务**
   ```python
   # slack/delivery_service.py
   class SlackDeliveryService: ...
   ```
5. **实现渠道服务**
   ```python
   # slack/service.py
   class SlackChannelService(BaseChannelService): ...
   ```
6. **在 app.py 中初始化和启动**

### 关键实现点

- **连接方式**: WebSocket 实时推送 或 Webhook
- **去重机制**: event_id 幂等检查
- **触发规则**: 群聊需@ / 私聊直接触发 / 白名单过滤
- **会话标识**: 私聊用用户ID，群聊用群ID+用户ID
- **消息格式**: 文本、富文本、图片、文件的处理

---

## 与飞书渠道的交互

详细实现见 [`feishu/README.md`](./feishu/README.md)。
