# 飞书 Channel 核心组件详解

## 目录

1. [FeishuChannelService](#1-feishuchannelservice)
2. [FeishuApiClient](#2-feishuapiclient)
3. [FeishuChannelStore](#3-feishuchannelstore)
4. [数据模型](#4-数据模型)

---

## 1. FeishuChannelService

文件: `console/server/channels/feishu/service.py` (937 行)

### 1.1 核心职责

- 消息接收与解析（Webhook / 长连接双模式）
- 会话状态管理（Session State Management）
- 消息批处理（Batching + Debounce）
- 与 Scheduler 的集成（Persistent Agent 生命周期）
- 响应格式化与发送

### 1.2 构造函数

```python
def __init__(
    self,
    *,
    config: ConsoleConfig,
    scheduler: Scheduler,
    agent_registry: AgentRegistry,
) -> None:
```

**初始化逻辑：**

```python
# 1. 配置提取
self._channel_instance_id = config.feishu_channel_instance_id
self._default_agent_id = config.feishu_default_agent_id
self._bot_open_id = config.feishu_bot_open_id
self._verification_token = config.feishu_verification_token
self._whitelist_open_ids = set(config.feishu_whitelist_open_ids)

# 2. 批处理参数
self._debounce_ms = config.feishu_debounce_ms
self._max_batch_window_ms = config.feishu_max_batch_window_ms
self._scheduler_wait_timeout = config.feishu_scheduler_wait_timeout

# 3. 依赖组件
self._api = FeishuApiClient(...)      # API 客户端
self._store = FeishuChannelStore(...)   # 存储层

# 4. 运行时状态
self._session_states: dict[str, _SessionState] = {}
self._session_locks: dict[str, asyncio.Lock] = {}
self._runtime_agents: dict[str, Agent] = {}

# 5. 长连接相关（仅长连接模式）
self._main_loop: asyncio.AbstractEventLoop | None = None
self._ws_thread: threading.Thread | None = None
self._ws_client: Any = None
```

### 1.3 Session 状态管理

```python
@dataclass
class _SessionState:
    pending_messages: list[FeishuInboundMessage] = field(default_factory=list)
    first_pending_at_ms: int | None = None    # 首批消息时间
    flush_task: asyncio.Task | None = None   # 当前定时任务
    running: bool = False                     # 是否正在处理
    latest_context: FeishuBatchContext | None = None
```

**Session Key 生成规则：**

```python
def _build_session_key(self, inbound: FeishuInboundMessage) -> str:
    if inbound.chat_type == "p2p":
        # 单聊：按用户隔离
        return f"feishu:{self._channel_instance_id}:dm:{inbound.sender_open_id}"
    # 群聊：按群+用户隔离
    return f"feishu:{self._channel_instance_id}:group:{inbound.chat_id}:user:{inbound.sender_open_id}"
```

### 1.4 消息批处理算法

```
消息到达
    │
    ▼
获取 Session 锁
    │
    ▼
追加到 pending_messages
    │
    ▼
更新 first_pending_at_ms（首次消息时间）
    │
    ▼
重新调度 Flush 任务
    │
    ├── 取消旧任务（如有）
    │
    ├── 计算等待时间：
    │   elapsed = now - first_pending_at_ms
    │   remaining = max_batch_window - elapsed
    │   delay = min(debounce, remaining)
    │
    └── 创建新定时任务（delay 后执行）
```

### 1.5 批处理 Prompt 渲染

```python
def _render_batch_prompt(self, chat_type: str, messages: list[FeishuInboundMessage]) -> str:
    lines: list[str] = []
    for msg in messages:
        # 去除 @ 标签
        clean_text = _TEXT_MENTION_PATTERN.sub("", msg.text).strip()
        lines.append(f"{msg.sender_name}: {clean_text}")

    latest_ask = _TEXT_MENTION_PATTERN.sub("", messages[-1].text).strip()

    if chat_type == "group":
        return (
            "<group_msgs>\n"
            + "\n".join(lines)
            + "\n</group_msgs>\n\n"
            + f"{latest_ask}"
        )

    return (
        "<dm_msgs>\n"
        + "\n".join(lines)
        + "\n</dm_msgs>\n\n"
        + f"{latest_ask}"
    )
```

### 1.6 长连接工作线程

```python
async def _start_long_connection_worker(self) -> None:
    # 1. 创建守护线程
    self._ws_thread = threading.Thread(
        target=self._run_long_connection_worker,
        name="feishu-long-connection",
        daemon=True,
    )
    self._ws_thread.start()

    # 2. 等待就绪信号（超时 15s）
    ready = await asyncio.to_thread(self._ws_ready.wait, 15.0)
    if not ready:
        raise RuntimeError("feishu_long_connection_start_timeout")

    # 3. 检查启动错误
    if self._ws_start_error is not None:
        raise RuntimeError(...)
```

**线程内执行逻辑：**

```python
def _run_long_connection_worker(self) -> None:
    # 1. 创建独立事件循环
    thread_loop = asyncio.new_event_loop()
    self._ws_loop = thread_loop
    asyncio.set_event_loop(thread_loop)

    # 2. 绑定 Lark SDK 到当前循环（关键！）
    if lark_ws_client_module is not None:
        lark_ws_client_module.loop = thread_loop

    # 3. 构建事件处理器
    event_handler = (
        lark.EventDispatcherHandler.builder(
            self._config.feishu_encrypt_key,
            self._config.feishu_verification_token,
        )
        .register_p2_im_message_receive_v1(self._on_long_connection_message)
        .build()
    )

    # 4. 启动 WebSocket 客户端
    self._ws_client = lark.ws.Client(
        self._config.feishu_app_id,
        self._config.feishu_app_secret,
        event_handler=event_handler,
        log_level=self._resolve_lark_log_level(),
    )
    self._ws_ready.set()
    self._ws_client.start()
```

**消息回调处理：**

```python
def _on_long_connection_message(self, data: Any) -> None:
    # 1. SDK 对象转标准 payload
    payload = self._build_payload_from_sdk_event(data)
    if payload is None:
        return

    # 2. 提交到主循环执行
    future = asyncio.run_coroutine_threadsafe(
        self._process_incoming_payload(payload, headers=None, require_auth=False),
        self._main_loop,  # 主事件循环
    )
    future.add_done_callback(self._on_long_connection_future_done)
```

### 1.7 Runtime Agent 生命周期

```python
async def _get_or_create_runtime_agent(self, runtime: FeishuSessionRuntime) -> Agent:
    # 1. 检查缓存
    existing = self._runtime_agents.get(runtime.runtime_agent_id)
    if existing is not None:
        return existing

    # 2. 获取基础配置
    base_config = await self._agent_registry.get_agent(runtime.base_agent_id)
    if base_config is None:
        raise RuntimeError(f"base_agent_not_found: {runtime.base_agent_id}")

    # 3. 构建运行时配置（继承 + 覆盖）
    runtime_config = self._to_runtime_config(base_config, runtime.runtime_agent_id)

    # 4. 创建 Agent 实例
    agent = await build_agent(runtime_config, self._config, self._agent_registry)
    self._runtime_agents[runtime.runtime_agent_id] = agent
    return agent
```

**配置继承机制：**

```python
def _to_runtime_config(
    self,
    base_config: AgentConfigRecord,
    runtime_agent_id: str,
) -> AgentConfigRecord:
    return AgentConfigRecord(
        id=runtime_agent_id,           # 使用运行时 ID（带 session 标识）
        name=base_config.name,           # 保留原始名称
        description=base_config.description,
        model_provider=base_config.model_provider,
        model_name=base_config.model_name,
        system_prompt=base_config.system_prompt,
        tools=list(base_config.tools),  # 深拷贝避免污染
        options=dict(base_config.options),
        model_params=dict(base_config.model_params),
        created_at=base_config.created_at,
        updated_at=datetime.now(),
    )
```

---

## 2. FeishuApiClient

文件: `console/server/channels/feishu/api_client.py` (134 行)

### 2.1 Token 管理

```python
class FeishuApiClient:
    def __init__(self, *, app_id: str, app_secret: str, api_base_url: str, ...):
        self._app_id = app_id
        self._app_secret = app_secret
        self._tenant_access_token: str | None = None
        self._token_expire_at: float = 0.0  # 过期时间戳
```

**自动刷新机制：**

```python
async def _get_tenant_access_token(self) -> str:
    now = time.time()
    # 提前 2 分钟刷新，避免边界竞争
    if self._tenant_access_token is not None and now < self._token_expire_at:
        return self._tenant_access_token

    # 请求新 token
    response = await self._request(
        "POST",
        "/open-apis/auth/v3/tenant_access_token/internal",
        json_body={"app_id": self._app_id, "app_secret": self._app_secret},
    )

    token = response.get("tenant_access_token")
    expire_seconds = int(response.get("expire", 7200))

    self._tenant_access_token = token
    # 至少保留 60s 有效期
    self._token_expire_at = now + max(60, expire_seconds - 120)
    return token
```

### 2.2 核心 API 方法

| 方法 | 功能 | 使用场景 |
|------|------|----------|
| `add_message_reaction` | 添加消息反应（emoji） | ACK 确认 |
| `reply_text` | 回复指定消息 | 正常响应 |
| `create_text_message` | 发送新消息 | Reply 失败时降级 |

### 2.3 统一请求处理

```python
async def _request(self, method: str, path: str, *, headers, params, json_body) -> dict:
    url = f"{self._api_base_url}{path}"
    response = await self._client.request(method, url, headers=headers, params=params, json=json_body)
    response.raise_for_status()

    payload = response.json()
    code = int(payload.get("code", -1))

    # 飞书 API 使用 code 字段表示业务错误
    if code != 0:
        msg = payload.get("msg", "unknown_error")
        request_id = payload.get("request_id", "")
        raise RuntimeError(f"feishu_api_error code={code} msg={msg} request_id={request_id}")

    return payload
```

---

## 3. FeishuChannelStore

文件: `console/server/channels/feishu/store.py` (178 行)

### 3.1 双模式存储

```python
class FeishuChannelStore:
    def __init__(self, db_path: str, use_persistent_store: bool):
        self._db_path = db_path
        self._use_persistent_store = use_persistent_store
        self._conn: aiosqlite.Connection | None = None

        # 内存模式回退
        self._event_dedup: set[str] = set()
        self._session_runtime_map: dict[str, FeishuSessionRuntime] = {}
```

### 3.2 数据表结构

```sql
-- 事件去重表
CREATE TABLE IF NOT EXISTS feishu_event_dedup (
    channel_instance_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (channel_instance_id, event_id)
);

-- Session Runtime 表
CREATE TABLE IF NOT EXISTS feishu_session_runtime (
    session_key TEXT PRIMARY KEY,
    agiwo_session_id TEXT NOT NULL,
    runtime_agent_id TEXT NOT NULL,
    scheduler_state_id TEXT NOT NULL,
    base_agent_id TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    chat_type TEXT NOT NULL,
    trigger_user_open_id TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

### 3.3 幂等去重

```python
async def claim_event(self, channel_instance_id: str, event_id: str) -> bool:
    dedup_key = f"{channel_instance_id}:{event_id}"

    if not self._use_persistent_store:
        # 内存模式：set 去重
        if dedup_key in self._event_dedup:
            return False
        self._event_dedup.add(dedup_key)
        return True

    # SQLite 模式：INSERT OR IGNORE
    cursor = await conn.execute(
        "INSERT OR IGNORE INTO feishu_event_dedup (...) VALUES (?, ?, ?)",
        (channel_instance_id, event_id, datetime.now(timezone.utc).isoformat()),
    )
    await conn.commit()
    return cursor.rowcount == 1  # 插入成功返回 True
```

---

## 4. 数据模型

文件: `console/server/channels/feishu/models.py`

### 4.1 FeishuInboundMessage

飞书原始消息的结构化表示：

```python
@dataclass
class FeishuInboundMessage:
    channel_instance_id: str   # Channel 实例标识
    event_id: str              # 事件唯一 ID（去重键）
    message_id: str            # 消息 ID
    chat_id: str               # 会话 ID
    chat_type: str             # "p2p" | "group"
    thread_id: str | None      # 话题 ID（可选）
    sender_open_id: str        # 发送者 Open ID
    sender_name: str           # 发送者名称（目前=open_id）
    text: str                  # 消息文本内容
    mentions: list[str]        # @ 的用户 Open ID 列表
    is_at_bot: bool            # 是否 @ 了机器人
    event_time_ms: int         # 事件时间戳
    raw_payload: dict[str, Any] # 原始 payload（用于调试）
```

### 4.2 FeishuSessionRuntime

Session 运行态持久化数据：

```python
@dataclass
class FeishuSessionRuntime:
    session_key: str           # Session 唯一键
    agiwo_session_id: str      # Agiwo 内部 Session ID
    runtime_agent_id: str      # 运行时 Agent ID（派生自 base_agent_id）
    scheduler_state_id: str    # Scheduler State ID
    base_agent_id: str         # 基础 Agent 配置 ID
    chat_id: str               # 飞书 Chat ID
    chat_type: str             # 会话类型
    trigger_user_open_id: str # 触发用户
    updated_at: datetime       # 最后更新时间
```

**ID 生成规则：**

```python
# agiwo_session_id: 基于 session_key 的 UUIDv5
agiwo_session_id = str(uuid5(NAMESPACE_URL, session_key))

# runtime_agent_id: base_agent_id + session_key 摘要
runtime_agent_id = f"{base_agent_id}--{sha1(session_key)[:12]}"

# scheduler_state_id: 初始与 runtime_agent_id 相同
scheduler_state_id = runtime_agent_id
```

### 4.3 FeishuBatchContext / FeishuBatchPayload

批处理上下文和负载：

```python
@dataclass
class FeishuBatchContext:
    session_key: str
    chat_id: str
    chat_type: str
    trigger_user_open_id: str
    trigger_message_id: str    # 用于回复
    base_agent_id: str

@dataclass
class FeishuBatchPayload:
    context: FeishuBatchContext
    messages: list[FeishuInboundMessage]  # 批量的消息列表
    rendered_user_input: str              # 渲染后的 Prompt
```

---

## 5. 关键设计决策

### 5.1 为什么需要 Runtime Agent ID？

- **隔离性**：每个 Session 有独立的 Agent 实例
- **可追踪**：从 Agent ID 可反推来源 Session
- **可清理**：Session 结束时能准确关闭对应 Agent

### 5.2 为什么群聊按用户隔离？

- 避免多用户对话互相干扰
- 每个用户有独立的上下文记忆
- 符合「个人助理」的使用模式

### 5.3 为什么选择 Debounce + 最大窗口？

- **Debounce**：快速连续输入合并处理（如用户分多条发送同一问题）
- **最大窗口**：防止无限等待，保证响应时效
- **权衡**：平衡延迟与批处理收益
