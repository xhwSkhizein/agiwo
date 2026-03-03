# 飞书 Channel 架构设计

## 概述

飞书 Channel 是 Agiwo Console 控制平面的外部接入层组件，负责将飞书（Lark/Feishu）的消息事件转换为 Agent 可处理的输入，并将 Agent 的执行结果回传给飞书用户。它实现了双向的消息桥接，支持两种接入模式：长连接模式（WebSocket）和 Webhook 回调模式。

## 架构定位

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Agiwo Console                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │  Sessions   │    │   Traces    │    │   Agents    │    │  Scheduler  │   │
│  │   Router    │    │   Router    │    │   Router    │    │   Router    │   │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘   │
│         └─────────────────┬───────────────────┴───────────────────┘           │
│                           │                                                  │
│  ┌────────────────────────┴────────────────────────────────────────────┐      │
│  │                    Feishu Channel Service                             │      │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐              │      │
│  │  │  Webhook    │   │  Long Conn  │   │   Store     │              │      │
│  │  │   Handler   │   │  (WebSocket)│   │             │              │      │
│  │  └─────────────┘   └─────────────┘   └─────────────┘              │      │
│  │         │                 │                  │                        │      │
│  │         └────────────────┴──────────────────┘                        │      │
│  │                              │                                       │      │
│  │                    ┌─────────┴──────────┐                            │      │
│  │                    │  Session Manager   │                            │      │
│  │                    │ (Batch/Debounce)   │                            │      │
│  │                    └─────────┬──────────┘                            │      │
│  │                              │                                       │      │
│  │                    ┌─────────┴──────────┐                            │      │
│  │                    │ Scheduler Bridge   │                            │      │
│  │                    └─────────┬──────────┘                            │      │
│  └────────────────────────────┼──────────────────────────────────────┘      │
│                               │                                              │
│  ┌────────────────────────────┼────────────────────────────────────────────┐│
│  │                    Scheduler (agiwo.scheduler)                           ││
│  │                            │                                             ││
│  │              ┌─────────────┴─────────────┐                               ││
│  │              │   Agent Runtime (per session)                             ││
│  │              │   - Persistent Agent                                      ││
│  │              │   - Child Agent Spawn                                     ││
│  │              │   - Sleep/Wait Pattern                                    ││
│  │              └───────────────────────────┘                               ││
│  └──────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               │ HTTPS / WebSocket
                               ▼
                    ┌───────────────────────┐
                    │   Feishu Open Platform  │
                    │   - Webhook Events      │
                    │   - WebSocket Stream    │
                    └───────────────────────┘
                               │
                               │
                    ┌──────────┴──────────┐
                    │   Feishu Client     │
                    │   (User's Phone/PC) │
                    └─────────────────────┘
```

## 核心设计原则

### 1. 双模式接入

| 模式 | 适用场景 | 特点 |
|------|----------|------|
| **长连接** (`long_connection`) | 开发环境、无公网 IP | WebSocket 直连飞书，无需配置回调 URL |
| **Webhook** (`webhook`) | 生产环境、有公网域名 | HTTP 回调，需要配置验证令牌 |

### 2. Session 隔离模型

每个会话通过 `session_key` 唯一标识，基于以下维度构建：

- **单聊 (p2p)**: `feishu:{instance_id}:dm:{sender_open_id}`
- **群聊 (group)**: `feishu:{instance_id}:group:{chat_id}:user:{sender_open_id}`

群聊按用户维度隔离，确保群聊中每个用户的对话独立。

### 3. 消息批处理 (Batching)

```
消息到达 → 进入 Session 队列 → Debounce 等待 → 批量处理 → 提交 Scheduler
```

- **Debounce 时间**: 默认 3000ms，快速连续消息合并处理
- **最大窗口**: 默认 15000ms，防止消息无限等待
- **批处理优势**: 将连续的多条消息合并为一次 Agent 调用，减少 LLM API 消耗

### 4. Persistent Agent 模式

每个 Session 对应一个 Persistent Agent：

- Agent 完成一次任务后进入 `SLEEPING` 状态（非终止）
- 新消息通过 `submit_task()` 唤醒，保持上下文连续性
- 超时或出错时重新创建新的 Agent 实例

### 5. 安全控制

| 控制点 | 实现 |
|--------|------|
| 白名单 | `feishu_whitelist_open_ids` 限制允许访问的用户 |
| @ 触发 | 群聊必须 @ 机器人才触发 |
| 事件去重 | `event_id` 级别的幂等处理 |
| Token 验证 | Webhook 模式支持 verification_token 校验 |

## 组件职责

### FeishuChannelService

核心服务类，负责：

- 生命周期管理（初始化、关闭）
- 双模式事件接收处理
- Session 状态管理
- 消息批处理调度
- 与 Scheduler 的桥接

### FeishuApiClient

飞书 OpenAPI 客户端：

- Tenant Access Token 自动获取与缓存
- 消息回复（reply/create）
- 消息反应（reaction emoji）

### FeishuChannelStore

数据持久层：

- 事件去重记录
- Session Runtime 状态
- 支持 SQLite / 内存两种模式

### Data Models

| Model | 职责 |
|-------|------|
| `FeishuInboundMessage` | 飞书原始消息解析后的结构化数据 |
| `FeishuSessionRuntime` | Session 运行态信息（agent_id, state_id 等） |
| `FeishuBatchContext` | 批处理上下文（chat_id, trigger_user 等） |
| `FeishuBatchPayload` | 批处理负载（messages + rendered_prompt） |

## 与 Scheduler 的集成

```python
# 首次消息：提交新的 Persistent Agent
state_id = await scheduler.submit(agent, input, session_id=xxx, persistent=True)

# 后续消息：唤醒现有 Agent
await scheduler.submit_task(state_id, new_input)

# 等待执行结果
output = await scheduler.wait_for(state_id, timeout=900)
```

### 状态流转

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ 首次消息 │ --> │ 提交    │ --> │ RUNNING │ --> │ SLEEPING│
└─────────┘     │ Agent   │     └─────────┘     └────┬────┘
                └─────────┘                          │
                                                     │ 新消息
                                                     ▼
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ 结果返回 │ <-- │ 等待    │ <-- │RUNNING  │ <-- │submit_task
└─────────┘     │ 完成    │     └─────────┘     └─────────┘
                └─────────┘
```

## 文件组织

```
console/server/channels/feishu/
├── __init__.py          # 导出 FeishuChannelService
├── service.py           # 核心服务实现 (937 行)
├── store.py             # SQLite/内存存储
├── models.py            # 数据模型定义
├── api_client.py        # 飞书 API 客户端
└── [内部实现细节...]
```

## 扩展点

如需扩展新的 Channel 类型（如钉钉、企业微信），可参考本架构实现：

1. 定义类似的 `ChannelService` 接口
2. 实现对应的消息接收适配器
3. 复用 `Session` + `Scheduler` 桥接模式
4. 遵循相同的 Persistent Agent 生命周期管理
