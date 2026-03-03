# 飞书 Channel API 接口

## 目录

1. [Webhook 回调接口](#1-webhook-回调接口)
2. [状态查询接口](#2-状态查询接口)

---

## 1. Webhook 回调接口

### POST `/api/channels/feishu/events`

接收飞书事件订阅的回调请求。

#### 请求

**Content-Type**: `application/json`

**Headers**:
| 字段 | 说明 |
|------|------|
| `token` (payload 内) | 验证令牌（Webhook 模式） |

**Body** (飞书标准事件格式):

```json
{
  "schema": "2.0",
  "header": {
    "event_id": "xxxx",
    "token": "verification_token",
    "create_time": "1234567890000",
    "event_type": "im.message.receive_v1",
    "tenant_key": "xxx",
    "app_id": "cli_xxx"
  },
  "event": {
    "message": {
      "message_id": "om_xxx",
      "root_id": "om_xxx",
      "parent_id": "om_xxx",
      "create_time": "1234567890000",
      "chat_id": "oc_xxx",
      "chat_type": "p2p",
      "message_type": "text",
      "content": "{\"text\":\"Hello\"}",
      "mentions": [
        {
          "key": "@_user_1",
          "id": {
            "union_id": "on_xxx",
            "open_id": "ou_xxx",
            "user_id": "xxx"
          },
          "name": "Bot",
          "tenant_key": "xxx"
        }
      ],
      "sender": {
        "sender_id": {
          "union_id": "on_xxx",
          "open_id": "ou_xxx",
          "user_id": "xxx"
        },
        "sender_type": "user",
        "tenant_key": "xxx"
      }
    }
  }
}
```

#### 响应

**URL 验证响应** (首次配置回调时):

```json
{
  "challenge": "xxxx"
}
```

**正常处理响应**:

```json
{
  "msg": "ok"
}
```

**忽略响应** (消息不触发处理):

```json
{
  "msg": "ignored_not_trigger"
}
```

或

```json
{
  "msg": "ignored_duplicate"
}
```

#### 错误响应

| HTTP 状态码 | 场景 |
|-------------|------|
| `400` | 请求体格式错误 |
| `503` | 飞书 Channel 未启用 |

---

## 2. 状态查询接口

### GET `/api/channels/feishu/status`

查询飞书 Channel 的运行状态。

#### 响应

**Channel 启用时**:

```json
{
  "enabled": true,
  "mode": "long_connection",
  "long_connection_alive": true,
  "session_count": 5
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `enabled` | boolean | 是否启用 |
| `mode` | string | 当前模式：`long_connection` / `webhook` |
| `long_connection_alive` | boolean | 长连接是否存活（仅长连接模式有效） |
| `session_count` | integer | 当前活跃 Session 数量 |

**Channel 禁用时**:

```json
{
  "enabled": false
}
```

---

## 3. 内部事件类型

以下事件类型由飞书平台发送，服务内部处理：

| 事件类型 | 处理行为 |
|----------|----------|
| `url_verification` | 返回 challenge 响应，用于 Webhook 配置验证 |
| `im.message.receive_v1` | 解析并进入消息处理流程 |
| `im.message.message_read_v1` | 忽略（仅防止 SDK 报错） |

---

## 4. 消息处理结果内部状态

以下状态由 `_process_incoming_payload` 返回（不直接暴露给飞书）：

| 状态码 | 含义 |
|--------|------|
| `ok` | 消息已接收并进入处理队列 |
| `ignored_webhook_mode_disabled` | Webhook 模式未启用 |
| `ignored_unauthorized` | 验证失败 |
| `ignored_non_message_event` | 非消息类型事件 |
| `ignored_invalid_payload` | Payload 解析失败 |
| `ignored_duplicate` | 事件重复（已处理过） |
| `ignored_not_trigger` | 不满足触发条件（非白名单/未@等） |
| `feishu_channel_closed` | Channel 服务已关闭 |

---

## 5. 与飞书平台的对接流程

### 5.1 长连接模式对接

```
1. 开发者启动服务
   │
2. FeishuChannelService.initialize()
   └── _start_long_connection_worker()
       └── 创建 WebSocket 连接
   │
3. 飞书平台识别连接建立
   │
4. 消息事件通过 WebSocket 推送到本地
   │
5. _on_long_connection_message() 处理
```

**无需公网地址，无需配置回调 URL**

### 5.2 Webhook 模式对接

```
1. 开发者配置飞书应用
   └── 设置请求 URL: https://api.example.com/api/channels/feishu/events
   │
2. 飞书发送 url_verification 事件
   │
3. 服务返回 challenge 响应
   │
4. 飞书验证通过，配置保存成功
   │
5. 后续消息通过 HTTP POST 推送
```

**需要公网可访问的 HTTPS 地址**

---

## 6. 安全考虑

### 6.1 Webhook 模式安全

| 机制 | 说明 |
|------|------|
| Verification Token | 校验请求来源合法性 |
| Encrypt Key | 支持事件内容加密（可选） |
| Event ID 去重 | 防止消息重放攻击 |

### 6.2 长连接模式安全

| 机制 | 说明 |
|------|------|
| TLS | WebSocket 使用 wss:// 加密 |
| App Secret | SDK 内部使用，不暴露 |

### 6.3 应用层安全

| 机制 | 说明 |
|------|------|
| 白名单 | 限制可访问用户 |
| @ 触发 | 群聊必须 @ 才响应，防止误触发 |
| ACK 机制 | 即时确认收到，提升用户体验 |
