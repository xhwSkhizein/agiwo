# 飞书 Channel 配置与部署

## 环境变量配置

所有配置通过环境变量 `AGIWO_CONSOLE_*` 前缀读取。

### 基础启用配置

| 变量名 | 必填 | 默认值 | 说明 |
|--------|------|--------|------|
| `AGIWO_CONSOLE_FEISHU_ENABLED` | 是 | `false` | 是否启用飞书 Channel |
| `AGIWO_CONSOLE_FEISHU_MODE` | 否 | `long_connection` | 接收模式：`long_connection` 或 `webhook` |
| `AGIWO_CONSOLE_FEISHU_CHANNEL_INSTANCE_ID` | 否 | `feishu-main` | Channel 实例 ID，用于多 Channel 隔离 |

### 飞书应用凭证（必填）

| 变量名 | 必填 | 说明 |
|--------|------|------|
| `AGIWO_CONSOLE_FEISHU_APP_ID` | 是 | 飞书应用 ID，如 `cli_xxx` |
| `AGIWO_CONSOLE_FEISHU_APP_SECRET` | 是 | 飞书应用 Secret |

### Webhook 模式专用配置

| 变量名 | 必填 | 说明 |
|--------|------|------|
| `AGIWO_CONSOLE_FEISHU_VERIFICATION_TOKEN` | Webhook 必填 | 事件订阅验证令牌 |
| `AGIWO_CONSOLE_FEISHU_ENCRYPT_KEY` | 否 | 事件加密密钥（如启用加密） |

### 功能行为配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `AGIWO_CONSOLE_FEISHU_API_BASE_URL` | `https://open.feishu.cn` | 飞书 OpenAPI 地址 |
| `AGIWO_CONSOLE_FEISHU_BOT_OPEN_ID` | `""` | 机器人 Open ID，群聊 @ 触发必需 |
| `AGIWO_CONSOLE_FEISHU_DEFAULT_AGENT_ID` | `""` | 默认 Agent ID，消息处理入口 |
| `AGIWO_CONSOLE_FEISHU_WHITELIST_OPEN_IDS` | `[]` | 白名单用户 Open ID 列表（JSON 数组） |
| `AGIWO_CONSOLE_FEISHU_SDK_LOG_LEVEL` | `info` | Lark SDK 日志级别：`debug`/`info`/`warn`/`error` |

### 批处理与超时配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `AGIWO_CONSOLE_FEISHU_DEBOUNCE_MS` | `3000` | 消息防抖等待时间（毫秒） |
| `AGIWO_CONSOLE_FEISHU_MAX_BATCH_WINDOW_MS` | `15000` | 最大批处理窗口（毫秒） |
| `AGIWO_CONSOLE_FEISHU_SCHEDULER_WAIT_TIMEOUT` | `900` | Scheduler 等待超时（秒） |

### ACK 响应配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `AGIWO_CONSOLE_FEISHU_ACK_REACTION_EMOJI` | `FIREWORKS` | ACK 表情反应类型 [表情列表](https://open.feishu.cn/document/server-docs/im-v1/message-reaction/emojis-introduce) |
| `AGIWO_CONSOLE_FEISHU_ACK_FALLBACK_TEXT` | `收到，正在处理。` | 表情失败时的回退文本 |

## 飞书平台配置

### 1. 创建飞书应用

1. 访问 [飞书开放平台](https://open.feishu.cn/)
2. 点击「创建应用」
3. 选择「企业自建应用」
4. 填写应用名称和描述

### 2. 配置应用能力

进入应用详情页，开启以下能力：

| 能力 | 配置位置 | 说明 |
|------|----------|------|
| 机器人 | 「添加应用能力」→「机器人」 | 开启后才能接收/发送消息 |
| 权限 | 「权限管理」 | 添加以下权限： |
| | | - `im:chat:readonly`（读取群组信息） |
| | | - `im:message:send`（发送消息） |
| | | - `im:message.reaction:write`（添加消息反应） |

### 3. 配置事件订阅（根据模式选择）

#### 长连接模式

1. 进入「事件订阅」
2. 开启「长连接模式」
3. 订阅事件类型：
   - `im.message.receive_v1`（接收消息）
   - `im.message.message_read_v1`（消息已读，可选）

#### Webhook 模式

1. 进入「事件订阅」
2. 关闭「长连接模式」（如开启）
3. 设置「请求地址 URL」: `https://your-domain.com/api/channels/feishu/events`
4. 设置「Verification Token」并同步到环境变量
5. 订阅事件类型同长连接模式

### 4. 获取凭证信息

| 信息 | 位置 | 用途 |
|------|------|------|
| App ID | 「凭证与基础信息」 | `AGIWO_CONSOLE_FEISHU_APP_ID` |
| App Secret | 「凭证与基础信息」→ 查看 | `AGIWO_CONSOLE_FEISHU_APP_SECRET` |
| Bot Open ID | 「机器人」→ 查看机器人信息 | `AGIWO_CONSOLE_FEISHU_BOT_OPEN_ID` |
| Verification Token | 「事件订阅」 | `AGIWO_CONSOLE_FEISHU_VERIFICATION_TOKEN` |

### 5. 发布应用

1. 进入「版本管理与发布」
2. 点击「创建版本」
3. 填写版本信息
4. 申请发布（需管理员审批）

## 部署模式对比

### 长连接模式（推荐开发环境）

**优点：**
- 无需公网 IP 或域名
- 无需配置 SSL 证书
- 本地开发即可调试

**缺点：**
- 依赖 WebSocket 连接稳定性
- 重启服务会中断连接
- 不适合多实例部署

**配置示例：**

```bash
AGIWO_CONSOLE_FEISHU_ENABLED=true
AGIWO_CONSOLE_FEISHU_MODE=long_connection
AGIWO_CONSOLE_FEISHU_APP_ID=cli_xxxxxxxxxxxx
AGIWO_CONSOLE_FEISHU_APP_SECRET=xxxxxxxxxxxxxxxx
AGIWO_CONSOLE_FEISHU_DEFAULT_AGENT_ID=default-agent
AGIWO_CONSOLE_FEISHU_BOT_OPEN_ID=ou_xxxxxxxxxxxxxxxx
```

### Webhook 模式（推荐生产环境）

**优点：**
- 更好的稳定性和可扩展性
- 支持多实例负载均衡
- 符合常规 Web 服务部署模式

**缺点：**
- 需要公网可访问的域名
- 需要配置 HTTPS
- 需要处理幂等去重

**配置示例：**

```bash
AGIWO_CONSOLE_FEISHU_ENABLED=true
AGIWO_CONSOLE_FEISHU_MODE=webhook
AGIWO_CONSOLE_FEISHU_APP_ID=cli_xxxxxxxxxxxx
AGIWO_CONSOLE_FEISHU_APP_SECRET=xxxxxxxxxxxxxxxx
AGIWO_CONSOLE_FEISHU_VERIFICATION_TOKEN=xxxxxxxxxxxxxxxx
AGIWO_CONSOLE_FEISHU_DEFAULT_AGENT_ID=default-agent
AGIWO_CONSOLE_FEISHU_BOT_OPEN_ID=ou_xxxxxxxxxxxxxxxx
```

## 完整配置示例

### .env 文件

```bash
# =============================================================================
# [A] Console Server
# =============================================================================
AGIWO_CONSOLE_STORAGE_TYPE=sqlite
AGIWO_CONSOLE_SQLITE_DB_PATH=.agiwo/agiwo.db
AGIWO_CONSOLE_HOST=0.0.0.0
AGIWO_CONSOLE_PORT=8422

# =============================================================================
# [B] Feishu Channel
# =============================================================================
# 基础配置
AGIWO_CONSOLE_FEISHU_ENABLED=true
AGIWO_CONSOLE_FEISHU_MODE=long_connection
AGIWO_CONSOLE_FEISHU_CHANNEL_INSTANCE_ID=feishu-main

# 应用凭证（从飞书开放平台获取）
AGIWO_CONSOLE_FEISHU_APP_ID=cli_xxxxxxxxxxxx
AGIWO_CONSOLE_FEISHU_APP_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Webhook 模式专用（长连接可不配置）
AGIWO_CONSOLE_FEISHU_VERIFICATION_TOKEN=xxxxxxxxxxxxxxxx
AGIWO_CONSOLE_FEISHU_ENCRYPT_KEY=

# 机器人标识（群聊 @ 触发必需）
AGIWO_CONSOLE_FEISHU_BOT_OPEN_ID=ou_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 默认 Agent（必须先在 Console 创建）
AGIWO_CONSOLE_FEISHU_DEFAULT_AGENT_ID=my-assistant-agent

# 访问控制（空数组表示允许所有）
AGIWO_CONSOLE_FEISHU_WHITELIST_OPEN_IDS=[]

# 性能调优
AGIWO_CONSOLE_FEISHU_DEBOUNCE_MS=3000
AGIWO_CONSOLE_FEISHU_MAX_BATCH_WINDOW_MS=15000
AGIWO_CONSOLE_FEISHU_SCHEDULER_WAIT_TIMEOUT=900

# ACK 响应
AGIWO_CONSOLE_FEISHU_ACK_REACTION_EMOJI=FIREWORKS
AGIWO_CONSOLE_FEISHU_ACK_FALLBACK_TEXT="收到，正在处理。"

# SDK 日志
AGIWO_CONSOLE_FEISHU_SDK_LOG_LEVEL=info

# =============================================================================
# [C] LLM Provider（根据实际使用的模型配置）
# =============================================================================
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.openai.com/v1

# 或 DeepSeek
DEEPSEEK_API_KEY=xxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

## 启动验证

### 1. 检查服务启动日志

```bash
$ uv run uvicorn server.app:app --reload --env-file .env

# 成功启动会看到：
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:feishu_long_connection_started
```

### 2. 检查健康状态

```bash
$ curl http://localhost:8422/api/health
{"status": "ok", "service": "agiwo-console"}

$ curl http://localhost:8422/api/channels/feishu/status
{
  "enabled": true,
  "mode": "long_connection",
  "long_connection_alive": true,
  "session_count": 0
}
```

### 3. 发送测试消息

在飞书客户端：
- 单聊：直接向机器人发送消息
- 群聊：@机器人并发送消息

预期响应：
1. 消息显示「FIREWORKS」表情（ACK）
2. 等待 3-15 秒后收到 Agent 回复

## 故障排查

### 启动失败

| 错误信息 | 原因 | 解决 |
|----------|------|------|
| `missing required config: AGIWO_CONSOLE_FEISHU_APP_ID` | 缺少必填配置 | 检查环境变量 |
| `default agent not found` | 默认 Agent 不存在 | 在 Console 创建对应 Agent |
| `lark_oapi_not_installed` | 缺少依赖 | `pip install lark-oapi` |
| `feishu_long_connection_start_timeout` | 长连接启动超时 | 检查网络连接，查看 SDK 日志 |

### 消息无响应

| 现象 | 排查步骤 |
|------|----------|
| 无任何反应 | 1. 检查 `feishu_enabled=true`<br>2. 检查 `default_agent_id` 存在<br>3. 查看日志是否有事件接收 |
| 有 ACK 无回复 | 1. 检查 Agent 配置（模型凭证）<br>2. 查看 Scheduler 状态 |
| 群聊不触发 | 1. 确认已 @ 机器人<br>2. 检查 `bot_open_id` 配置正确<br>3. 确认机器人在群内 |
| 提示无权限 | 1. 检查应用权限配置<br>2. 确认应用已发布 |

### 日志查看

```bash
# 设置调试日志级别
AGIWO_CONSOLE_FEISHU_SDK_LOG_LEVEL=debug
LOG_LEVEL=DEBUG

# 查看结构化日志
LOG_JSON=true
```

## 生产环境建议

1. **使用 Webhook 模式**：更稳定，支持多实例
2. **配置白名单**：限制可访问用户
3. **调整批处理参数**：
   - 低并发：降低 `DEBOUNCE_MS` 提升响应速度
   - 高并发：提高 `MAX_BATCH_WINDOW_MS` 提升吞吐
4. **监控关键指标**：
   - `session_count`：活跃会话数
   - `long_connection_alive`：连接健康状态
5. **配置告警**：Scheduler 等待超时、长连接断开等
