# 飞书 Channel 文档

本文档详细介绍 Agiwo Console 飞书 Channel 的架构设计、实现细节和使用方法。

## 文档目录

| 文档 | 内容 |
|------|------|
| [01-architecture.md](./01-architecture.md) | 架构设计 overview、定位、核心设计原则 |
| [02-core-flows.md](./02-core-flows.md) | 启动流程、消息接收、批处理、Scheduler 交互、响应发送 |
| [03-configuration.md](./03-configuration.md) | 环境变量、飞书平台配置、部署模式对比、故障排查 |
| [04-components.md](./04-components.md) | 核心组件详解（Service、ApiClient、Store、Models） |
| [05-api-reference.md](./05-api-reference.md) | API 接口定义、事件类型、对接流程 |

## 快速开始

### 1. 配置环境变量

```bash
# 基础配置
AGIWO_CONSOLE_FEISHU_ENABLED=true
AGIWO_CONSOLE_FEISHU_MODE=long_connection
AGIWO_CONSOLE_FEISHU_APP_ID=cli_xxx
AGIWO_CONSOLE_FEISHU_APP_SECRET=xxx
AGIWO_CONSOLE_FEISHU_DEFAULT_AGENT_ID=my-agent

# 群聊 @ 触发必需
AGIWO_CONSOLE_FEISHU_BOT_OPEN_ID=ou_xxx
```

### 2. 启动服务

```bash
cd console
uv run uvicorn server.app:app --reload --env-file .env
```

### 3. 验证状态

```bash
curl http://localhost:8422/api/channels/feishu/status
```

### 4. 发送测试消息

- **单聊**：直接向机器人发送消息
- **群聊**：@机器人并发送消息

## 架构速览

```
飞书客户端 ──→ 飞书开放平台 ──→ FeishuChannelService ──→ Scheduler ──→ Agent Runtime
                (Webhook/WS)        (批处理/会话管理)       (编排)         (LLM+Tools)
```

## 核心特性

- **双模式接入**：长连接（开发）+ Webhook（生产）
- **消息批处理**：自动合并连续消息，优化 LLM 调用
- **Persistent Agent**：会话级持久化，支持上下文记忆
- **安全控制**：白名单 + @触发 + 事件去重
- **降级处理**：Reaction 失败自动降级到文本回复

## 代码位置

```
console/server/channels/feishu/
├── __init__.py          # 导出 FeishuChannelService
├── service.py           # 核心服务 (937 行)
├── store.py             # 存储层 (SQLite/内存)
├── models.py            # 数据模型
├── api_client.py        # 飞书 API 客户端

console/server/routers/feishu.py      # HTTP 路由
console/server/config.py               # 配置定义
console/server/dependencies.py         # 依赖管理
console/server/app.py                  # 生命周期
```

## 依赖

```bash
pip install lark-oapi  # 飞书 SDK（长连接模式必需）
```

## 相关文档

- [Agiwo AGENTS.md](../../../../AGENTS.md) - SDK 整体架构
- [SCHEDULER_DESIGN.md](../../../../SCHEDULER_DESIGN.md) - Scheduler 设计
