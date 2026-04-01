# Console Server

Agiwo Console 的控制平面后端，基于 FastAPI，为前端管理界面和 Feishu 渠道提供统一的 Agent 运行时入口。

## 架构概览

```text
console/server/
├── app.py                    # FastAPI 入口与 runtime 装配
├── config.py                 # ConsoleConfig（嵌套 server/storage/channels 配置）
├── dependencies.py           # FastAPI runtime 依赖注入
├── models/
│   ├── view.py               # API 请求/响应视图模型
│   ├── session.py            # session / chat-context / runtime binding
│   ├── agent_config.py       # Agent 配置输入模型
│   └── metrics.py            # 聚合指标模型
├── response_serialization.py # SDK/runtime -> API/SSE payload
├── routers/                  # HTTP/SSE 边界
├── services/
│   ├── runtime/              # agent factory / runtime cache / session runtime
│   ├── tool_catalog/         # tool reference / catalog / runtime builder
│   ├── agent_registry/       # Agent 配置持久化
│   ├── storage_wiring.py     # 存储配置构建
│   └── metrics.py            # run/session/state 聚合
└── channels/
    ├── session/              # 渠道批处理（SessionManager）
    └── feishu/               # Feishu 适配层
```

## 分层职责

- `routers/`
  只做 HTTP/SSE 输入输出、状态码和 response 装配。
- `services/runtime/`
  负责 Agent 构建、runtime cache、session runtime orchestration，并直接调用 SDK `Scheduler` facade。
- `services/tool_catalog/`
  负责 tool reference 解析、可用工具列表和 runtime tool 组装。
- `models/`
  放 Console 自己的数据模型；`view.py` 仅服务 API/SSE 边界，其他模块承载共享运行时模型。
- `channels/`
  只保留渠道接入逻辑：解析消息、触发规则、批处理、渠道回复、渠道 store。

## 主流程

### Web Chat

1. `routers/chat.py` 校验 Agent 配置并构建 Agent。
2. `SessionRuntimeService` 统一调用 `scheduler.route_root_input(...)`。
3. `response_serialization.py` 负责把 `AgentStreamItem` 转成 SSE payload。

### Feishu

1. `channels/feishu/` 负责消息解析、过滤、批处理和 delivery。
2. `SessionContextService` 解析/创建当前 session。
3. `AgentRuntimeCache` 负责 runtime Agent 复用与配置变更刷新。
4. `SessionRuntimeService` 负责调度执行与 stream/no-stream 统一语义。

## 开发约定

- 新的 API 请求/响应模型放进 [`models/view.py`](./models/view.py)。
- SDK/runtime 到 API/SSE 的转换统一放进 [`response_serialization.py`](./response_serialization.py)。
- 所有 Console 数据模型统一放进 `models/`，不要重新引入 `schemas.py` 或平级 `domain/` 目录。
- 不要在 router 里直接编码 Scheduler 状态机，统一走 `services/runtime/`。
