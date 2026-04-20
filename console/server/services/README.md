# Services

`services/` 是 Console Server 的应用服务层，负责把 SDK 能力组织成 Console 可用的运行时与配置服务。

## 目录

```text
services/
├── runtime/
│   ├── agent_factory.py       # build_agent / rehydrate / resume
│   ├── agent_runtime_cache.py # runtime Agent 缓存
│   ├── scheduler_tree_view_service.py
│   ├── session_runtime_service.py
│   ├── session_service.py
│   └── session_view_service.py
├── tool_catalog/
│   ├── tool_references.py
│   ├── tool_catalog.py
│   └── tool_builder.py
├── agent_registry/
├── session_store/
├── runtime_config.py
├── storage_wiring.py
└── metrics.py
```

## 核心边界

- `runtime/`
  Console 的唯一运行时编排层。这里负责 Agent 构建、session runtime orchestration、runtime cache 和持久 Agent 恢复。
- `tool_catalog/`
  负责 tool reference 解析、可用工具目录、runtime tool 组装。Router 和 runtime service 都通过它访问工具相关能力。
- `agent_registry/`
  持久化 Agent 配置并做 full-replace 校验。
- `storage_wiring.py`
  负责 run-step / trace / scheduler-state / citation 的存储配置构建。
- `metrics.py`
  负责 run/session/state 聚合，不承担 HTTP 响应序列化。

## 原则

- Router 不自己拼装 Agent 运行状态机。
- Channel 不自己管理 runtime identity 或 scheduler 路由。
- API 请求/响应视图模型在 `models/view.py`，serialization 在 `response_serialization.py`，不要回流到 `services/`。
