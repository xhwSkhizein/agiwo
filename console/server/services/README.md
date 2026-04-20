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
- `session_store/`
  Console 的 session / chat-context 存储边界，负责 `Session` 与 `ChannelChatContext` 的持久化接口、memory/sqlite backing store adapter，以及 session 读写工厂。主入口看 [`SessionStore`](./session_store/base.py) 协议和 [`create_session_store`](./session_store/__init__.py)。Owner: Console runtime maintainers。稳定性：这是 server 内部服务边界，不承诺第三方直接依赖其具体实现，但 `SessionStore` 协议和 `create_session_store(...)` 作为 Console 内部稳定接缝应保持兼容。
- `runtime_config.py`
  Console 的 runtime-only 配置服务，负责构建可读 snapshot、校验 runtime-editable payload、mask secret、应用 SDK `settings` override，并返回 API 层消费的 typed runtime config 响应。主入口看 [`RuntimeConfigService`](./runtime_config.py)。Owner: Console runtime maintainers。稳定性：该模块服务于 `/api/runtime-config` 等控制面接口，外部应依赖其 API 输出而非内部 helper 细节；内部实现可演进，但 service contract 应保持稳定。
- `storage_wiring.py`
  负责 run-step / trace / scheduler-state / citation 的存储配置构建。
- `metrics.py`
  负责 run/session/state 聚合，不承担 HTTP 响应序列化。

## 原则

- Router 不自己拼装 Agent 运行状态机。
- Channel 不自己管理 runtime identity 或 scheduler 路由。
- API 请求/响应视图模型在 `models/view.py`，serialization 在 `response_serialization.py`，不要回流到 `services/`。
