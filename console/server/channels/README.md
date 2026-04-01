# Channels

`channels/` 是 Console 的渠道适配层。它不负责 Agent 编排语义，只负责把外部 IM 渠道接入到共享运行时。

## 当前职责

- 渠道消息解析、去重、触发规则
- 渠道侧批处理与防抖
- 渠道侧回复、附件、长连接管理
- 调用 `services/runtime/` 提供的共享运行时服务

## 当前结构

```text
channels/
├── exceptions.py         # 渠道异常
├── utils.py              # 文本切分、stream 文本提取、关闭辅助
├── session/
│   └── manager.py        # 批处理与 debounce
└── feishu/
    ├── commands/         # Feishu 命令系统
    ├── store/            # 渠道 metadata store
    ├── connection.py     # 长连接适配
    ├── inbound_handler.py
    ├── message_parser.py
    ├── message_builder.py
    ├── delivery_service.py
    ├── factory.py
    └── service.py
```

## 运行时协作

- `SessionContextService`
  位于 `services/runtime/`，负责 session/chat-context 生命周期。
- `AgentRuntimeCache`
  位于 `services/runtime/`，负责 runtime Agent 缓存与配置指纹刷新。
- `SessionRuntimeService`
  位于 `services/runtime/`，负责统一调用 `scheduler.route_root_input(...)`。

Feishu service 只持有这些服务并消费它们，不再在 `channels/` 内自己实现一套执行状态机。
