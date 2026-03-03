# 飞书 Channel 核心流程

## 1. 系统启动流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         应用启动 (app.py)                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────┴───────────────┐
                    │      ConsoleConfig 加载       │
                    │   从环境变量 AGIWO_CONSOLE_*   │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │   feishu_enabled = true?      │
                    └───────────────┬───────────────┘
                          │                       │
                         是                      否
                          │                       │
                          ▼                       ▼
            ┌─────────────────────┐      ┌─────────────────┐
            │  必填配置校验       │      │ set_feishu_     │
            │  - app_id           │      │ channel_        │
            │  - app_secret       │      │ service(None)   │
            │  - default_agent_id │      └─────────────────┘
            │  - verification_token
            │    (webhook模式)    │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │ 验证 Agent 存在性   │
            │ agent_registry.get  │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │ 创建 FeishuChannel  │
            │     Service        │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │ service.initialize() │
            │  - store.connect()   │
            │  - 启动长连接工作线程 │
            └─────────────────────┘
```

### 关键代码

```python
# console/server/app.py:48-76
if config.feishu_enabled:
    # 必填项校验
    missing = []
    if not config.feishu_app_id:
        missing.append("AGIWO_CONSOLE_FEISHU_APP_ID")
    # ... 其他校验

    base_agent = await agent_registry.get_agent(config.feishu_default_agent_id)
    if base_agent is None:
        raise RuntimeError("default agent not found")

    feishu_channel_service = FeishuChannelService(
        config=config,
        scheduler=sched,
        agent_registry=agent_registry,
    )
    await feishu_channel_service.initialize()
    set_feishu_channel_service(feishu_channel_service)
```

## 2. 消息接收流程

### 2.1 Webhook 模式

```
┌─────────────────────────────────────────────────────────────────┐
│  Feishu 服务器                                                    │
│  (当用户发送消息时)                                               │
└──────────────────┬──────────────────────────────────────────────┘
                   │ POST /api/channels/feishu/events
                   │ payload: {header, event, ...}
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  Feishu Router (feishu.py)                                      │
│  @router.post("/events")                                        │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  service.handle_webhook_event(headers, payload)                   │
│                                                                 │
│  1. 校验 feishu_mode == "webhook"                               │
│  2. 调用 _process_incoming_payload()                           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 长连接模式

```
┌─────────────────────────────────────────────────────────────────┐
│  长连接工作线程 (_run_long_connection_worker)                    │
│  - 独立线程运行 WebSocket 客户端                                 │
│  - 使用 lark-oapi SDK                                           │
└──────────────────┬──────────────────────────────────────────────┘
                   │ WebSocket 事件
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  _on_long_connection_message(data)                               │
│  - 将 SDK 事件对象转为标准 payload                               │
│  - 提交到主事件循环处理                                          │
└──────────────────┬──────────────────────────────────────────────┘
                   │ asyncio.run_coroutine_threadsafe()
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  _process_incoming_payload(payload, headers=None,               │
│                            require_auth=False)                  │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 统一消息处理流程

```
┌──────────────────────────────────────────────────────────────────────┐
│ _process_incoming_payload(payload, headers, require_auth)              │
└────────────────────┬─────────────────────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
         ▼           ▼           ▼
    ┌─────────┐ ┌──────────┐ ┌──────────┐
    │关闭检查  │ │URL验证   │ │认证检查  │
    │_closed?  │ │challenge │ │token校验 │
    └─────────┘ └──────────┘ └──────────┘
                     │
                     ▼
            ┌────────────────┐
            │ 解析消息内容    │
            │ _parse_inbound  │
            │ _message()      │
            └────────┬───────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
    ┌──────────┐            ┌──────────┐
    │ 事件去重  │            │ 触发检查  │
    │claim_event│            │_should_  │
    │          │            │trigger() │
    └──────────┘            └──────────┘
                                   │
                         ┌─────────┴─────────┐
                         │                   │
                         ▼                   ▼
                    ┌─────────┐       ┌──────────┐
                    │ 发送 ACK │       │ 忽略消息 │
                    │ reaction│       │ (返回)   │
                    │ 或文本   │       │          │
                    └─────────┘       └──────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ _enqueue_message │
                │ (进入批处理队列) │
                └─────────────────┘
```

### 触发条件检查 (_should_trigger)

```python
def _should_trigger(self, inbound: FeishuInboundMessage) -> bool:
    # 1. 必须有默认 Agent
    if not self._default_agent_id:
        return False

    # 2. 白名单检查
    if not self._is_whitelisted(inbound.sender_open_id):
        return False

    # 3. 群聊必须 @ 机器人
    if inbound.chat_type == "group":
        return inbound.is_at_bot  # bot_open_id in mentions

    # 4. 单聊直接触发
    if inbound.chat_type == "p2p":
        return True

    return False
```

## 4. 消息批处理流程

```
┌────────────────────────────────────────────────────────────────────┐
│ _enqueue_message(inbound)                                            │
│ - 构建 session_key                                                  │
│ - 获取 session 锁                                                   │
└──────────────────┬─────────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────────────┐
│ 获取/创建 _SessionState                                              │
│ - pending_messages: 消息队列                                        │
│ - first_pending_at_ms: 首次消息时间                                  │
│ - running: 是否正在处理                                             │
└──────────────────┬─────────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────────────┐
│ _reschedule_flush_locked()                                           │
│                                                                     │
│ 计算等待时间:                                                        │
│ elapsed = now - first_pending_at                                    │
│ remaining = max_batch_window - elapsed                              │
│ delay = min(debounce_ms, remaining)                                  │
│                                                                     │
│ 取消旧任务 → 创建新定时任务                                          │
└──────────────────┬─────────────────────────────────────────────────┘
                   │
                   │ 等待 delay_ms
                   ▼
┌────────────────────────────────────────────────────────────────────┐
│ _flush_session_after_delay() → _flush_session()                     │
│                                                                     │
│ 1. 提取所有 pending_messages                                        │
│ 2. 清空队列，重置计时器                                              │
│ 3. 设置 running = True                                              │
│ 4. 构建 FeishuBatchPayload                                          │
└──────────────────┬─────────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────────────┐
│ _run_batch_with_scheduler(batch)                                   │
│                                                                     │
│ 核心逻辑：                                                          │
│ - 获取/创建 Session Runtime                                         │
│ - 获取/创建 Agent 实例                                              │
│ - 与 Scheduler 交互提交任务                                          │
└────────────────────────────────────────────────────────────────────┘
```

### 批处理时间线示例

```
时间轴 (ms)
0      1000   2500   3000   4000   5000
│       │      │      │      │      │
▼       ▼      ▼      ▼      ▼      ▼
Msg1   Msg2   Msg3
│       │      │
└───────┴──────┘
      Debounce
      delay=3000ms
              │
              ▼
         触发处理
         (Batch: Msg1+Msg2+Msg3)

如果持续有新消息，最大窗口保证 15000ms 后必处理
```

## 5. Scheduler 交互流程

```
┌─────────────────────────────────────────────────────────────────────┐
│ _run_batch_with_scheduler(batch)                                    │
└────────────────────┬────────────────────────────────────────────────┘
                     │
         ┌───────────┴────────────┐
         │                        │
         ▼                        ▼
┌─────────────────┐      ┌─────────────────┐
│ _get_or_create  │      │ _get_or_create  │
│ _runtime()      │      │ _runtime_agent() │
│                 │      │                 │
│ - 查 store      │      │ - 查缓存        │
│ - 创建新 runtime│      │ - build_agent() │
│ - 生成 agent_id │      │ - 缓存 agent    │
└────────┬────────┘      └────────┬────────┘
         │                        │
         └───────────┬────────────┘
                     │
                     ▼
         ┌─────────────────────┐
         │ 检查当前 state 状态  │
         │ scheduler.get_state │
         └──────────┬──────────┘
                    │
        ┌───────────┼───────────┬───────────┐
        │           │           │           │
       None    SLEEPING   RUNNING/PENDING  其他
        │           │           │           │
        ▼           ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │submit() │ │submit_  │ │wait_for │ │submit() │
   │persistent│ │task()   │ │+ check  │ │(新)     │
   │=True    │ │         │ │+ submit_│ │         │
   │         │ │         │ │task()   │ │         │
   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
        │           │           │           │
        └───────────┴───────────┴───────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │ 统一等待执行结果     │
        │ wait_for(timeout)   │
        └─────────────────────┘
```

### 状态处理详细逻辑

```python
# 伪代码展示核心状态机
async def _run_batch_with_scheduler(batch):
    runtime = await _get_or_create_runtime(batch.context)
    agent = await _get_or_create_runtime_agent(runtime)

    current_state = await scheduler.get_state(runtime.scheduler_state_id)

    if current_state is None:
        # 首次：提交新的 Persistent Agent
        state_id = await scheduler.submit(
            agent, batch.rendered_user_input,
            session_id=runtime.agiwo_session_id,
            persistent=True
        )
        runtime.scheduler_state_id = state_id

    elif current_state.status == SLEEPING:
        # 休眠中：提交新任务唤醒
        await scheduler.submit_task(
            runtime.scheduler_state_id,
            batch.rendered_user_input
        )

    elif current_state.status in (RUNNING, PENDING):
        # 运行中：等待完成后检查状态
        wait_output = await scheduler.wait_for(
            runtime.scheduler_state_id,
            timeout=self._scheduler_wait_timeout
        )
        refreshed = await scheduler.get_state(runtime.scheduler_state_id)

        if refreshed.status == SLEEPING:
            # 已休眠：提交新任务
            await scheduler.submit_task(...)
        else:
            # 仍在运行或异常：报错或重建
            raise RuntimeError("previous_task_still_running")

    # 统一等待最终结果
    output = await scheduler.wait_for(
        runtime.scheduler_state_id,
        timeout=self._scheduler_wait_timeout
    )
    return output
```

## 6. 响应发送流程

```
┌─────────────────────────────────────────────────────────────────────┐
│ _send_final_response(context, response_text)                        │
└────────────────────┬────────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    群聊?                      单聊
         │                       │
         ▼                       ▼
┌─────────────────┐      ┌─────────────────┐
│ 添加 @用户前缀   │      │ 直接使用回复文本 │
│ <at user_id="x"> │      │                 │
└────────┬────────┘      └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌─────────────────────┐
         │ 尝试 reply_text() │
         │ (回复原消息)       │
         └──────────┬──────────┘
                    │
            失败? ──┴── 成功
            │            │
            ▼            ▼
    ┌─────────────┐ ┌─────────────┐
    │ fallback    │ │   Done      │
    │ create_text │ │             │
    │ _message()  │ │             │
    └─────────────┘ └─────────────┘
```

## 7. 关闭清理流程

```
service.close()
    │
    ├── 设置 _closed = True
    │
    ├── 取消所有 flush_task
    │   └── 等待任务完成/取消
    │
    ├── _stop_long_connection_worker()
    │   ├── 关闭 WebSocket 客户端
    │   ├── 停止事件循环
    │   └── 等待线程退出
    │
    ├── 关闭所有 runtime_agents
    │   └── agent.close()
    │
    ├── _api.close()
    │
    └── _store.close()
```

## 8. 错误处理策略

| 阶段 | 错误类型 | 处理策略 |
|------|----------|----------|
| 配置校验 | 缺少必填项 | 启动时抛出 RuntimeError，阻止服务启动 |
| Agent 不存在 | 默认 Agent 未找到 | 启动时抛出 RuntimeError |
| 消息解析 | 无效 payload | 静默忽略，返回 ignored 响应 |
| 认证失败 | Token 不匹配 | 返回 unauthorized 响应 |
| 事件重复 | event_id 已处理 | 返回 duplicate 响应 |
| ACK 发送 | Reaction/Reply 失败 | 降级处理，记录 warning 日志 |
| 执行失败 | Agent 执行出错 | 捕获异常，发送用户友好的错误消息 |
| 超时 | Scheduler 等待超时 | 返回超时提示，下次请求重建 Agent |
