## Console/server 代码问题清单（按严重程度排序）

### 1. **Session mutation 过度抽象** — 最严重

`@console/server/channels/session/binding.py` 的 [SessionMutationPlan](cci:2://file:///Users/hongv/workspace/agiwo/console/server/channels/session/binding.py:9:0-14:47) 模式增加复杂度却无收益：

```python
@dataclass(frozen=True)
class SessionMutationPlan:
    chat_context: ChannelChatContext
    current_session: Session
    previous_session: Session | None = None
    retired_runtime_agent_id: str | None = None
```

**问题**：
- 6 个函数都返回这个 wrapper 对象，但调用方只用了里面的 [session](cci:9://file:///Users/hongv/workspace/agiwo/console/server/channels/session:0:0-0:0)
- [apply_session_mutation()](cci:1://file:///Users/hongv/workspace/agiwo/console/server/channels/session/models.py:127:4-130:18) 多做一层间接调用
- 直接 `return Session(...)` 然后 [store.upsert_session(session)](cci:1://file:///Users/hongv/workspace/agiwo/console/server/channels/session/models.py:126:4-126:65) 就够了

**简化建议**：删掉 [SessionMutationPlan](cci:2://file:///Users/hongv/workspace/agiwo/console/server/channels/session/binding.py:9:0-14:47)，函数直接返回 [Session](cci:2://file:///Users/hongv/workspace/agiwo/console/server/channels/session/models.py:56:0-70:43)，调用方自己决定要不要调用 [store.upsert_session()](cci:1://file:///Users/hongv/workspace/agiwo/console/server/channels/session/models.py:126:4-126:65)。

---

### 2. **Feishu Service Factory 过度分层**

`@console/server/channels/feishu/factory.py` 制造了大量中间对象：

```
FeishuServiceFactory.create_components() → FeishuServiceComponents (13个字段)
```

**问题**：
- 这些组件只是在 [service.py](cci:7://file:///Users/hongv/workspace/agiwo/console/server/channels/feishu/service.py:0:0-0:0) 的 [__init__](cci:1://file:///Users/hongv/workspace/agiwo/console/server/channels/agent_executor.py:29:4-38:31) 中被拆包再赋值给 `self._xxx`
- [FeishuServiceComponents](cci:2://file:///Users/hongv/workspace/agiwo/console/server/channels/feishu/factory.py:35:0-51:20) 这个 dataclass 没有业务逻辑，纯粹是传递容器
- 增加 50+ 行代码，减少的直接性收益几乎为零

**简化建议**：直接在 [FeishuChannelService.__init__](cci:1://file:///Users/hongv/workspace/agiwo/console/server/channels/feishu/service.py:52:4-105:28) 里创建这些组件。

---

### 3. **RuntimeAgentPool 缓存逻辑过重**

`@console/server/channels/runtime_agent_pool.py:50-104` 的 fingerprint 缓存机制：

```python
def _build_runtime_config_fingerprint(...) -> str:
    payload = {"name": ..., "description": ..., ...}
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()
```

**问题**：
- 用户真的会在 session 存续期间改 agent config 吗？
- 即使有变化，重建 agent 的代价真的大到需要缓存吗？
- 50 行代码处理了一个边界情况，却增加了理解负担

**简化建议**：删除 fingerprint 缓存，每次直接 [build_agent()](cci:1://file:///Users/hongv/workspace/agiwo/console/server/services/agent_lifecycle.py:77:0-128:5)。

---

### 4. **AgentExecutor 是薄薄一层包装**

`@console/server/channels/agent_executor.py` 只有 86 行，却多了一层抽象：

```python
class AgentExecutor:
    async def execute(self, agent, session, user_input) -> RouteResult:
        ...
        result = await self._scheduler.route_root_input(...)
```

**问题**：
- 只是把 `scheduler.route_root_input()` 包了一层
- 增加了 [_touch_session()](cci:1://file:///Users/hongv/workspace/agiwo/console/server/channels/agent_executor.py:82:4-84:49) 和 [mark_session_task_started()](cci:1://file:///Users/hongv/workspace/agiwo/console/server/channels/session/binding.py:166:0-169:34) 调用，但这些可以移到调用方

**简化建议**：删掉这个类，直接在 [service.py](cci:7://file:///Users/hongv/workspace/agiwo/console/server/channels/feishu/service.py:0:0-0:0) 和 [chat_sse.py](cci:7://file:///Users/hongv/workspace/agiwo/console/server/services/chat_sse.py:0:0-0:0) 调用 `scheduler.route_root_input()`。

---

### 5. **SessionManager 的 debounce 机制可能过度**

`@console/server/channels/session/manager.py:35-161` 实现了复杂的 batching/debounce：

```python
class SessionManager:
    def _reschedule_flush_locked(self, ...):
        ...
        delay_ms = min(self._debounce_ms, remaining_window_ms)
```

**问题**：
- Feishu 需要 debounce 是因为群聊可能有突发消息
- Console Web 用户是单会话交互，真的需要 debounce 吗？
- Console 和 Feishu 共用一套机制，但需求不同

**简化建议**：Console chat 路径直接处理，不经过 [SessionManager](cci:2://file:///Users/hongv/workspace/agiwo/console/server/channels/session/manager.py:34:0-159:21)。

---

### 6. **Response 序列化层重复**

`@console/server/response_serialization.py` 和 `@console/server/domain/sessions.py` 都有转换逻辑：

```python
# response_serialization.py
def session_aggregate_to_summary_data(session: SessionAggregate) -> SessionSummaryData:
    ...

# 实际上只是字段拷贝，中间多了一个 SessionAggregate dataclass
```

**问题**：
- [SessionAggregate](cci:2://file:///Users/hongv/workspace/agiwo/console/server/domain/sessions.py:15:0-22:31) 是内存中间结构，没有独立语义
- [collect_session_aggregates()](cci:1://file:///Users/hongv/workspace/agiwo/console/server/services/metrics.py:121:0-143:5) → [SessionAggregate](cci:2://file:///Users/hongv/workspace/agiwo/console/server/domain/sessions.py:15:0-22:31) → [session_aggregate_to_summary_data()](cci:1://file:///Users/hongv/workspace/agiwo/console/server/domain/sessions.py:37:0-58:5) 是多余的跳转

**简化建议**：合并 [SessionAggregate](cci:2://file:///Users/hongv/workspace/agiwo/console/server/domain/sessions.py:15:0-22:31) 和 [SessionSummaryData](cci:2://file:///Users/hongv/workspace/agiwo/console/server/domain/sessions.py:25:0-34:33)，直接从 run 聚合到 Pydantic schema。

---

### 7. **Schema 中嵌套过多业务验证**

`@console/server/schemas.py:45-75`：

```python
class AgentOptionsInput(AgentOptions):
    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        return sanitize_agent_options_data(data, preserve_non_dict=True)
```

**问题**：
- Schema 层应该只管数据形状，业务逻辑验证应该放 service 层
- 嵌套的 validator 让代码追踪困难

**简化建议**：简化 schema，验证逻辑移到 `AgentLifecycle` 或 `AgentRegistry`。

---

### 8. **Metric 聚合的遍历次数过多**

`@console/server/services/metrics.py:150-177`：

```python
async def build_metrics_by_state(states, run_storage):
    for session_id, agent_ids in session_to_agent_ids.items():
        async for runs in iter_runs_paginated(...):
            ...
```

**问题**：
- 每个 API 请求都要扫描 runs（可能有数百次 DB 查询）
- 没有缓存或预聚合

**简化建议**：这是一个数据模型问题，RunStepStorage 应该支持 `aggregate_by_session()` 原生查询。

---

## 总结

| 问题 | 简化收益 | 风险 |
|------|---------|------|
| SessionMutationPlan | 高（删 100+ 行） | 低 |
| Feishu factory 模式 | 高（删 150+ 行） | 低 |
| RuntimeAgentPool 缓存 | 高（删 50+ 行） | 中（可能影响热更新场景）|
| AgentExecutor 抽象 | 中（删 86 行） | 低 |
| SessionManager debounce | 中 | 中（需确认 Console 是否真的不用）|
| SessionAggregate 中间层 | 中 | 低 |
| Schema validator | 低 | 低 |

**最优先修复**：1、2、4 —— 这些是纯粹的多余抽象，删除后代码更易读。