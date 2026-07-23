# Task 13：Scheduler 公开 Worker API — `worker_bridge` 去私有化

Phase 1 · 可与 Task 11/12 并行 · 修复 review 问题 #4（私有成员穿透）

## 现状与问题

`agiwo/scheduler/worker_bridge.py`（`SchedulerWorkerPort`）本应是
"agent 侧 Worker 端口 ↔ 瘦 Scheduler 公开边界"的适配器，却穿透了三处
Scheduler 私有实现：

| 位置 | 穿透 | 用途 |
| --- | --- | --- |
| `worker_bridge.py:49` | `self._scheduler._tool_control.spawn_child(...)` | 派生 Worker |
| `worker_bridge.py:93` | `self._scheduler._rt.get_result_summary(state)` | 读取 Worker 报告 |
| `worker_bridge.py:104-106` | `self._scheduler._save_state(...)` | 把 parent 状态写回 IDLE |

后果：

- Scheduler 内部重构（如 `_tool_control` / `_rt` 拆分）会静默破坏 bridge；
- import-linter / repo guard 无法约束这种"合法 import、非法访问"；
- 违背 ADR 0049 "slim Scheduler 只暴露 waitset + cancel-subtree +
  Worker 派生"的公开面设计——公开面应该收口在 `Scheduler` facade 上。

## 目标设计

在 `agiwo/scheduler/engine.py` 的 `Scheduler` facade 上补三个公开方法，
bridge 只依赖公开 API：

```python
class Scheduler:
    async def spawn_worker(self, request: SpawnChildRequest) -> AgentState:
        """Spawn a depth-1 worker child (public Worker delegation surface)."""
        return await self._tool_control.spawn_child(request)

    async def get_result_summary(self, state: AgentState) -> str | None:
        """Resolve the final report/result summary for a terminal state."""
        return await self._rt.get_result_summary(state)

    async def mark_parent_idle(self, parent_id: str) -> None:
        """Force a registered worker-parent back to IDLE after its loop ends."""
        state = await self._store.get_state(parent_id)
        if state is None or state.status is AgentStateStatus.IDLE:
            return
        await self._save_state(state.with_updates(status=AgentStateStatus.IDLE))
```

命名说明：

- `spawn_worker` 与既有 `register_worker_parent`（`engine.py`，本次改动
  已新增）形成对仗，公开面语义完整：注册 parent → 派生 worker →
  等待/取消 → 收报告 → parent 归位 IDLE；
- `mark_parent_idle` 吸收 bridge 里的 `sync_parent_idle` 实现，
  状态判断下沉到 Scheduler（它才是 AgentState 生命周期 owner）。

`worker_bridge.py` 相应简化为纯转发：

```python
async def spawn_worker(self, request: WorkerSpawnRequest) -> WorkerStateView:
    state = await self._scheduler.spawn_worker(SpawnChildRequest(...))
    return _to_worker_view(state)

async def get_worker_report(self, worker_id: str) -> str:
    await self._scheduler.wait_for(worker_id)
    state = await self._scheduler.get_state(worker_id)
    ...
    summary = await self._scheduler.get_result_summary(state)
    ...

async def sync_parent_idle(self, parent_id: str) -> None:
    await self._scheduler.mark_parent_idle(parent_id)
```

同时清理 `register_parent` 的类型噪音：`worker_bridge.py:32-43` 目前
参数是 `agent: object` + `type: ignore[arg-type]`——`WorkerSchedulerPort`
协议（`agiwo/agent/worker_port.py:53-59`）已用 `TYPE_CHECKING` 引入
`Agent` 类型，bridge 侧照做即可删掉 ignore。

## 机器护栏

在 `lint/` 的 import-linter contract 或 `scripts/repo_guard.py` 增加规则：

- **禁止 `agiwo/scheduler/` 包外访问 `Scheduler._` 前缀属性**。
  repo_guard 实现建议：对 `agiwo/`（排除 `agiwo/scheduler/`）与 `console/`
  做 AST 扫描，匹配 `Attribute(attr 以 "_" 开头)` 且 value 推导为
  scheduler 对象的模式成本高，可退化为文本规则：
  `grep -rn "scheduler\._\|_scheduler\._" --include="*.py"` 白名单为空。
  简单、有误报风险低（变量命名惯例统一）。

## 任务拆分

1. **先写契约测试**（`tests/scheduler/test_worker_public_api.py`，新文件）：
   - `Scheduler.spawn_worker` 返回的 `AgentState` 具备 depth=1、
     parent 关联正确；
   - `mark_parent_idle` 对不存在/已 IDLE 的 parent 幂等；
   - `SchedulerWorkerPort` 满足 `WorkerSchedulerPort` 协议
     （`isinstance(port, WorkerSchedulerPort)`，协议是 runtime_checkable）。
2. **在 `engine.py` 增加三个公开方法**（含 docstring，说明属于
   Worker 委派公开面，引用 ADR 0049）。
3. **改写 `worker_bridge.py`** 为纯公开 API 转发；删除 `type: ignore`。
4. **护栏落地**：repo_guard 加文本规则 + 在 `scripts/repo_guard.py`
   现有结构里注册；跑一遍确认现存代码零违规。
5. **文档同步**：`AGENTS.md` scheduler 段落补一句
   "Worker 委派公开面：`register_worker_parent` / `spawn_worker` /
   `get_result_summary` / `mark_parent_idle`；bridge 见
   `agiwo.scheduler.worker_bridge`"。

## 验收标准

- `grep -n "_scheduler\._" agiwo/scheduler/worker_bridge.py` 只命中
  公开方法调用（无 `_tool_control` / `_rt` / `_save_state`）。
- `tests/scheduler/` 全量通过；repo_guard 通过。

## 风险

- `_rt.get_result_summary` 的签名如与预期不符（接收 state 还是 id），
  以源码为准调整公开方法签名；bridge 的对外行为（`get_worker_report`
  返回字符串兜底语义，`worker_bridge.py:88-98`）保持不变。
