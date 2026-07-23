# Task 11：统一队列收敛 — `_pending` 成为唯一入队原语

Phase 1 · 优先级最高 · 修复 review 问题 #1（队头阻塞）、#5（双真相源）、#6（enqueue 返回值被忽略）

## 现状与问题

### 领域背景

ADR 0049 规定：用户中途输入、完成门反馈、Worker 报告共享
**one enqueue primitive and one Loop message queue**。当前实现里实际存在两条队列：

- `MainAgent._pending`（`agiwo/agent/main_agent.py`）：`QueueItem` 列表；
- `SessionRuntime._pending_inputs`（`agiwo/agent/runtime/session.py:114-130`）：
  live run 循环每个 assistant turn 前真正消费的 pending inputs。

真实投递走的是 `handle.enqueue_message`（落到 SessionRuntime）；
`MainAgent._pending` 对 `USER_INPUT` / `GATE_FEEDBACK` 只是**写入后永不 ack**
的记账。

### Bug：队头阻塞（确定性丢报告）

`_process_pending_worker_reports`（`main_agent.py:267-272`）从队头 peek，
遇到非 `WORKER_REPORT` 直接 `return`：

1. run 进行中用户发消息 → `accept` 把 `USER_INPUT` 追加进 `_pending`（永不 ack）；
2. 异步 Worker 完成 → `WORKER_REPORT` 排在 `USER_INPUT` 之后；
3. 任何时刻 drain 都在队头的 `USER_INPUT` 上 return → **报告永久无法投递**。

`GATE_FEEDBACK`（经 `_enqueue_gate_feedback`，`main_agent.py:109-116`）同理。

### 伴生问题

- `accept`（`main_agent.py:164`）忽略 `handle.enqueue_message` 的 `bool`
  返回值：`is_active` 为真但 SessionRuntime 恰已 close 时返回 `False`，
  消息静默丢失且不开新 run。
- `_process_pending_worker_reports` 第 281-282 行：ack 之后发现
  `_state is RUNNING` 就 `continue`——报告已出队，**静默丢弃**。
- `_state`（`MainAgentState`）与 `_handle.is_active` 两个"是否在跑"的真相源
  在 run 结束到 `_await_run_completion` finally 执行之间短暂不一致。

## 目标设计

### 队列语义重定义

明确两条队列的分工，消除重复记账：

- **`SessionRuntime._pending_inputs`**：run 内部循环队列，属于单次 run 的
  runtime，保持不变。
- **`MainAgent._pending`**：**Session 级 staging 队列**。所有跨 run 入口
  （外部用户输入、异步 Worker 报告）先入此队列，再由**唯一的 drain 例程**
  决定去向：活 run 则转投 `enqueue_message`，idle 则开新 run / 续 run。
  每个入队项**必然被 ack**（成功投递、成功开 run、或带日志地显式丢弃）。

`GATE_FEEDBACK` 不再进入 `_pending`：完成门反馈只发生在 run 存活期间，
`run_loop.py:373-378` 已直接写入 SessionRuntime 队列；MainAgent 侧的
`on_gate_feedback` 记账没有消费者，属于伪需求（详见任务拆分第 4 步）。

### drain 例程（伪代码）

```python
async def _drain(self) -> None:
    async with self._drain_lock:          # 防并发 drain
        while (item := self.peek_pending()) is not None:
            if self._handle is not None and self._handle.is_active:
                message = self._queue_item_to_message(item)
                if await self._handle.enqueue_message(message):
                    self.ack_pending()
                    continue
                # run 恰好收尾：等 completion task 落地后按 idle 路径重试
                await self._wait_completion_settled()
                continue
            # idle 路径
            self.ack_pending()
            if item.kind is QueueItemKind.USER_INPUT:
                self._start_new_run()          # 原 accept 的开 run 分支
            elif item.kind is QueueItemKind.WORKER_REPORT:
                self._continue_run(item)       # continue_completed_run
            # 每次开 run 后回到循环头，后续项会走 enqueue_message 路径
```

要点：

- `enqueue_message` 返回 `False` 不再静默：等待 completion task 结束后
  在同一循环内落到 idle 分支，消息不丢。
- `_start_new_run` / `_continue_run` 即现有 `accept` 第 174-185 行与
  `_process_pending_worker_reports` 第 283-292 行的逻辑，抽成私有方法复用。
- `RUNNING 但无活 handle` 的静默丢弃分支被消除——统一由锁 + 重试兜住。

### 状态收敛

`MainAgentState` 改为**派生属性**，删除独立的 `_state` 字段：

```python
@property
def state(self) -> MainAgentState:
    if self._handle is not None and self._handle.is_active:
        return MainAgentState.RUNNING
    return MainAgentState.IDLE
```

注意核对现有消费者：`console/server/services/runtime/session_turn_service.py`
（`wait_until_idle` / `cancel_if_active`）、
`console/server/services/runtime/agent_runtime_cache.py`
（config refresh 的 RUNNING 判断）只读 `state`，签名不变即兼容。

## 任务拆分

1. **先写回归测试**（`tests/agent/test_main_agent_queue.py`，新文件）：
   - run 进行中 `accept` 一条用户消息，随后投递异步 Worker 报告，
     断言报告最终进入 run 上下文（当前实现下该测试必然失败，锁定 bug）；
   - `enqueue_message` 返回 `False`（构造已 close 的 SessionRuntime）时
     消息不丢，落到新 run；
   - drain 后 `_pending` 为空（所有项被 ack）。
2. **实现 drain**：在 `MainAgent` 增加 `_drain_lock: asyncio.Lock` 与
   `_drain()`；`accept` / `deliver_worker_report` 改为
   "构造 `QueueItem` → `enqueue` → `await self._drain()`"，
   删除 `_process_pending_worker_reports`。
3. **状态派生化**：删除 `_state` 字段及所有赋值点
   （`main_agent.py:83, 174, 209, 281-283, 322`），改为派生属性。
4. **移除 `GATE_FEEDBACK` 记账**：
   - 删除 `MainAgent._enqueue_gate_feedback` 与
     `CompletionGateContext.on_gate_feedback`
     （`agiwo/agent/completion_gates/context.py:14`）；
   - `run_loop.py:_enqueue_gate_feedback` 只保留写 SessionRuntime 队列与日志；
   - 若决定保留 Session 级 gate 观测需求，改为日志/metrics，不进队列。
   - `QueueItemKind.GATE_FEEDBACK` 枚举值同步删除（fail-closed，无兼容层）。
5. **修正断言错误行为的测试**：
   - `tests/agent/test_completion_gates_e02.py:210-216`
     （断言 `_pending` 中残留 `GATE_FEEDBACK`）改为断言 gate 反馈进入了
     run 的消息流（从 RunLog / step views 验证）；
   - `tests/agent/test_main_agent_ab01.py:82-95` 的 enqueue/ack 单测保留
     （原语行为不变）。
6. **文档同步**：`CONTEXT.md` 中统一队列的表述若涉及 gate feedback 入队，
   改为"gate 反馈直接进入 live run 循环队列"。

## 验收标准

- 新回归测试全绿；`tests/agent/`、`console/tests/` 全量通过。
- `grep -rn "_process_pending_worker_reports\|on_gate_feedback" agiwo console` 无残留。
- `MainAgent` 不再有独立 `_state` 字段。

## 风险与回退

- drain 循环里 `_start_new_run` 后立即处理后续项，依赖
  "`start_prevalidated` 返回后 handle 立即 `is_active`"——
  该前提成立（task 尚未被调度即非 done），但需在测试中覆盖
  "队列中连续 USER_INPUT + WORKER_REPORT" 的场景。
- `deliver_worker_report` 的调用方 `WorkerService._monitor_async_worker`
  不感知本次改动，接口签名不变。
