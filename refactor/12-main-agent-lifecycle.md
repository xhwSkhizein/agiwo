# Task 12：MainAgent 生命周期加固 — 异常吞噬与 cancel 竞态

Phase 1 · 依赖 Task 11 · 修复 review 问题 #2（Task 异常无人 retrieve）、#3（cancel 期间 run 被"复活"）

## 现状与问题

### 问题 A：`_await_run_completion` 吞掉非 Cancelled 异常

`agiwo/agent/main_agent.py:197-202`：

```python
async def _await_run_completion(self, handle: AgentExecutionHandle) -> None:
    output: RunOutput | None = None
    try:
        output = await handle.wait()
    except asyncio.CancelledError:
        pass
```

`handle.wait()` 等待的是 `Agent._execute_root` 的 task；run 失败时
`RunLoopOrchestrator.execute_run`（`run_loop.py:134-136`）会 re-raise。
该异常从 `_completion_task`（后台 Task）逸出且无人 `await`/retrieve：

- 事件循环打印 `Task exception was never retrieved` 警告；
- **finally 块虽会执行**（handle 清理不受影响），但失败原因没有任何
  结构化日志，Console 侧无法归因。

### 问题 B：cancel 期间 drain 可"复活" run

`cancel()`（`main_agent.py:310-322`）的时序：

1. `cancel_all_workers` → `handle.cancel` → `await completion_task`；
2. `_await_run_completion` 的 finally 会触发队列 drain
   （Task 11 后为 `_drain()`）；
3. 若队列中有未投递的 `WORKER_REPORT`，drain 会走
   `continue_completed_run` **启动新 run**；
4. `cancel` 随后把 `_handle = None`——新 run 成为无人持有的孤儿，
   且用户"取消"的意图被违背。

Task 11 完成后此竞态依然存在（drain 本身不知道正处于 cancel 流程）。

## 目标设计

### A：显式记录 run 失败

```python
async def _await_run_completion(self, handle: AgentExecutionHandle) -> None:
    output: RunOutput | None = None
    try:
        output = await handle.wait()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.error(
            "main_agent_run_failed",
            session_id=self._session_id,
            run_id=handle.run_id,
            exc_info=True,
        )
    finally:
        ...
```

不向外传播：MainAgent 的失败事实已由 RunLog（`RunFailed` fact）承载，
这里只补 SDK 侧结构化日志（遵循 `docs/logging-guidelines.md` 的
`event_name + key=value` 格式）。

### B：cancel 流程对 drain 的显式抑制

引入 `_cancelling: bool` 标志：

```python
async def cancel(self, reason: str | None = None) -> None:
    self._cancelling = True
    try:
        # 现有取消序列：workers → handle → completion_task
        ...
        self._discard_pending_on_cancel(reason)
    finally:
        self._cancelling = False
```

配套规则：

- `_drain()` 循环入口检查 `self._cancelling`，为真则直接返回
  （不 ack、不开 run）；
- `_discard_pending_on_cancel`：清空 `_pending`，对每个被丢弃的
  `WORKER_REPORT` 记 `logger.info("worker_report_discarded_on_cancel", ...)`。
  **决策依据**：`cancel_all_workers` 已取消全部 Worker，不会再有新报告；
  用户显式取消后自动续跑残留报告违背取消意图。ADR 0049 的
  "resume the same Run" 针对的是正常完成后的异步报告，不覆盖 cancel 场景。
- cancel 结束后队列为空、状态为 IDLE（派生），下一次 `accept` 从干净
  状态开始。

### 备选方案（记录，不采纳）

- *保留报告待下次 accept 时投递*：让"取消"的语义变得模糊
  （取消了 run 却保留其副产物），且报告对应的 `run_id` 已终止，
  `continue_completed_run` 会因状态校验失败（`run_resume.py:128-131`
  要求 COMPLETED）——本就走不通，佐证丢弃是正确选择。

## 任务拆分

1. **先写回归测试**（`tests/agent/test_main_agent_cancel.py`，新文件）：
   - run 进行中投递异步 Worker 报告后立刻 `cancel()`，断言：
     cancel 返回后 `state is IDLE`、`peek_pending() is None`、
     没有新 run 被创建（`run_log_storage` 中无新增 `RunStarted`）；
   - 构造必然失败的 run（如 model stub 抛错），断言 `cancel`/等待后
     无未 retrieve 异常（`pytest` 开 `-W error::RuntimeWarning` 或
     捕获 event loop exception handler）且日志含 `main_agent_run_failed`。
2. **实现 A**：`_await_run_completion` 增加 `except Exception` 分支。
3. **实现 B**：`_cancelling` 标志 + `_drain` 入口检查 +
   `_discard_pending_on_cancel`。
4. **核对 `close()` 路径**：`close()`（`main_agent.py:341-346`）复用
   `cancel()`，确认 B 的改动对 close 幂等。
5. **核对 Console 消费方**：`SessionTurnService.cancel_if_active` 与
   feishu 渠道的 cancel 命令走的都是 `MainAgent.cancel`，无需改动，
   但在 `console/tests/test_scheduler_chat_api.py` 相应用例中补一条
   "cancel 后再 accept 能正常开新 run" 的断言。

## 验收标准

- 新回归测试全绿；`tests/agent/` 全量通过。
- 人工验证：`uv run pytest tests/agent -q 2>&1 | grep -c "never retrieved"` 为 0。

## 风险

- `_cancelling` 是进程内单协程标志，MainAgent 实例被跨事件循环使用时
  不成立——当前 Console 每 session 单实例 + 单 loop，前提成立；
  在 docstring 中写明该约束。
