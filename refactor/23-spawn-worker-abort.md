# Task 23：`SpawnWorkerTool` 同步等待可中断

Phase 2 · 独立任务 · 修复 review 问题 #9（abort_signal 被丢弃、无超时）

## 现状与问题

`agiwo/agent/worker_tools.py:63-69`：

```python
async def execute(self, parameters, context, abort_signal=None) -> ToolResult:
    del abort_signal
```

同步派生（`sync=true`）路径：`WorkerService.spawn_worker`
（`worker.py:85-88`）→ `scheduler.get_worker_report(worker_id)` →
`Scheduler.wait_for`，**无超时、无中断点**。后果：

- 工具批次层面的 abort（用户取消当前 tool batch、上游 abort_signal
  触发）无法打断同步 Worker 等待，`bash` 等其他工具都尊重
  abort_signal，此处是行为不一致的洞；
- 唯一逃生通道是 `MainAgent.cancel` 整体取消——粒度过粗。

## 目标设计

### abort 与等待竞速

`SpawnWorkerTool.execute` 尊重 abort_signal，在 WorkerService 层实现
"等待 vs 中断"竞速（工具层保持薄）：

```python
# worker.py — spawn_worker 的 sync 分支
if sync:
    report = await self._wait_report_abortable(
        state.worker_id, abort_signal=abort_signal
    )
    ...

async def _wait_report_abortable(
    self, worker_id: str, *, abort_signal: AbortSignal | None
) -> str | None:
    """Wait for the worker report; on abort, cancel the worker and return None."""
    wait_task = asyncio.ensure_future(self._scheduler.get_worker_report(worker_id))
    if abort_signal is None:
        return await wait_task
    abort_task = asyncio.ensure_future(abort_signal.wait())  # 以源码 API 为准
    done, _ = await asyncio.wait(
        {wait_task, abort_task}, return_when=asyncio.FIRST_COMPLETED
    )
    if wait_task in done:
        abort_task.cancel()
        return wait_task.result()
    wait_task.cancel()
    await self._scheduler.cancel_worker(worker_id, "Aborted by parent tool batch")
    return None
```

工具层（`worker_tools.py`）：

- `execute` 不再 `del abort_signal`，透传给
  `WorkerService.spawn_worker(..., abort_signal=abort_signal)`；
- 同步分支收到 `None` 报告时返回
  `ToolResult.aborted(...)`（生产代码统一走
  `ToolResult.success()/failed()/aborted()/denied()` 构造器）。

**执行前核对**：`AbortSignal`（`agiwo/utils/abort_signal.py`）的等待
原语名称（`wait()` / 内部 Event / 回调注册），以源码为准替换伪代码；
若只有同步 `is_aborted` 轮询接口，可退化为
`asyncio.wait_for(wait_task, timeout=poll)` 循环 + 每轮检查。

### 超时不做

不给工具加 `timeout` 参数：run 级别已有 `timeout_at` / limits 体系，
工具级超时会制造第二套超时语义。abort 打通后，上游任何超时机制
（run 超时触发 abort）自然覆盖此处。

### 异步分支不变

`sync=false` 路径本就立即返回，monitor task 的取消已由
`cancel_all_workers`（`worker.py:96-110`）覆盖，不动。

## 任务拆分

1. **先写回归测试**（`tests/agent/test_worker_d04_abort.py`，新文件，
   复用 `tests/agent/worker_test_helpers.py` 的 stub scheduler）：
   - 同步派生后触发 abort_signal，断言：`execute` 在有限时间内返回、
     结果为 aborted、stub 收到 `cancel_worker` 调用；
   - abort 未触发时行为与现有 `test_worker_d02.py` 一致（不回归）。
2. **`WorkerService.spawn_worker` 增加 `abort_signal` 形参**与
   `_wait_report_abortable`；注意 `WorkerSchedulerPort` 协议不需要变
   （竞速在 service 层完成）。
3. **`SpawnWorkerTool.execute` 透传 abort_signal**，同步分支处理
   `None` 报告 → `ToolResult.aborted`。
4. **核对 stub**：`worker_test_helpers.py` 的端口 stub 如
   `get_worker_report` 立即返回，补一个可 hang 的变体供 abort 测试用。

## 验收标准

- 新测试全绿且无泄漏 task 警告（`asyncio` debug 模式跑一遍）；
- `tests/agent/test_worker_d01-d03` 不回归。

## 风险

- `Scheduler.wait_for` 内部实现若不容忍外部 `wait_task.cancel()`
  （如持有不可重入锁），需核对 `engine.py` waitset 实现；
  取消一个"等待者"不应影响 waitset 中其他等待者。
