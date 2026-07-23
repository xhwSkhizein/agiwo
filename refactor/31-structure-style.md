# Task 31：结构与风格收尾

Phase 3 · 建议在 Phase 1/2 合入后统一排期 · 覆盖 review 低优先级项

本文档聚合五个小任务，可拆成独立 commit 逐个完成。

## 31.1 WorkerService 依赖窄回调，消除循环 import

**现状**：`MainAgent.__init__`（`main_agent.py:88-93`）用函数内局部
import 引入 `WorkerService` / `SpawnWorkerTool`，因为 `worker.py`
反向依赖 `MainAgent`（仅用于 `TYPE_CHECKING` 与
`_deliver_report_to_main` / `sync_parent_idle` / 读取
`agent_id` / `session_id` / `agent`）。

**方案**：`WorkerService` 改为依赖一个窄参数集而非整个 MainAgent：

```python
class WorkerService:
    def __init__(
        self,
        *,
        agent_id: str,
        session_id: str,
        agent: Agent,
        scheduler: WorkerSchedulerPort,
        deliver_report: Callable[[str, str], Awaitable[None]],  # (run_id, report)
    ) -> None: ...
```

MainAgent 构造时传 `deliver_report=self.deliver_worker_report`。
收益：循环依赖消失（局部 import 可提回文件顶部）、WorkerService
可脱离 MainAgent 单测（现有 `worker_test_helpers.py` 也能简化）。

**验收**：`main_agent.py` 顶部 import 全部在文件头；
`grep -n "PLC0415" agiwo/agent/main_agent.py` 零命中。

## 31.2 completion gate context 传参化

**现状**：`Agent.bind_completion_gate_context`（`agent.py:139-147`）
把 MainAgent 的关注点做成 Agent 的**可变实例状态**，与
"AgentConfig 只承载纯配置、不放 live object" 的既有原则同源冲突
（这里污染的是 Agent 实例而非 config，性质相同）。

**方案**：`completion_gate_context` 走执行入口参数：

- `start_prevalidated(...)` / `continue_completed_run(...)` 增加
  `completion_gate_context: CompletionGateContext | None = None` 形参，
  透传到 `execute_run`；
- 删除 `bind_completion_gate_context` 与 `_completion_gate_context`
  实例字段；MainAgent 在每次开 run / 续 run 时显式传入。

**注意**：先完成 Task 22（gate context 里的 `on_gate_feedback` 已在
Task 11 中删除），此时 `CompletionGateContext` 只剩
`active_worker_ids` 一个字段，评估是否直接传
`active_worker_ids: Callable[[], frozenset[str]]` 而砍掉 dataclass——
若语义门注入将来需要挂载点则保留 dataclass。执行时二选一并记录理由。

**验收**：`grep -rn "bind_completion_gate_context" agiwo console tests`
零命中；`tests/agent/test_completion_gates_*` 全绿。

## 31.3 `RunLoopOrchestrator` mixin → 组合（评估后决定）

**现状**：`run_loop.py:53-56` 通过继承
`RunLoopFinalizationOps` / `RunLoopCompactionOps` 两个 mixin 组装
orchestrator，mixin 隐式依赖 `self.context / self.runtime / self.writer`。

**立场**：当前两个 mixin 还在可控范围，**不强制立即重写**；本任务是
设一条护栏并做一次评估：

1. 评估把 `RunLoopFinalizationOps` / `RunLoopCompactionOps` 改为
   显式构造的协作对象（`FinalizationOps(context, writer)`）的成本，
   若单纯是机械替换（预计是），执行之；
2. 无论是否重写，在 `run_loop.py` 模块 docstring 写明：
   **不再新增第三个 mixin**，新的循环阶段逻辑以组合对象进入。

**验收**：决策记录在本文件或 commit message；如执行重写，
`tests/agent/test_run_loop_contracts.py` 全绿。

## 31.4 session-history 双写投影回归测试

**现状疑点**：`MainAgent.accept` 在 run 进行中既把用户消息写入
session history（合成 `run_id=session-history-{sid}`，
`main_agent.py:153-160`）又 `enqueue_message` 进 live run。下一个 run
以 `session_id + agent_id` 重建 messages 时，该消息是否恰好出现一次，
取决于投影实现（`prepare_run_context` / `session_history.py`）。

**任务**：补一条投影语义测试（`tests/agent/test_main_agent_history_projection.py`）：

1. 开 run → run 进行中 accept 第二条消息 → run 结束；
2. accept 第三条消息开新 run；
3. 断言新 run 的初始 messages 中第二条消息**恰出现一次**。

若测试暴露双写重复，根因修复方向：投影层按
（合成 history run 的 user fact）与（live run 的 rebuilt messages）
去重，收口在 `session_history.py` / `run_bootstrap.py`——届时另开任务，
不在本任务内动投影逻辑。

**验收**：测试进库并绿（或红——转化为新的 Phase 1 级任务）。

## 31.5 杂项清理

- **import 顺序**：`run_loop.py:12-13`
  （`models.model_call` 在 `models.config` 之前）按 isort 规则修正；
  顺手 `uv run ruff check agiwo/agent/run_loop.py --select I`。
- **`MainAgent.subscribe` 的空迭代器**（`main_agent.py:329-333`）：
  `if False: yield` 写法可读性差，改为
  `return` 之前构造空 async generator 的惯用形式，或直接返回
  `None` 并让调用方判断——保持返回类型不变的前提下取前者。
- **`AgentSpec` 价值核对**：`spec.py` 目前只是 `AgentConfig` 的
  frozen 包装。保留（它是 ADR 0049 的词汇锚点），但在 docstring
  中标注"字段将随 spec 复用场景（多 Session 绑定同一 spec）扩展"，
  防止后来者误删。

## 建议执行顺序

31.4（可能升级为高优先级）→ 31.1 → 31.2 → 31.5 → 31.3
