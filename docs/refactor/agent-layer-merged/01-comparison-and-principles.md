# Agent 层整合方案：对比与原则

> 状态说明（2026-03）：这套 merged 方案已经在代码中落地。当前目录和边界请以 `agiwo/agent/lifecycle/`、`agiwo/agent/engine/`、`AGENTS.md` 和 `agiwo/agent/README.md` 为准。
> 范围：`agiwo/agent/**`
> 目标：整合 `docs/refactor/agent-layer/` 与 `docs/refactor/agent-layer-by-opus/` 两套方案，在保持行为不变的前提下，得到一版更清晰、更克制、代码量更少的目标设计。

## 1. 为什么需要第三版方案

现有两套方案各有明显优点，也各有明显过度设计风险：

- `agent-layer-by-opus/`
  - 优点：目录语义清晰，owner 边界表达得很好，敢于删除中间层。
  - 风险：`Runner + Executor` 过早合并、`RunRecorder + EventBus` 拆分过度、child type 简化过猛。
- `agent-layer/`
  - 优点：抓住了执行主链路真正的问题，强调 phase pipeline、状态收敛、行为快照测试。
  - 风险：`phases.py / effects.py / policies.py / state_store.py` 这类抽象如果一次性上齐，文件和概念会反弹。

这份整合版的核心判断是：

1. **目录结构应该更像 `by-opus`**：按 `lifecycle/` 和 `engine/` 组织，阅读路径明显更短。
2. **执行骨架应该吸收 `agent-layer` 的 phase 思想**：但 phase 只体现在 `ExecutionEngine` 的显式方法里，不额外引入 phase framework。
3. **真正要保留的单 owner 是 `RunRecorder`**：不能为了“event-driven”把当前已经稳定的写入顺序打散。

---

## 2. 两套方案的对比结论

| 维度 | `agent-layer` | `agent-layer-by-opus` | 整合结论 |
| --- | --- | --- | --- |
| 外部 API | 保持不变 | 保持不变 | 保持不变 |
| 目录组织 | 仍偏执行骨架视角 | `lifecycle/ + engine/ + recording/` 更清晰 | 采用 `lifecycle/ + engine/` |
| 执行控制 | phase pipeline | `ExecutionEngine` 合并 setup/loop/teardown | 保留 `Orchestrator + Engine` 双层 |
| `RunRecorder` | 保持单 owner，但提议 event 化 | 拆成 Recorder + EventBus | 保持单 owner，不拆 EventBus |
| 状态模型 | `RunStateStore` 聚合子状态 | `RunState` 方法封装 | 保留 `RunState`，但做内聚化封装 |
| child 定义 | 强调复用 | 强调减类型 | 保留 public spec + internal snapshot，不做激进合并 |
| 抽象强度 | 偏强 | 偏强 | 只保留能真实删代码的抽象 |

---

## 3. 当前代码的硬约束

以下行为在整合版中必须保持：

1. `Agent.start()` 仍同步返回 `AgentExecutionHandle`，执行继续在后台 task 中推进。
2. `AgentExecutionHandle` 仍是唯一 live execution surface，继续拥有 `stream()/wait()/steer()/cancel()`。
3. root run 与 child run 仍共享 `AgentSessionRuntime` 的 session 语义。
4. `RunRecorder` 仍是 run/step lifecycle 的唯一写入 owner。
5. scheduler 仍通过窄接口依赖 agent，而不是直接穿透内部执行实现。
6. tool gate / timeout / cache / concurrency-safe 行为保持不变。
7. stream item 顺序、step sequence 分配、termination reason 触发语义保持不变。

---

## 4. 整合后的设计原则

### 4.1 按“生命周期语义”组织目录，而不是按历史演进痕迹组织

最终内部结构分为两个主包：

- `lifecycle/`
  - 负责 definition、resource、session、root/child run orchestration。
- `engine/`
  - 负责单次 run 的 prepare、llm/tool/compact、termination、recording。

这比现在的 `inner/` 更清晰，也比 `recording/ + engine/ + lifecycle/` 三个内部子包更克制。

### 4.2 保留两层执行 owner：`ExecutionOrchestrator` + `ExecutionEngine`

不直接把 `Runner` 和 `Executor` 粗暴合成一个类，原因是：

- `start()` 的同步返回 handle；
- root session runtime 的创建；
- active root execution 的注册；
- child run 复用 parent session runtime；

这些都是“执行生命周期管理”，与单次 run 的 loop 本身不同。

因此最终结构是：

- `ExecutionOrchestrator`
  - 负责 root/child 的 session、task、handle、resource 关联。
- `ExecutionEngine`
  - 负责一次 run 的完整 pipeline：prepare -> before_run -> loop -> after_run -> finalize。

### 4.3 `RunRecorder` 继续保持单 owner，不引入 EventBus

当前系统已经形成了稳定契约：

- 先更新 state；
- 再写 storage；
- 再 trace；
- 再 hooks / observers；
- 最后 publish stream item。

这个顺序是行为的一部分，不应该被一个额外的 EventBus 打散。

整合版只做两件事：

1. 让 `RunRecorder` 更薄，尽量不持有重复上下文。
2. 让所有 lifecycle 写入都继续只经过 `RunRecorder`。

### 4.4 phase 思想只进入 `ExecutionEngine`，不外扩成框架

最终主循环应该像这样：

```python
async def execute(...):
    state = await self._prepare_state(...)
    await self._before_run(...)
    while not state.is_terminal:
        await self._maybe_resume_pending_tools(state)
        await self._maybe_compact(state)
        step, llm = await self._llm_step(state)
        await self._maybe_run_tools(state, step)
        self._check_limits(state, step, llm)
    await self._after_run(...)
    await self._finalize(...)
```

也就是说：

- 使用“显式 phase”；
- 不引入 `phases.py`；
- 不引入 `effects.py`；
- 不引入通用 `Policy` 基类，除非后续真的出现第二套实现。

### 4.5 child 类型只做“必要的最小简化”

不采用 “4 -> 2” 的激进压缩，也不维持今天这种跨文件跳转成本。

整合版建议：

- 保留 public `ChildAgentSpec`
- 保留 `ResolvedExecutionDefinition`
- 将 scheduler child 模板构造结果保留为独立 internal materialized type
- 删除单独的 `ChildDefinitionInputs` 文件级中转层
- 如果仍需中间结果，只保留同文件 private helper / private dataclass

这能减少跳转，同时不混淆“覆写输入”和“构造产物”。

---

## 5. 这版方案明确不做什么

1. 不做 EventBus / pub-sub 化 recorder。
2. 不做大规模 policy framework。
3. 不做 root/child/scheduler child 三套并存执行骨架。
4. 不为减少 LOC 而压缩 runtime types、storage adapter、trace adapter。
5. 不为了“减少文件数”而把 lifecycle、engine、tool runtime 全揉成新的 God object。

---

## 6. 最终追求的结果

这版方案追求的是三件事同时成立：

1. **目录更清楚**：像 `by-opus` 那样，一眼能看出 definition、resource、execution 各放哪。
2. **执行链更短**：像 `agent-layer` 那样，主流程读起来是显式 phase，而不是 helper 跳转图。
3. **代码更少**：通过删除 `execution_bootstrap.py`、`attach_state()`、重复 child materialization 逻辑、重复 assembly 透传层来减代码，而不是靠堆新抽象“重写”。
