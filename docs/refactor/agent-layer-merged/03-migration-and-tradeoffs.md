# Agent 层整合方案：迁移与权衡

> 状态说明（2026-03）：本文中的迁移步骤和权衡已经基本执行完成。保留此文是为了说明为何最终落在 `lifecycle/ + engine/`，而不是更激进或更框架化的拆分。

## 1. 总体判断

这版整合方案的目标不是“提出一套更抽象的架构词汇”，而是：

1. 让目录结构像 `by-opus` 一样清晰；
2. 让执行主干像 `agent-layer` 一样可读；
3. 同时避免两边都会出现的抽象反弹。

因此迁移策略必须遵守一个原则：

> **先删薄层和重复层，再考虑更大形态调整。**

---

## 2. 优先级排序

### P0：必须做

1. 建立行为快照测试
2. 删除 `execution_bootstrap.py`
3. 删除 `RunRecorder.attach_state()`
4. 把 before/after run 流程并入 `ExecutionEngine`
5. 形成 `lifecycle/ + engine/` 目录结构

### P1：建议做

1. `definition_runtime.py` + `definition.py` 收口到 `lifecycle/definition.py`
2. `summarizer.py` 并入 `engine/termination.py`
3. child materialization 路径去重
4. `assembly.py` 改成真正的组装入口

### P2：观察后再做

1. `scheduler_port.py` 进一步瘦身
2. `tool_runtime.py` 最小策略注入
3. `RunState` 进一步拆成更细的 internal accumulators

---

## 3. 为什么不采用更激进的拆分

### 3.1 为什么不做 `Recorder + EventBus`

代价大于收益：

- 当前 `RunRecorder` 的顺序是行为契约，不只是内部实现。
- hooks / observer / trace / stream fanout 都严格依赖提交顺序。
- 增加 EventBus 会多一个调试层，也会增加“谁负责顺序保证”的认知成本。

结论：

- 保留 `RunRecorder` 单 owner；
- 如需提取重复逻辑，只在 recorder 内部抽私有 helper。

### 3.2 为什么不做 `phases.py / effects.py / policies.py`

这些抽象在只有一套默认实现时会先增加文件数和概念数。

当前最真实的问题不是“phase 不可扩展”，而是：

- phase 没有直接体现在主循环里；
- prepare 流程横跨多个文件；
- state 绑定方式不自然。

结论：

- 先把 phase 变成 `ExecutionEngine` 的显式私有方法；
- 后续只有在第二套实现出现时，再最小化抽象出 policy。

### 3.3 为什么不把 `Runner + Executor` 完全合成一个类

`Runner` 当前有两类职责：

1. 真正属于 run pipeline 的 before/after 流程。
2. 真正属于 lifecycle 的 handle / session / task 管理。

整合版只吸收第 1 类，保留第 2 类为 `ExecutionOrchestrator`。

这样才能同时满足：

- `Agent.start()` 立即返回 handle；
- root run 可注册 active execution；
- child run 继续复用 parent session runtime。

---

## 4. 推荐迁移步骤

### Step 0：冻结行为

补齐或固定以下回归基线：

1. root run 的 stream item 顺序
2. root run 的 step sequence
3. child run 复用 parent session runtime
4. handle 的 `wait()/stream()/steer()/cancel()` 契约
5. termination reason 触发时机
6. scheduler 观察 root stream 的行为

### Step 1：只做目录重组，不改语义

先完成文件迁移：

- `inner/runner.py` -> `lifecycle/orchestrator.py`
- `inner/session_runtime.py` -> `lifecycle/session.py`
- `inner/resource_owner.py` -> `lifecycle/resource_owner.py`
- `inner/context.py` -> `engine/context.py`
- `inner/run_state.py` -> `engine/state.py`
- `inner/run_recorder.py` -> `engine/recorder.py`
- `inner/executor.py` -> `engine/engine.py`

这一步优先保证 import 与测试可通过。

### Step 2：删除 `execution_bootstrap.py`

把以下逻辑并入 `ExecutionEngine._prepare_state()`：

- compact metadata 加载
- existing steps 加载
- messages assemble
- state 初始化
- recorder 初始化
- compaction runtime 初始化

完成后删除：

- `PreparedExecution`
- `prepare_execution()`
- `RunRecorder.attach_state()`

### Step 3：把 before/after run 逻辑收进 `ExecutionEngine`

从 orchestrator 中移入 engine：

- `on_before_run`
- `on_memory_retrieve`
- user step commit
- `on_after_run`
- memory write
- run finalize / fail handling

此时 orchestrator 应只剩 lifecycle wiring。

### Step 4：收口 definition 域

把以下内容整合进 `lifecycle/definition.py`：

- `AgentDefinitionRuntime`
- `ResolvedExecutionDefinition`
- scheduler child materialization helper

并删除：

- 单独的 `inner/definition.py`
- 独立 `ChildDefinitionInputs` 跳转层

### Step 5：小幅清理

1. 合并 `summarizer.py` 到 `termination.py`
2. 让 `assembly.py` 真正承担组装职责
3. 更新 README / AGENTS.md 中的 agent 结构描述

---

## 5. 风险与缓解

| 风险 | 表现 | 缓解 |
| --- | --- | --- |
| 行为漂移 | stream item 顺序或 termination reason 变化 | 先补 golden tests，再做结构迁移 |
| 新 God object | `ExecutionEngine` 膨胀过快 | 只吸收 run pipeline，不接手 lifecycle 管理 |
| 抽象反弹 | 新增过多 phase/policy/event 类型 | 坚持“没有第二个用例就不抽象” |
| 迁移窗口认知分裂 | 新旧路径并存太久 | 每一步落地后立即删除旧 helper/旧文件 |

---

## 6. 成功标准

重构完成后，应同时满足：

1. 读 root run 主路径时，只需顺着 `Agent -> Orchestrator -> Engine -> Recorder` 看。
2. 排查 child 派生问题时，只需看 `lifecycle/definition.py`。
3. 排查 handle/session/resource 生命周期问题时，只需看 `lifecycle/`。
4. `engine/engine.py` 主循环能在 50 行以内解释清楚。
5. 不新增 EventBus、phase framework、policy framework 这类通用基础设施。

---

## 7. 最终建议

如果只允许做一轮重构，我建议优先拿下这三件事：

1. **目录重组为 `lifecycle/ + engine/`**
2. **删除 `execution_bootstrap.py` 与 `attach_state()`**
3. **把 before/after run 收进 `ExecutionEngine`，把 `ExecutionOrchestrator` 压薄为 `ExecutionOrchestrator`**

这三步完成后，即使后面什么都不再做，`agent` 层的清晰度也会明显高一个档次，而且不会因为“为了更优雅”引入新的维护负担。
