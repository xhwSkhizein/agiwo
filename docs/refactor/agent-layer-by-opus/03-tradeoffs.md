# Agent Layer — Trade-Off Analysis

> 状态说明（2026-03）：本文分析的是 `by-opus` 方案内部的权衡，保留作为历史讨论材料。最终实现采用的是 `merged` 方案中的落点：保留 `ExecutionOrchestrator + ExecutionEngine` 双层，并维持 `RunRecorder` 单 owner。
> 每一个重构决策都有代价。本文对 `02-architecture-design.md` 中的每个主要变更进行 Trade-Off 分析。

---

## Decision 1: Runner + Executor → ExecutionEngine 合并

### Option A: 保持现状 (Runner + Executor 分离)

| Pros | Cons |
|---|---|
| 职责分离：Runner = lifecycle, Executor = loop | 边界模糊：RunRecorder 在两者间共享 |
| 更小的单个文件 | 参数传递多 (executor.execute 有 10+ 参数) |
| Runner 可独立测试 lifecycle 逻辑 | 调用深度 8 层，难以追踪 |

### Option B: 合并为 ExecutionEngine ✅ (推荐)

| Pros | Cons |
|---|---|
| 调用深度从 8 减到 4 | 单文件 ~450 LoC (可控范围) |
| 消除 Runner ↔ Executor 参数传递 | 合并后方法较多 (~15 methods) |
| 一个地方理解整个执行流程 | 如果未来 lifecycle 逻辑暴增需要再拆 |
| 统一测试入口 | — |

### 推荐理由

Runner 的 lifecycle 逻辑 (before-run hooks, memory retrieval, user step creation) 本质上是执行流程的 **前置阶段**，和 Executor 的 "主循环" 是同一个 pipeline 的不同 phase。把它们放在一个类里用 `_setup_run()` → `_run_loop()` → `_finalize()` 三阶段方法组织，比分布在两个类里更直观。

450 LoC 是一个单文件的合理上限。如果未来需要拆分，可以按 phase 提取 mixin 或 helper function，而不需要回到 Runner + Executor 的双类模式。

### 风险评估

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ExecutionEngine 膨胀 | Medium | Medium | 按 phase 提取 helper functions |
| 合并期间引入 bug | Low | High | Phase 2 逐步迁移，每步跑测试 |

---

## Decision 2: RunRecorder → Recorder + EventBus 拆分

### Option A: 保持 RunRecorder 现状

| Pros | Cons |
|---|---|
| 所有录制逻辑在一处 | 6 种职责违反 SRP |
| 无需新抽象 | 每次添加 observer 类型需改 RunRecorder |
| 现有测试不变 | 测试需要 mock 5 种依赖 |

### Option B: 拆出 EventBus ✅ (推荐)

| Pros | Cons |
|---|---|
| SRP：Recorder = storage+state, EventBus = side effects | 多一个类和文件 |
| 新增 observer 只改 EventBus | EventBus 构造需要注入多个依赖 |
| Recorder 测试只需 mock storage | 多一层间接调用 |
| EventBus 可独立测试 | — |

### Option C: 完全 Event-Driven (Publish-Subscribe)

| Pros | Cons |
|---|---|
| 最大解耦 | 过度设计：当前 observer 类型只有 4-5 种 |
| 添加 listener 零侵入 | 事件顺序难以保证 |
| — | 调试困难：side effect 在 subscriber 中 |
| — | 性能开销：event dispatch |

### 推荐理由

Option B 在解耦和复杂度之间取得平衡。当前 observer 类型固定且数量少（trace, hooks, stream, step_observers），用一个显式的 EventBus 类比 pub-sub 系统更直接。如果未来 observer 数量增长到 10+ 种，可以升级到 Option C。

---

## Decision 3: Child Spec 类型从 4 → 2

### Option A: 保持 4 种类型

| Pros | Cons |
|---|---|
| 每层有明确的领域类型 | 4 层类型转换，每层几乎 1:1 |
| 变更隔离 | ChildDefinitionInputs 和 ChildAgentSpec 重复 |
| — | 新增字段需要改 4 个地方 |

### Option B: 减到 2 种 ✅ (推荐)

保留 `ChildAgentSpec` (public) + `ResolvedExecutionDefinition` (internal)

| Pros | Cons |
|---|---|
| 新增字段只改 2 处 | ChildAgentSpec 承担多一点语义 |
| 代码减少 ~60 LoC | Scheduler clone 需要 ChildAgentSpec 扩展 |
| 类型转换路径清晰 | — |

### Option C: 只保留 1 种类型

| Pros | Cons |
|---|---|
| 最简 | ChildAgentSpec 会暴露内部实现细节 |
| — | Public API 中混入 internal fields |
| — | 违反信息隐藏原则 |

### 推荐理由

`ChildAgentSpec` 是 public API，`ResolvedExecutionDefinition` 是 internal immutable snapshot，两者语义不同，应该保留。中间的 `ChildDefinitionInputs` 是纯粹的传递层，`AgentCloneSpec` 是 scheduler 的变体——这两个可以合并到 `ChildAgentSpec` 中。

---

## Decision 4: DefinitionRuntime 瘦身

### Option A: 保持现状 (314 LoC all-in-one)

| Pros | Cons |
|---|---|
| 所有定义逻辑集中 | 违反 SRP：组装 + 管理 + 快照 |
| 无拆分成本 | 314 LoC 且持续增长 |

### Option B: 提取组装逻辑到 assembly.py ✅ (推荐)

| Pros | Cons |
|---|---|
| DefinitionRuntime 瘦身到 ~180 LoC | assembly.py 增长到 ~130 LoC |
| 组装逻辑和运行时管理分离 | 需要同时看两个文件理解初始化 |
| DefinitionRuntime 的构造参数变为已组装对象 | — |

### Option C: 彻底拆分为 ToolsManager + HooksManager + PromptManager + DefinitionSnapshot

| Pros | Cons |
|---|---|
| 每个管理器 ~80 LoC | 4 个新类 = 4 个新文件 |
| 最大 SRP | 过度拆分，组装复杂度转移到 Agent.py |
| — | 难以理解整体 |

### 推荐理由

Option B 在瘦身和复杂度之间平衡。关键洞察：**组装（assembly）是一次性行为**，**管理（runtime）是持续行为**。把一次性逻辑移到工厂函数中，让 DefinitionRuntime 专注于运行时快照生成，符合 SRP 且不引入过多新抽象。

---

## Decision 5: RunState 封装化

### Option A: 保持裸属性包

| Pros | Cons |
|---|---|
| 简单直接 | 任何持有者都能随意修改 |
| 无学习成本 | 状态变更无法追踪 |
| — | track_step 逻辑分散在多处 |

### Option B: 封装为方法 ✅ (推荐)

| Pros | Cons |
|---|---|
| 状态变更有明确入口 | 比直接属性访问略冗长 |
| 内部可以加验证/日志 | 需要更新所有调用方 |
| 编译器可检测非法访问 | — |

### Option C: 完全不可变 (immutable state + copy-on-write)

| Pros | Cons |
|---|---|
| 函数式风格，无副作用 | 每次变更都产生新对象 |
| 天然线程安全 | 性能开销 (messages list 拷贝) |
| — | 与现有代码风格不一致 |

### 推荐理由

Option B 提供了足够的封装保护，同时保持可变性（agent 执行本质上是有状态的过程）。完全不可变在 messages 列表频繁修改的场景下性能不可接受。

---

## Decision 6: scheduler_port.py 精简

### Option A: 保持完整 Adapter (124 LoC)

| Pros | Cons |
|---|---|
| 完全解耦 scheduler 和 agent | 124 LoC 几乎全是 1:1 转发 |
| Protocol 强制类型检查 | 每次 Agent API 变更需要同步更新 adapter |

### Option B: 精简为 mixin 或 thin wrapper ✅ (推荐)

| Pros | Cons |
|---|---|
| ~60 LoC | 耦合稍增 |
| 减少维护负担 | — |
| 保留 Protocol 但减少转发 | — |

### Option C: 删除 adapter，scheduler 直接依赖 Agent

| Pros | Cons |
|---|---|
| 最简，零样板代码 | scheduler → agent 直接依赖 |
| — | 违反 `scheduler -> agent` 依赖方向的 Port 抽象 |
| — | 测试 scheduler 时需要 mock 完整 Agent |

### 推荐理由

Port adapter 的存在是为了 scheduler 层的解耦，但当前实现过于冗长。通过保留 Protocol 定义但简化 adapter 实现（使用 `__getattr__` 代理或只覆写有差异的方法），可以减少 50% 代码。

---

## Decision 7: 是否重构 storage/ 层

### 结论：暂不重构 ✅

| Pros of not touching | Rationale |
|---|---|
| 稳定且经过测试 | SQLite/MongoDB 实现虽冗长但正确 |
| 样板代码是 storage 层的固有特性 | 抽象减少的收益不值得引入新复杂度 |
| 与 inner/ 重构正交 | 可以作为后续独立 PR |

如果未来要优化，方向是：
- 提取 `_ensure_connection + try/except + log` 模式为基类 template method
- 使用 SQLAlchemy async 或 beanie 减少手写 SQL/Mongo 查询

---

## Decision 8: 是否重构 runtime/ 和 prompt/ 层

### 结论：保持不变 ✅

- `runtime/` 是公开数据模型，修改影响面最大
- `prompt/` 职责清晰、规模小 (~280 LoC)
- 两者都不是当前的痛点

---

## Summary Matrix

| Decision | Recommendation | Code Reduction | Risk | Effort |
|---|---|---|---|---|
| Runner+Executor → Engine | ✅ Merge | ~190 LoC | Medium | Medium |
| RunRecorder → Recorder+EventBus | ✅ Split | ~55 LoC | Low | Low |
| Child spec 4→2 types | ✅ Simplify | ~60 LoC | Low | Low |
| DefinitionRuntime slim | ✅ Extract assembly | ~130 LoC | Low | Medium |
| RunState encapsulate | ✅ Methods | ~0 (style change) | Low | Low |
| scheduler_port slim | ✅ Simplify | ~60 LoC | Low | Low |
| storage/ refactor | ❌ Skip | — | — | — |
| runtime/prompt/ refactor | ❌ Skip | — | — | — |
| **Total** | — | **~495 LoC** | — | — |

> 加上文件合并和删除中间类型带来的结构简化，总体预期从 **63 files / ~5800 LoC** 降至 **~45 files / ~4200 LoC**，减少约 **28%**。
