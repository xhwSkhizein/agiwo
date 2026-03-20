# Agent 层重构洞察（保持行为不变）

> 范围：`agiwo/agent/**`（含 public facade、inner runtime、runtime types、prompt/trace/storage glue）。
> 目标：在不改变用户可见能力与执行语义的前提下，提升可维护性、可读性、可扩展性，并减少总代码体量。

## 1) 当前实现我确认的“做得对”的地方（必须保留）

1. **Definition / Resource / Execution 三层职责已经初步分离**：
   - `AgentDefinitionRuntime`（定义域）
   - `AgentResourceOwner`（资源生命周期）
   - `AgentExecutionHandle` + `AgentRunner` + `AgentExecutor`（执行域）
2. **`RunRecorder` 成为唯一记账 owner**，run/step/stream/trace 聚合写入路径收敛，是非常正确的方向。
3. **`AgentSessionRuntime` 统一了会话级序列号、steering、stream 订阅和 abort 信号**，root/child 能共享 session 语义。
4. **Tool 执行已抽象为 runtime adapter（`RuntimeToolLike`）+ `ToolRuntime`**，支持 gate/cache/并发安全策略。

这些能力是重构时的“硬约束”，不能为了“简化代码”而破坏。

---

## 2) 当前代码复杂度热点（导致维护成本高）

### 2.1 控制流分散（读代码要跳很多层）

一次 root run 的关键路径要在多个文件之间跳转：

```text
agent.py(start)
  -> inner/runner.py(start_root/_execute_workflow)
    -> inner/executor.py(execute/_run_loop/_run_cycle)
      -> inner/execution_bootstrap.py
      -> inner/llm_handler.py
      -> inner/tool_runtime.py
      -> inner/termination_runtime.py
      -> inner/compaction/*
```

问题不是“分层多”，而是 **phase 语义没有变成显式状态机**，导致 debug 某个 bug 时要反复拼 mental model。

### 2.2 状态字段过多且分散

- `RunState` 本身字段较多（token、step、pending tool call、compact 元数据等）。
- 同时 `AgentRunContext`、`AgentSessionRuntime`、`RunRecorder`、`ExecutionTerminationRuntime` 各自也持有一部分“运行时真相”。
- 结果是“变更一个语义（例如中止/超时）要改多个 owner”。

### 2.3 Hook / Observer / Stream 的 fanout 点过多

同一生命周期事件可能同时触发：
- hooks
- step observers
- run_step_storage
- trace runtime
- session subscribers

目前借由 `RunRecorder` 已有收敛，但依旧存在调用时机分布在 runner / executor / tool runtime 的现象，可继续收束为统一 phase hook。

### 2.4 “功能增长 vs 模块增长”不成比例

目前 Agent 子目录 63 个 Python 文件、约 7.9k LOC。对单一领域（agent runtime）来说偏碎片化，认知负担高。

---

## 3) 必须保持不变的行为契约（回归基线）

以下语义在重构后必须逐条保持：

1. `Agent.start()` 立即返回 `AgentExecutionHandle`，执行在后台 task 继续推进。
2. `run()/run_stream()` 仍为 facade convenience API，不破坏 `start()+handle` 主语义。
3. `derive_child_spec()` 仍返回纯覆写数据对象，不携带 live runtime。
4. `run_child()` 仍复用 parent session runtime（sequence/steering/trace 链路）。
5. `RunRecorder` 仍是 run/step lifecycle 唯一写入 owner（storage + trace + stream fanout）。
6. `steer()` 仍属于执行实例（`AgentExecutionHandle`），而不是 `Agent` 模板对象。
7. tool gate / timeout / cache / concurrency-safe 行为不变。
8. termination reason 的集合及触发条件语义不变（含 `SLEEPING`、`CANCELLED` 等）。

---

## 4) 我建议的重构原则

1. **“显式 phase”优先于“隐式 helper 链”**：把 execution loop 改为可枚举阶段。
2. **“状态收敛”优先于“模块继续拆分”**：减少同时持有状态的 owner 数量。
3. **“边界对象稳定”优先于“内部对象自由演化”**：
   - 外部稳定：`Agent`、`AgentExecutionHandle`、runtime types
   - 内部可演化：planner/phase engine/effect handlers
4. **“可替换策略注入”优先于“if/else 组合扩展”**：compaction、termination、tool scheduling 采用 policy objects。
5. **“删代码”是第一目标，不是附带收益**：优先合并薄封装、去重复参数透传。

---

## 5) 预期收益（定性）

- 新同学定位“run 为何结束”只看 phase state + recorder event log。
- 新增 termination 规则或 tool 执行策略不需要改动主循环结构。
- root/child/scheduler-child 的定义派生逻辑复用度更高。
- 总体文件数和跨文件调用链减少，阅读路径缩短。
