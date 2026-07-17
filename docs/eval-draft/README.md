# Agiwo 通用 Benchmark Eval 方案草稿

状态：Draft

> **与 Objective 重构无关：** `docs/adr-plan` 的本次 Objective 优化执行方案明确不包含、不依赖、不跟进本目录。实现 Objective 时请忽略此处内容；本草稿若继续推进，应作为独立计划，不得假设已与 adr-plan 对齐。

这组文档用于把 Agent benchmark evaluation 变成 Agiwo 的一等实验能力。它描述实现顺序、模块接缝、数据契约、测试门槛和后续外部 benchmark 适配方式；不等同于已经批准的最终设计，也不代表当前代码已经包含这些模块。

## 阅读顺序

1. [00-proposal.md](./00-proposal.md)：整体目标、研究结论和目标架构。
2. [phase-00-contracts.md](./phase-00-contracts.md)：先冻结身份、版本、隔离和事实归属。
3. [phase-01-evaluation-core.md](./phase-01-evaluation-core.md)：实现通用评测内核。
4. [phase-02-deterministic-suites.md](./phase-02-deterministic-suites.md)：实现本地、确定性的工具调用和状态化工具套件。
5. [phase-03-graders-and-statistics.md](./phase-03-graders-and-statistics.md)：实现 grader、轨迹分析、稳定性指标和报告。
6. [phase-04-external-adapters.md](./phase-04-external-adapters.md)：按环境成本接入外部 benchmark。
7. [phase-05-console-and-operations.md](./phase-05-console-and-operations.md)：增加 CLI、Console 查询和运行控制。
8. [phase-06-hardening-and-release.md](./phase-06-hardening-and-release.md)：完成安全、复现、性能、文档和发布门槛。

## 方案的核心判断

- 不建立一个代表全部 Agent 能力的总分；结果正确性、过程质量、稳定性、效率和安全性分别报告。
- benchmark 是任务、环境和判分器的组合；eval 是运行协议、轨迹采集、判分、聚合和报告。
- 评测模块依赖 `agiwo.agent` 的公开执行面，不进入 `run_loop.py` 成为 Agent 主循环的依赖。
- 环境状态和副作用必须由程序检查器判定；LLM-as-a-judge 只用于开放式文本质量等无法稳定程序化判断的维度。
- 每个 case 都必须有独立 session、可记录的 suite/case/attempt/seed/config fingerprint，并保留原始运行证据。
- `RunLog` 和 `Trace` 仍是 Agent 运行事实；评测结果是独立的 evaluation artifact，不在运行日志中混入“成功分数”等派生事实。
- 外部 benchmark 采用 Adapter；核心评测内核不理解 WebArena、SWE-bench 或 OSWorld 的专有任务格式。

## Phase 依赖

```text
Phase 00 contracts
        |
        v
Phase 01 evaluation core -----> Phase 02 deterministic suites
        |                                  |
        v                                  v
Phase 03 graders/statistics --> Phase 04 external adapters
        |
        v
Phase 05 Console/operations --> Phase 06 hardening/release
```

Phase 00 和 Phase 01 必须先完成。Phase 02 是第一条可独立验收的价值路径；Phase 04 不应在没有本地确定性套件和报告协议时提前开始。

## 每个 batch 的标准生命周期

```text
suite version
  -> select cases
  -> create isolated attempt
  -> prepare environment
  -> run Agent
  -> collect RunOutput / RunLog / Trace
  -> grade outcome and trajectory
  -> teardown environment
  -> persist raw result
  -> aggregate report
```

## 研究依据

主流 benchmark 的能力边界和判分方式见 [00-proposal.md](./00-proposal.md)。本草稿采用了 AgentBench、GAIA、BFCL、tau-bench、ToolSandbox、AppWorld、WebArena、BrowserGym、OSWorld、AndroidWorld、SWE-bench、Terminal-Bench、MLE-bench、AgentDojo、Agent-SafetyBench 和 AgentHarm 的公开论文或官方页面作为参考。

