# 通用 Benchmark Eval 总体方案

状态：Draft

## 1. 背景

当前 Agiwo 已能执行 Agent run、记录可重放的 `RunLog`，并从 committed facts 构建 `Trace`。缺少的是一层独立的实验能力：它应能批量提供任务、准备隔离环境、调用 Agent、收集运行证据、执行不同类型的 grader，并以可复现的统计口径比较 Agent 系统。

Benchmark 不应被理解为一张题目表。一个可用的 benchmark 至少包含：

```text
Task cases + Environment + Agent adapter + Grader + Run protocol + Report
```

## 2. 调查结论

### 2.1 Benchmark 按能力分层

| 层次 | 代表项目 | Agiwo 应借鉴的机制 |
| --- | --- | --- |
| 通用助理 | [AgentBench](https://arxiv.org/abs/2308.03688)、[GAIA](https://arxiv.org/abs/2311.12983) | 多环境、多模态、工具与最终答案的组合评估 |
| 工具调用 | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) | 工具选择、参数结构、并行调用、多轮调用 |
| 状态化工具 | [tau-bench](https://arxiv.org/abs/2406.12045)、[ToolSandbox](https://arxiv.org/abs/2408.04682)、[AppWorld](https://arxiv.org/abs/2407.18901) | 数据库/应用最终状态、政策遵守、缺少信息、意外副作用 |
| 网页与研究 | [WebArena](https://arxiv.org/abs/2307.13854)、[BrowserGym](https://arxiv.org/abs/2412.05467)、[BrowseComp](https://arxiv.org/abs/2504.12516) | 长流程网页交互、开放网页搜索、状态或参考答案判分 |
| 桌面与移动端 | [OSWorld](https://arxiv.org/abs/2404.07972)、[AndroidWorld](https://arxiv.org/abs/2405.14573) | 真实 GUI、设备状态检查、任务初始化和清理 |
| 编程与终端 | [SWE-bench](https://arxiv.org/abs/2310.06770)、[Terminal-Bench](https://arxiv.org/abs/2601.11868)、[MLE-bench](https://arxiv.org/abs/2410.07095) | 容器化环境、隐藏测试、资源预算和专业任务指标 |
| 安全 | [AgentDojo](https://arxiv.org/abs/2406.13352)、[Agent-SafetyBench](https://arxiv.org/abs/2412.14470)、[AgentHarm](https://arxiv.org/abs/2410.09024) | 提示注入、越权、恶意任务、风险识别与安全效用 |

这些项目不能放到一个分数轴上直接比较。它们测试的是不同的系统组合：模型、prompt/scaffold、工具、环境、预算和判分器共同决定结果。

### 2.2 常用 eval 手段

1. **结果判分**：精确匹配、规范化匹配、JSON/schema 校验、隐藏测试、数据库状态比较、文件状态比较。
2. **过程判分**：工具选择、参数、调用顺序、工具错误、恢复能力、政策遵守、无效调用和额外副作用。
3. **交互判分**：多轮用户模拟、状态化对话、信息不足时是否询问、是否正确处理用户修正。
4. **稳定性判分**：单次成功率、`pass@k`、`pass^k`、任务变体成功率和跨 seed 方差。
5. **开放式质量判分**：结构化 LLM judge 或人工 pairwise/rubric 评测；不能替代环境事实检查。
6. **效率判分**：耗时、首 token 延迟、token、LLM 调用、工具调用、成本和成功率/成本。
7. **安全判分**：攻击成功率、越权动作率、危险动作阻断率、正常任务效用和过度拒答率。

### 2.3 比较协议

比较两个 Agent 系统时，必须冻结或显式报告：模型版本、provider、系统 prompt、工具 schema、工具实现、环境版本、初始状态、最大步骤、超时、token/cost budget、采样参数、seed、网络策略、数据版本和 evaluator 版本。

单次成功率不能代表可靠性。建议至少同时报告：

```text
success@1 = 单次成功 case 数 / case 总数
pass@k    = k 次尝试中至少成功一次
pass^k    = k 次独立尝试全部成功
```

聚合时使用按能力类别的 macro average，并保留每个 case 的原始结果和置信区间；不要只给一个被任务数量偏斜的 micro average。

## 3. 目标架构

建议新增顶层 package：`agiwo/evaluation/`。它是一个依赖 Agent 公开接口的深模块；外部调用者只需提供 suite、Agent factory 和报告选项，复杂的生命周期、隔离、证据采集和统计逻辑隐藏在内部。

```text
EvaluationSuite
  ├── TaskCase
  ├── EnvironmentAdapter
  ├── AgentFactory
  └── Grader

EvaluationRunner
  ├── prepares isolated attempt
  ├── calls Agent.start/run
  ├── collects run_id / trace_id / RunLog / Trace
  ├── invokes Grader
  └── persists CaseResult

EvaluationStorage
  ├── immutable suite/case/version metadata
  ├── raw attempt result
  ├── grader result
  └── aggregate report
```

## 4. 接入当前 Agiwo 的事实

- `Agent.start()` 是异步执行句柄入口，适合由 `CaseRunner` 调用：[agent.py](/Users/hongv/workspace/agiwo/agiwo/agent/agent.py:366)。
- `RunOutput` 已提供响应、终止原因、token、步骤、工具调用、耗时和成本：[run.py](/Users/hongv/workspace/agiwo/agiwo/agent/models/run.py:129)。
- `RunLogStorage` 已提供 entries、run views、step views 和 replay 查询：[base.py](/Users/hongv/workspace/agiwo/agiwo/agent/storage/base.py:36)。
- `Trace`/`Span` 已记录 LLM/tool 调用、输入参数、输出、层级和指标：[trace.py](/Users/hongv/workspace/agiwo/agiwo/observability/trace.py:33)。

当前缺口是评测身份没有被完整保留下来。`Agent.start()` 接收 `metadata`，但 `RunStarted` 构造时没有持久化它：[state_writer.py](/Users/hongv/workspace/agiwo/agiwo/agent/runtime/state_writer.py:444)。最终 `RunOutput.metadata` 也只保留 `run_start_seq`：[run_loop.py](/Users/hongv/workspace/agiwo/agiwo/agent/run_loop.py:137)。Phase 00 必须先解决这个契约问题，或者明确由独立的 evaluation storage 维护 run_id 到 case identity 的映射。

## 5. 目标边界

### 包含

- 通用 suite/case/attempt/result/report 模型。
- Agent 运行、环境准备和清理的统一生命周期。
- response、state、trajectory、safety、judge 五类 grader 接缝。
- `success@1`、`pass@k`、`pass^k`、成本、延迟、置信区间和失败分类。
- 本地确定性的 function-calling 与 stateful-tool suite。
- 外部 benchmark 的 Adapter，而非把外部环境逻辑放进 Agent 核心。
- CLI、Console 查询和运行取消。

### 不包含

- 不把某个 benchmark 设为 Agiwo 的唯一标准。
- 不在第一版直接支持所有 GUI、浏览器和容器环境。
- 不让 LLM judge 判定真实副作用或安全事实。
- 不复制 `RunLog`/`Trace` 作为第二套 Agent 运行真相。
- 不在现有数据库上写复杂迁移；新评测存储应使用新表或独立数据库，并遵守仓库“无 schema migration”约定。

## 6. 关键决策

| 决策 | 选择 | 原因 |
| --- | --- | --- |
| Package 名称 | `agiwo.evaluation` | 语义清楚，不与 Python 内置 `eval` 混淆 |
| 运行事实 | 继续使用 `RunLog`/`Trace` | 当前已有 replay 和观测能力 |
| 评测结果 | 独立 `EvaluationStorage` | 评测分数是 derived artifact，不是 Agent runtime fact |
| 第一套 suite | 本地确定性工具环境 | 成本低、可重复，能覆盖最重要的调用与状态接缝 |
| 外部环境 | Adapter | 每种环境的启动、reset、grader 和资源需求不同 |
| 总分 | 不设默认总分 | 不同 benchmark 的能力含义不可相加 |

