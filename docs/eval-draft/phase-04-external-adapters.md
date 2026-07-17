# Phase 04：外部 Benchmark Adapters

状态：Planned

## 目标

把外部 benchmark 接入为独立 Adapter，同时保留 Agiwo 的统一运行、证据和报告协议。外部 benchmark 的环境、任务格式和 grader 逻辑不得渗入 `EvaluationRunner` 或 `agiwo.agent`。

## 接入顺序

按环境成本和故障面分层：

### Tier 1：数据与工具调用

- BFCL：工具选择、参数、并行和多轮调用。
- GAIA：通用工具助理和最终答案。
- BrowseComp：开放网页搜索结果，优先从官方数据和答案检查器开始。

### Tier 2：状态化容器环境

- tau-bench：用户模拟、领域政策、数据库目标状态和 `pass^k`。
- AppWorld：多应用、API 和 collateral damage。
- SWE-bench：容器化代码修复和隐藏测试。
- Terminal-Bench：终端任务、独立环境和综合测试。

### Tier 3：网页、桌面和移动端

- BrowserGym/WorkArena++：浏览器观察与动作接口。
- OSWorld：桌面 GUI、真实应用和状态检查。
- AndroidWorld：Android emulator、动态参数和设备状态。

### Tier 4：安全

- AgentDojo：不可信工具输出和提示注入。
- Agent-SafetyBench：多类别安全风险。
- AgentHarm：恶意多步任务和 jailbreak 场景。

## 文件计划

### 创建

- `agiwo/evaluation/adapters/__init__.py`
- `agiwo/evaluation/adapters/bfcl.py`
- `agiwo/evaluation/adapters/tau_bench.py`
- `agiwo/evaluation/adapters/appworld.py`
- `agiwo/evaluation/adapters/swebench.py`
- `agiwo/evaluation/adapters/terminal_bench.py`
- `agiwo/evaluation/adapters/browsergym.py`
- `agiwo/evaluation/adapters/osworld.py`
- `agiwo/evaluation/adapters/androidworld.py`
- `agiwo/evaluation/adapters/agentdojo.py`
- `tests/evaluation/adapters/test_adapter_contracts.py`
- 各 Adapter 对应的 integration test，默认标记为 external/expensive

### 可能创建

- `docs/guides/evaluation.md`，仅在内核稳定后从 draft 晋升
- `scripts/run_external_evaluation.py`
- `docker/evaluation/`，仅当隔离环境确有复用价值时创建

## Adapter 接口要求

每个 Adapter 必须声明：

- benchmark 名称和版本。
- 环境依赖和启动方式。
- 任务数量、split 和是否包含 hidden test。
- reset/teardown 方式。
- agent action/tool/observation 映射。
- outcome grader 和 trajectory grader。
- 网络、文件、容器和 credential 要求。
- 成本、耗时和并发限制。
- 原始 benchmark 结果与 Agiwo `Grade` 的映射。

## 适配原则

- 不修改 benchmark 官方数据来迁就 Agiwo；写转换 Adapter。
- 不把外部 benchmark 的总分直接映射成 Agiwo 的通用总分。
- 保留官方 evaluator 原始输出，同时保存 Agiwo 标准化 grade。
- 记录官方 benchmark 版本、commit/tag 和 evaluator command。
- 只在许可证、数据使用条款和网络策略允许时下载或缓存数据。
- GUI 和容器任务必须设置资源上限、超时、自动清理和失败后的隔离回收。
- 所有 external tests 与本地单元测试分开，避免 CI 被外部环境污染。

## 每个 Adapter 的验收

- 一个最小 case 可以通过统一 `CaseRunner` 执行。
- 官方 evaluator 与 Adapter grade 的对应关系有文档和 fixture。
- 失败时能区分 Agent 失败、环境失败、grader 失败和基础设施失败。
- 至少保留一个成功样例和一个已知失败样例。
- 外部环境中断后不会留下运行中的容器、浏览器或 emulator。

