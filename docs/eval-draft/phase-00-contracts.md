# Phase 00：冻结契约与运行身份

状态：Planned

## 目标

在写评测 runner 之前，先把身份、版本、隔离、事实归属和敏感数据策略固定下来。这个 phase 不追求可运行 benchmark；它的退出条件是后续实现不需要靠隐式 metadata、字符串约定或直接扫描内部运行状态来猜测 case 关系。

## 设计不变量

1. `suite_id + suite_version + case_id + attempt` 唯一标识一次评测尝试。
2. `run_id` 和可选的 `trace_id` 是 Agent 运行证据的关联键。
3. 每个 attempt 都有独立 session 和明确的环境 reset/teardown。
4. evaluator、suite、case、agent config 和模型配置都必须能生成稳定 fingerprint。
5. 原始运行证据与聚合分数分开存储；聚合结果可重新计算。
6. 环境事实由程序 grader 判定；judge 不能覆盖程序事实。
7. 评测模块只能依赖 `agiwo.agent` 的公共 facade，不依赖 `run_loop`、`RunContext` 等内部模块。

## 拟新增文件

- `agiwo/evaluation/__init__.py`
- `agiwo/evaluation/models.py`
- `agiwo/evaluation/protocols.py`
- `tests/evaluation/test_contracts.py`

## 拟修改文件

- `agiwo/agent/models/log.py`
- `agiwo/agent/models/run.py`
- `agiwo/agent/runtime/state_writer.py`
- `agiwo/agent/run_loop.py`
- `agiwo/observability/trace.py`
- `agiwo/agent/__init__.py`，仅在决定公开运行身份类型时修改
- `AGENTS.md`，在真正实现并稳定后补充 `agiwo/evaluation/` 的目录职责

## 身份模型草稿

```python
@dataclass(frozen=True)
class EvaluationIdentity:
    evaluation_id: str
    suite_id: str
    suite_version: str
    case_id: str
    attempt: int
    seed: int | None
    config_fingerprint: str
```

`EvaluationIdentity` 属于 evaluation artifact，不应成为 Agent runtime 的业务概念。运行层只需要可靠地保留一个可关联的 `evaluation_id` 或完整的 evaluation metadata；评测存储仍是身份的 canonical owner。

## 需要冻结的运行改动

二选一，并在实现前做决定：

### 方案 A：RunLog 一等保留 evaluation metadata

- 给 `RunStarted` 增加结构化、可选的 `metadata`。
- `RunView` 和 `RunOutput` 读取并保留 metadata。
- `Trace` 增加受控的 run metadata 或 evaluation linkage。
- SQLite/in-memory replay 对 metadata 做 parity 测试。

优点是 Console 和 Trace 查询天然可关联；代价是扩展 Agent runtime 的公共数据契约。

### 方案 B：EvaluationStorage 独立维护关联

- `CaseAttempt` 写入 `run_id`、`session_id` 和 `trace_id`。
- 评测查询先从 evaluation storage 找身份，再读取 Agent 的 run/trace。
- 不修改 RunLog entry 结构，只要求 run_id 在执行完成前后稳定。

第一版建议采用方案 B，除非 Console 明确需要按 `suite_id/case_id` 直接查询 run。这样可以把 benchmark 复杂度留在 evaluation 模块；以后有真实查询用例时，再增加最小的一等 linkage 字段。

## 测试与退出门槛

- 用字符串输入和结构化 `UserInput` 各跑一次，身份关联结果一致。
- SQLite 与 in-memory 对同一 identity 的保存/读取结果一致。
- 运行失败、取消、超时仍能保存 `CaseAttempt`，并保留 `run_id`。
- secrets、完整 API key 和未授权的原始环境数据不会进入 artifact。
- `uv run python scripts/lint.py ci`
- `uv run pytest tests/agent tests/observability tests/evaluation -q`
- `uv run python -m compileall -q agiwo`

## 不在本 phase

- 不实现 suite runner。
- 不接入外部 benchmark。
- 不增加 Console 页面。
- 不把评测状态写成新的 runtime event bus。

