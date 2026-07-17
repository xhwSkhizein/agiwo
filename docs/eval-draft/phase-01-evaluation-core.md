# Phase 01：Evaluation Core

状态：Planned

## 目标

实现一个不理解具体 benchmark 领域的通用评测内核。它只负责：选择 case、创建 attempt、准备环境、调用 Agent、采集证据、调用 grader、保存结果和关闭环境。

## 文件计划

### 创建

- `agiwo/evaluation/models.py`
- `agiwo/evaluation/protocols.py`
- `agiwo/evaluation/runner.py`
- `agiwo/evaluation/artifacts.py`
- `agiwo/evaluation/storage/__init__.py`
- `agiwo/evaluation/storage/base.py`
- `agiwo/evaluation/storage/memory.py`
- `agiwo/evaluation/storage/sqlite.py`
- `tests/evaluation/test_runner.py`
- `tests/evaluation/test_evaluation_storage.py`
- `tests/evaluation/test_artifact_redaction.py`

### 可能修改

- `agiwo/evaluation/__init__.py`
- `pyproject.toml`，只在确有新依赖时修改
- `docs/README.md`，最终稳定后再加入正式入口

## 核心模型

```python
@dataclass(frozen=True)
class TaskCase:
    case_id: str
    input: UserInput
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class PreparedCase:
    case: TaskCase
    agent_config: AgentConfig
    tools: tuple[BaseTool, ...] = ()
    environment: EnvironmentHandle | None = None


@dataclass(frozen=True)
class CaseAttempt:
    evaluation_id: str
    suite_id: str
    suite_version: str
    case_id: str
    attempt: int
    seed: int | None
    run_id: str | None
    session_id: str | None
    trace_id: str | None
    status: Literal["completed", "failed", "cancelled", "skipped"]
    run_output: RunOutput | None
    grade: Grade | None
    error: str | None
```

上述代码只是方向草稿；最终模型应服从仓库现有 dataclass/Pydantic 选择，并使用结构化 `UserInput`。

## Protocols

```python
class EvaluationSuite(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def version(self) -> str: ...

    async def cases(self) -> Sequence[TaskCase]: ...

    async def prepare(
        self, case: TaskCase, *, attempt: int, seed: int | None
    ) -> PreparedCase: ...


class AgentFactory(Protocol):
    async def build(self, prepared: PreparedCase) -> Agent: ...


class Grader(Protocol):
    async def grade(self, context: GradingContext) -> Grade: ...


class EnvironmentHandle(Protocol):
    async def snapshot(self) -> Mapping[str, Any]: ...
    async def close(self) -> None: ...
```

`CaseRunner` 是外部唯一需要了解的深模块入口，建议只暴露：

```python
await runner.run_suite(
    suite,
    agent_factory=factory,
    attempts=1,
    selection=selection,
)
```

不要让调用者自己拼接 reset、`Agent.start()`、trace 查询、grade 和 teardown；否则复杂度会在每个 benchmark Adapter 中重复出现。

## 生命周期规则

```text
select -> prepare -> build agent -> start/run -> collect -> grade -> persist -> teardown
```

- `prepare` 失败：记录 `skipped` 或 `failed`，不得启动 Agent。
- Agent 运行失败：保存异常、run_id 和已有证据，随后仍尝试 teardown。
- grader 失败：运行结果不得被覆盖为 Agent 失败；应标记 `grader_error`。
- teardown 失败：记录资源清理错误，但保留原始 case 结果。
- 取消 batch：当前 attempt 可取消，已完成 attempt 不得被重写。

## 证据采集

每次 attempt 至少保存：

- `CaseAttempt` identity
- `RunOutput`
- `run_id`、`session_id`、`trace_id`
- environment snapshot 或 snapshot fingerprint
- agent/model/tool/config fingerprint
- grader 名称、版本和原始 grade payload
- 错误和取消原因

完整 prompt、LLM 消息和工具输出应由 artifact policy 控制，默认提供脱敏和大小上限；不能因为“方便 debug”而无限复制敏感数据。

## Storage 设计

`EvaluationStorage` 与 `RunLogStorage`、`BaseTraceStorage` 并列，不互相替代。第一版提供 memory 和 SQLite 两个 Adapter；SQLite 可以使用独立的 `evaluation.sqlite`，避免为已有运行库引入 schema migration。

建议的逻辑记录：

- `evaluation_suite`
- `evaluation_case`
- `evaluation_batch`
- `evaluation_attempt`
- `evaluation_grade`
- `evaluation_artifact`

派生聚合报告可以重新计算，不作为唯一事实保存。

## 测试与退出门槛

- fake AgentFactory 能运行一个无工具 case。
- fake environment 能验证 `prepare -> run -> grade -> teardown` 顺序。
- Agent exception、grader exception、teardown exception 分别得到不同状态。
- 同一 suite 重跑不会复用上一次 session 或环境状态。
- memory 与 SQLite storage 通过同一 contract test。
- `run_id` 能反查 run view，`trace_id` 能反查 trace。

