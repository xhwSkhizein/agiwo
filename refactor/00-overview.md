# Refactor 总览：MainAgent M3 落地后的收敛与加固

> 基于 2026-07-23 对未提交改动（ADR 0049 M3 落地）的 code review。
> 本目录按阶段拆分任务文档；每个文档可独立执行、独立验收。
> 文档与源码冲突时以源码为准，执行前先复核引用的行号。

## 背景

本次未提交改动引入了 MainAgent / SessionIntent / Worker / 完成门，并把
Scheduler 瘦身为 waitset + cancel-subtree。架构方向正确，但存在：

1. 一处确定性 bug（`MainAgent._pending` 队头阻塞导致 Worker 报告永久丢失）；
2. 数个生命周期竞态窗口；
3. `worker_bridge` 对 Scheduler 私有成员的穿透，违背"瘦 Scheduler 公开边界"初衷；
4. 若干存储并发与配置项语义问题。

## 阶段划分

### Phase 1 — 提交前必须修（正确性）

| 文档 | 任务 | 修复的 review 问题 |
| --- | --- | --- |
| [11-unified-queue.md](11-unified-queue.md) | 统一队列收敛：`_pending` 成为唯一入队原语 + 单一 drain 路径 | #1 队头阻塞、#5 双真相源、#6 enqueue 返回值被忽略 |
| [12-main-agent-lifecycle.md](12-main-agent-lifecycle.md) | MainAgent 生命周期加固：异常吞噬、cancel 竞态 | #2 Task 异常无人 retrieve、#3 cancel 期间 run 被"复活" |
| [13-scheduler-worker-api.md](13-scheduler-worker-api.md) | Scheduler 公开 Worker API，`worker_bridge` 去私有化 | #4 私有成员穿透 |

### Phase 2 — 提交后尽快跟进（健壮性）

| 文档 | 任务 | 修复的 review 问题 |
| --- | --- | --- |
| [21-session-intent-store.md](21-session-intent-store.md) | SessionIntent 存储并发保护 + 追加化 + 截断策略 | #7 读-改-写竞态、无限增长 |
| [22-semantic-gate-flag.md](22-semantic-gate-flag.md) | 处理 `enable_semantic_completion_gates` no-op 开关 | #8 配置项看似生效实则无效 |
| [23-spawn-worker-abort.md](23-spawn-worker-abort.md) | `SpawnWorkerTool` 同步等待可中断 | #9 abort_signal 被丢弃、无超时 |

### Phase 3 — 结构与风格（可排期）

| 文档 | 任务 |
| --- | --- |
| [31-structure-style.md](31-structure-style.md) | WorkerService 窄回调解循环依赖、gate context 传参化、mixin 评估、双写投影测试、杂项 |

## 依赖关系

```
11 ──► 12   （12 的 cancel 逻辑建立在 11 的统一 drain 之上）
13         （独立，可并行）
21, 22, 23 （相互独立，依赖 Phase 1 合入后的稳定基线）
31         （最后做；其中"gate context 传参化"与 22 有交集，先做 22）
```

## 统一验收命令

每个阶段完成后运行：

```bash
uv run pytest tests/agent/ tests/scheduler/ -x -q
uv run pytest console/tests/ -x -q
uv run python scripts/repo_guard.py
```

## 原则（贯穿所有任务）

- **最小上游修复**：先定位根因，不做下游 workaround。
- **域语言一致**：改动后同步 `CONTEXT.md` / `AGENTS.md` 中受影响的表述；
  不引入 ADR 0048/0049 的作废词。
- **测试先行**：每个任务文档列出"先写的回归测试"；不删除、不弱化既有测试，
  但允许修正**断言了错误行为**的测试（如 `test_completion_gates_e02.py:213`
  对 `_pending` 残留的断言）。
- **删除文件一律 `mv` 到 `trash/`**，不直接 `rm`。
