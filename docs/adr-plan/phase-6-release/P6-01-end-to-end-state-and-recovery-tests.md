# P6-01：建立端到端状态与恢复测试矩阵

状态：done

## 目标

用一组可重复、可故障注入的端到端测试证明 ObjectiveLog、RunLog、Scheduler state、outbox 和 Console 投影在正常、并行、预算、暂停、故障、运行中用户输入与重启下保持一致。测试必须验证领域不变量，而不仅是接口返回 200。

## 对应决定

- 覆盖 ADR 0001 至 0045 的跨模块不变量；具体主责关系以根目录 ADR 覆盖矩阵为准。
- 重点验证 ADR 0010、0019、0024、0028、0032、0033、0036 中的预算、可靠派发、Outcome、收敛、恢复和 identity 约束。

## 依赖

- P0 至 P5 全部完成。

## 测试计划

测试由确定性基础设施、分场景状态矩阵和跨存储一致性断言三部分组成。以下内容共同构成本任务的执行步骤，不能只选 happy path。

## 测试基础设施

1. 建立 deterministic fake model，能按调用 phase/ordinal 返回 assistant、tool call、finalization JSON、解析错误和 fault。
2. 建立带幂等声明、外部副作用计数和 request_sent 控制的 fake tools。
3. 使用 fake clock 控制 backoff、outbox claim lease、active window 和 checkpoint resume。
4. 同一场景分别运行 memory 与 SQLite backend；恢复场景必须关闭并重建 service/scheduler/agent/store 对象。
5. 提供日志一致性断言：ObjectiveLog sequence、RunLog sequence、outbox 状态和 projector view。

## 必测场景矩阵

### 正常职责主线

- intake -> work -> verification pass -> ObjectiveDelivered。
- verification reject -> fresh work -> second verification -> delivered。
- target=user expects_reply true -> user reply -> fresh Assignment。
- 同 Session 完成 Objective 后，新输入创建第二个 Objective，旧 Objective 不重开。

### 计划与上下文

- root RunPlan 未解决时 guard 继续 loop，全部解决后 finalization。
- max_steps_per_run 触发，Outcome 带 carry_forward，新 Run 从空计划开始。
- child RunPlan 不阻止 root finalization，也不进入 Objective 第二份计划。
- 多 Assignment 消息前缀稳定；formal compaction 后跨 Run 正确恢复 MessagesRebuilt。

### 并发与预算

- 顺序调用允许最后一个 attempt 越过 max_llm_cost_usd，之后不再启动调用。
- N 个 child 并行 LLM 在阈值前获准后均完成并按实际响应记账；DRAINING 后没有第 N+1 个调用启动。
- tool preflight 与 execute 之间触发 DRAINING：execute 前可推进检查失败则不得开始；已经开始的在途动作允许完成。
- handoff、verification、cost、active time 四维分别耗尽。
- 任一 child 触发预算，全 Objective DRAINING，未开始动作被阻止。
- summary/finalization 前耗尽，resume 从准确 phase 继续。

### Pause 与恢复

- User Pause 和 Budget Pause 都复用原 Assignment/run_id/context。
- 多 Run resume 任一 prepare 失败时全部保持 PAUSED；全部 prepare 成功后共享 barrier 只 release 一次。
- DRAINING barrier 未齐不进入稳定暂停。
- active time resume 开新窗口，first_started_at 不变。
- cancel/shutdown 仍不可恢复，与 pause 清楚区分。

### Fault

- retryable+idempotent 成功重试；每个 attempt 独立 cost。
- retry exhaustion -> system Outcome -> fresh work。
- nonblocking nonretryable tool fault 留给当前 agent。
- blocking nonretryable 与 outcome_unknown -> user，且 max_handoffs=0 仍可达。

### 运行中用户输入与 Session

- root 运行中收到输入：写 ObjectiveUserInput，并向同一 root Run 注入 false user；无 Outcome、无新 Assignment、无 DRAINING。
- 连续不同输入按序追加注入；幂等 key 不重复。
- 用户 pause/停止与预算 pause：DRAINING → PAUSED，可按消息末项恢复；普通停止不走 CANCELLED/FAILED。
- archive active Session 先可恢复中断，restore 不自动运行。
- fork 不复制 Objective/checkpoint，summary 只使用一次。

### Crash injection

- Objective facts/outbox commit 后。
- command receipt/Session slot commit 前后。
- outbox claim 后、RunStarted 前。
- RunStarted 后、outbox dispatched 前。
- RunStarted 后、AssignmentExecutionStarted 前。
- LLM 无响应、部分响应、cost 已写 RunLog 但未应用 Objective、tool 外部副作用 marker 后无结果。
- RunLog=RUNNING 但进程重启后 runtime 不存在。
- RunPaused/checkpoint batch 前后。
- 多 Run resume prepare、Objective resume commit、barrier release 前后。
- Run terminal 后、Outcome 前。
- Outcome 后、outbox complete 前。

## 一致性断言

- 一个 Objective 最多一个非终态 Assignment；一个 Session 最多一个非终态 Objective。
- Session slot 唯一，command receipt 同 key/hash 重放原响应、同 key 不同 hash 冲突。
- 每个终态 Assignment 恰有一个 Outcome；PAUSED 没有 Outcome。
- outbox completed 必有 Outcome；同 run_id 只有一个 RunStarted。
- RunStarted 后最终恰有一个 AssignmentExecutionStarted；outbox dispatched 不直接改变 ObjectiveView。
- 每个有响应的 LLM attempt 恰有一次实际成本记录和一次 Objective 用量应用；无响应 attempt 为零，used 可以由最后一个或在途 N 个调用解释性越过 limit。
- 每个实际模型 attempt 保持 logical_call_id/phase 语义并有唯一 attempt_no；provider retry 不是 phase。
- ObjectiveDelivered 引用存在的 final Outcome/report，不依赖最后 assistant message。
- 用户来源 false 的消息不产生 ObjectiveUserInput 或普通用户气泡。
- 每个未外置 ObjectiveUserInput 在后继 Assignment 的实际模型上下文中恰好出现一次（经稳定前缀复用，非中段补回）；已授权外置输入在对应历史位置只出现 path 与 summary，系统模板不复制其内容。历史缺少对应 input_id 时 fail closed。
- ContextCapacityExceeded 不派发不可执行 Assignment；用户 externalize command 后原始输入仍可查询，容量通过才继续。

## 主要改动位置

- `tests/objective/e2e/`
- `tests/agent/` 的 prefix/run-resume contract tests
- `tests/scheduler/` 的 tree/drain tests
- `console/tests/` 的 gateway/session tests
- 共享 deterministic fixtures

## 完成标准

- 每个 ADR 主责任务至少有一个直接测试引用或断言。
- memory/SQLite 行为一致。
- crash tests 证明不会重复模型/tool 外部副作用。
- 整套测试无真实网络、真实时间或随机竞态依赖。
- SDK、Console backend 和前端测试全绿。

## 风险与回退

不要只建立一个超长 happy-path 测试。状态矩阵应由小型可定位场景组成，再用少量完整 E2E 串联；否则失败时无法判断是领域、存储、调度还是 UI 投影问题。
