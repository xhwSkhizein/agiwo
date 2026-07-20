# P3-04：实现 Run checkpoint 与可恢复中断 resume

状态：done

## 目标

让 Run 可以在安全边界进入 `PAUSED`（可恢复中断），并在进程内或重启后以相同 run_id、Assignment、agent identity、消息和已提交 step 继续。Pause 不等于 cancel：不写 termination reason，不生成 Outcome。恢复下一步由当前 llm context（RunLog 重建的消息列表）最后一项形态决定，而不是维护厚重的独立 resume_phase 状态机。

## 对应决定

- ADR 0029：用户暂停 / 预算暂停 / 用户停止 = 可恢复中断。
- ADR 0032：RunStatus 一等 PAUSED/RESUMED。
- ADR 0033：checkpoint 以消息边界恢复为主。

## 依赖

- P0-04、P2-02 已完成。

## 范围

包含：RunPaused/RunResumed/RunCheckpoint facts、消息末项恢复规则、RunLog replay、Scheduler 可恢复中断 control、Agent 内部恢复入口、与 cancel 的硬分离。

不包含：Objective 级 DRAINING barrier（P3-05）。

## 实施步骤

1. 定义 RunCheckpoint：checkpoint_id、last_committed_sequence、可选最小提示字段、AgentConfig hash、Assignment template hash 和时间。不要求完整 resume_phase 枚举覆盖所有业务阶段。
2. 恢复规则（确定性，写入测试）：
   - 最后一项为 assistant 且含 tool_call → 执行这些 tool call；
   - 最后一项为 assistant 且无 tool_call → 追加 `is_user_provided=false` 的 user 消息，继续 loop；
   - 最后一项为 tool result → 同样追加 false user，继续 loop。
3. checkpoint 不保存完整 messages、ledger、tool result、Artifact 或 RunContext；从 RunLog 重放至 last_committed_sequence。
4. 只在安全边界提交：已开始的 LLM/tool 完成并提交，已观察到的 LLM 响应成本已经记账，外部副作用无未知结果。
5. RunCheckpoint 与 RunPaused 在同一 RunLog committed batch 写入；任一失败时对外仍是 RUNNING。
6. 实现 **P2-02 已定义的** Scheduler facade 可恢复中断 pause/resume control 的行为，**不**复用 cancel_subtree；cancel/shutdown 仅用于不可恢复强制终止与进程清理。不得另起第二套 pause API。
7. Agent 内部恢复路径使用相同 run_id 重建 RunContext；公开 start/run API 不增加 resume 参数。
8. 恢复前验证 RunLog 连续性、Objective/Assignment 状态，并加载 checkpoint/Assignment **已钉住**的 AgentConfig 与模板快照；用快照自身的 content hash 做完整性校验。与当前 Registry 最新 hash 不一致是预期，不得拒绝 resume。快照缺失或自检失败时结构化故障并保持 PAUSED，禁止静默换用新配置。
9. 增加 `RunResumePrepared`：重建 runtime、验证后等待 Objective 级共享 release barrier；此时 RunStatus 仍是 PAUSED。
10. 全部活动 Run prepare 成功且 ObjectiveService 提交 resume 后，释放 barrier；各 Run 追加 RunResumed，并按步骤 2 决定下一步。
11. 任一 prepare 失败时销毁全部 prepared runtime，所有 Run 保持 PAUSED。
12. Python Agent 对象可重建，但使用钉住的 stable agent identity 与配置快照，而非 resume 时刻的 live 默认配置。
13. 向 run loop 提供 objective-agnostic 的内部信号枚举，至少区分：`pause`（可恢复中断）、`force_fail`（不可恢复）；**不再**为运行中用户输入提供 `finalize_for_steering` 类信号。
14. Assignment 创建时即持久化所用模板/默认 AgentConfig 快照（或等价不可变引用）；checkpoint 引用该快照。P5-05 编辑路径不得改写已钉住快照。

## 主要改动位置

- `agiwo/agent/models/run.py`
- `agiwo/agent/models/log.py`
- `agiwo/agent/runtime/state_writer.py`
- `agiwo/agent/run_bootstrap.py`
- `agiwo/agent/run_loop.py`
- `agiwo/scheduler/commands.py`
- `agiwo/scheduler/engine.py`
- `agiwo/scheduler/runner.py`
- `tests/agent/test_run_resume.py`
- `tests/scheduler/test_pause_resume.py`

## 测试计划

- 三种消息末项形态各自只执行正确的下一步。
- PAUSED 不产生 RunFinished/Failed/Interrupted、TerminationDecided 或 Outcome。
- 钉住快照：PAUSED 期间修改 Registry 默认配置/模板后，resume 仍用原快照成功；新 Assignment 才用新模板。
- 快照缺失或 content hash 自检失败时拒绝 resume 并保持 PAUSED；不得因与 live Registry hash 不同而拒绝。
- 同 run_id 进程内恢复与 SQLite 关闭重开恢复。
- RunLog sequence 缺口拒绝恢复。
- 多 Run prepare 任一失败时无 RunResumed；全部成功后单一 barrier release。
- 在途调用完成、结果提交且实际成本记账前不能宣称 PAUSED。
- cancel 仍不可恢复；普通用户停止不走 CANCELLED/FAILED。

## 完成标准

- RunView 能只凭 RunLog 投影 RUNNING/PAUSED/终态。
- checkpoint 只含最小控制游标；恢复规则以消息末项为准。
- 恢复不创建第二个 run_id，不重放已提交副作用。
- Agent 公开 API 无变化；cancel ≠ pause。
- agent/scheduler/storage tests 与 lint 通过。

## 风险与回退

若恢复逻辑依赖内存 coroutine 或完整 resume_phase 状态机，就偏离 ADR 0029/0033。测试必须真正关闭运行时和存储后重建；只做同进程 pause 不算完成。
