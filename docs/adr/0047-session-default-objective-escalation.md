# Session 默认对话；Objective 按需升级；取消强制 Decision 与独立 Outbox

运行证据表明：Console/渠道几乎每条消息都强制创建 Objective，并把同一句用户原文同时写入 Session 历史与 thin Run Input，导致模型上下文重复；即使 `hi, 1+1?` 也要支付收尾 Decision LLM、Contribution 与完整 Objective 事实税。决定把公开交互改成「Session 管对话，Objective 管跨 Run 长任务」：默认用户消息只进入 Session 历史一次并启动普通 root Turn/Run；仅在出现计划 milestone、需要等用户/验收/接力、或显式长任务控制时才升级并附着 Objective。取消每次 root Run 强制的收尾 Decision 模型调用；下一跳由机械规则或显式控制工具推导为 `HandoffDecision`（无 finalization LLM）。取消独立 Transactional Outbox 表与后台 dispatcher；Objective 续派发在同一事务写入 `RootRunRequested`，优先同步 `dispatch_execution`，崩溃恢复扫描「已 Requested 未 Started」的事实（supersedes ADR 0019；部分取代 ADR 0005 / ADR 0046 中“每 Run 必收尾 Decision / 每消息必 Objective”的隐含假设）。

## Status

superseded by ADR-0048

## Considered Options

- 保留永远有 Objective，但把简单路径薄到几乎不可见：改动面较小，却继续强迫「每次问好都是一次 Objective」，文档与实现仍易长回厚模板与强制收尾。
- 保留 Decision 收尾 LLM，仅对简单路径跳过：仍维护两套收尾契约与 JSON schema，模型与代码分叉成本高。
- 保留 Outbox，但去掉 `run_input` 快照：可靠性形状不变，独立 lease/dispatcher 对单进程 MVP 过重；`RootRunRequested` + reconcile 已能表达同一不变量。
- Session 默认对话 + Objective 升级；Decision/Outbox 降为规则与日志事实（选此）。

## Consequences

- 用户原文在 Session RunLog 中唯一；禁止 history 追加后再用同类文本作为 `UserMessage.from_system` Run Input。
- 无活动 Objective 且无升级信号时，Gateway 走 plain turn（**要求 session runtime 已接线**；未接线 fail-closed，不隐式创建 Objective），不创建 ObjectiveLog / 不做入口复杂度评估 / 不做 finalization LLM。
- 升级信号（现行实现）：当前 root RunPlan 曾出现 ≥1 milestone（latch）→ 机械 `HandoffDecision(target=verifier)` → Gateway `upgrade_from_plain_run`。plain 路径暂不支持 `ask_user` / 长任务 pause 升级；那些流程走已有 Objective 或显式 `create_objective_turn`。
- 无 open plan 时机械交付（`target=user`, `expects_reply=false`）。
- `verification_required` 仍由计划 latch 置位；声称完成但 latch 为 true 时由规则强制 verification Run，不要求模型再输出 Decision JSON。
- 已升级 Objective 的后续 root Run 预分配 `run_id`，事务内写 `RootRunRequested`；同步派发失败或进程崩溃后由 startup reconcile 补派，不引入 `DispatchRequested` outbox 投影为领域状态。
- ADR 0019（Transactional Outbox）与 ADR 0005（每成功 Assignment/Run 必须结构化 Decision）由本 ADR 取代。
- 一并废止或收窄：ADR 0004 / 0006 / 0008 / 0009 / 0015 / 0024–0027 / 0031 / 0038 / 0039（详见 `docs/adr/README.md`）。
- 实现切换前清理开发库，不写 migration。
- Assignment 时代分阶段建成计划已归档至 `trash/adr-plan-historical-2026-07-21/`，不再作为现行规格。
