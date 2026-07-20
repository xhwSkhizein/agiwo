# P2-06：打通正常接力、验收与最终交付

状态：**done**（2026-07-18）

## 目标

让 ObjectiveService 消费唯一 AssignmentOutcome 和结构化 Decision，可靠地创建下一 Assignment 或完成最终交付。正常闭环必须包含独立 intake、work、verification 职责，并始终复用 Session persistent root agent identity。

## 对应决定

- ADR 0002、0004、0008、0009：接力、入口、验收和返工。
- ADR 0005：Decision 是唯一路由来源。
- ADR 0012：max_steps 机械 handoff 与 carry_forward 必须被下一 Assignment 消费。
- ADR 0025：职责差异不映射到不同 AgentConfig。
- ADR 0030、0031：Objective/Assignment 状态。
- ADR 0038：计划项与 carry_forward。

## 依赖

- P1-02、P2-03、P2-05 已完成。

## 范围

包含：Outcome 提交事务、Decision 消费、peer Assignment 创建、verification、WAITING_USER、ObjectiveDelivered 和 happy-path integration。

不包含：预算上限、fault retry、运行中用户输入注入和 HTTP/Console。

## 实施步骤

1. ObjectiveService 接收 root Run finalization result，写入普通文本 report、可选文件 Artifact 引用、Contribution/annotations、合法 current_goal revision 和唯一 AssignmentOutcome。
2. Outcome、Assignment 终态、DecisionAccepted、预算计数预备 fact、下一 Assignment/outbox 必须在一个 ObjectiveStore 事务中提交；本阶段先记录计数，P3-01 加限额检查。
3. intake Assignment 形成第一项 Decision。它不成为长期 supervisor，也不预先规划完整 workflow。
4. target=agent 创建 kind=work 的 peer Assignment；target=verifier 创建 kind=verification。前一 Assignment 保持终态，下一项不使用 parent-child 关系。
5. work 完成提议必须 target=verifier；不能直接 ObjectiveDelivered。
6. verifier 通过时产生 target=user, expects_reply=false，并从明确 Outcome 的文本 report 与文件 Artifact 引用写 ObjectiveDelivered，Objective 进入 COMPLETED。
7. verifier 不通过时 target=agent，保留 verifier report 中未满足要求、证据缺口和理由，创建 fresh work Assignment；不恢复旧 Run。
8. target=user, expects_reply=true 关闭活动推进并令 Objective 进入 WAITING_USER；用户回复后的新 Assignment 在 P5/P4 接入完整入口。
9. peer Assignment 使用同一 Session root agent_id、不同 assignment_id/run_id；kind 只决定模板，不改变 config、model、tools 或 skills。
10. Assignment 内 Parallel/Pipeline/agent tool child 继续使用现有委派树；child 只返回 RunOutput，由 root 汇总，不产生独立 Outcome。
11. outbox 在 Outcome 提交后完成旧 record，并为下一 Assignment 创建新 record。
12. **`carry_forward` → 下一 Assignment Input（ADR 0012 / 0038）：**
    - 当上一 Outcome 含 `carry_forward` 计划项（含 max_steps INTERRUPTED 路径）且 Decision 为 `target=agent` 时，新 work Assignment 的模板/装配必须显式注入这些项：稳定 milestone id、description、中断时状态，以及「须在本 Run 从空 RunPlan 重新声明并继续」的边界说明。
    - 不得只把 carry_forward 留在 Outcome 投影里却不进入模型可见的 Assignment Input。
    - 新 Run 仍从空 RunPlan 开始；历史中的旧 `update_plan` 不代表当前计划（P2-04 边界 notice 与此一致）。
13. max_steps 机械 handoff：ObjectiveService 若收到非 `target=agent` 的 Decision 且 Outcome 标记为 max_steps 机械收口，拒绝或纠正为 `agent` 后再开 work Assignment（与 P2-05 双保险）。
14. 增加 memory backend 端到端测试和 SQLite 重放测试，证明 Timeline 与当前投影一致。

## 主要改动位置

- `agiwo/objective/service.py`
- `agiwo/objective/finalization.py`
- `agiwo/objective/projection.py`
- `agiwo/objective/dispatch.py`
- `agiwo/scheduler/runner.py`
- `tests/objective/test_mainline.py`

## 测试计划

- intake -> work -> verifier pass -> ObjectiveDelivered。
- work 试图直接交付时被协议校验拒绝并进入既定纠正/最小 handoff。
- verifier reject -> fresh work，旧 work/verifier 保持终态。
- max_steps INTERRUPTED → 下一 work Assignment Input **包含**上一 Outcome 的全部 carry_forward 项（按 id）；新 RunPlan 初始为空，模型须重新 declare。
- max_steps 路径 Decision 非 agent 时不能创建 verifier/user Assignment。
- user expects_reply true/false 的状态差异。
- parent-child delegation 与 peer handoff 在 IDs、depth、Outcome 数量上的差异。
- 同 Session 多个 Assignment agent_id 相同，run_id 不同。
- SQLite 关闭重开后 ObjectiveView、Outcome 文本 report 和 delivered 文件 Artifact 引用一致。

## 完成标准

- 每个终态 Assignment 恰有一个 Outcome。
- 一次 Objective 同时最多一个非终态 Assignment。
- 只有 verification pass 能触发正式交付。
- HandoffDecision 无具体 executor、agent name、config 或 pattern。
- max_steps 产生的 handoff 不得被模型改成 verifier/user；carry_forward 必须进入下一 work 的实际模型输入。
- Scheduler 不做语义判断，ObjectiveService 不解析自由文本。
- objective mainline tests 与 lint 通过。

## 风险与回退

本阶段尚未启用硬预算，因此不得切换 Console/渠道用户流量。它只是正常主链的内部垂直切片；P3/P4/P5 完成后才具备生产入口条件。

