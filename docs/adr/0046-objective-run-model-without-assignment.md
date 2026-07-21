# Objective 以 Run 为责任边界，删除 Assignment；验收由轨迹 Latch 驱动

运行证据表明 Assignment 三态协议、强制 work→verifier、厚模板与收尾 correction 使简单问答承担过高延迟与 token，且与 Scheduler mailbox 心智重叠。决定删除 Assignment 聚合，公开模型固定为 `Session → Objective → Run`；root Run 携带 `Run Role`（`work` / `verification`），Run Outcome / outbox / Decision 均挂 `run_id`。强制验收不再绑定 kind，而由 Objective 上的 `verification_required` Latch 决定：任一 root `RunPlan` 一旦出现过 ≥1 个 milestone（任意状态，含 abandoned）即置 true 且只增不减。入口仅做一次非关键复杂度评分（0–10，默认阈值 >3）以建议规划，不置验收标志。收尾调用保留但分档；简单路径解析失败直接机械交付且无 correction。切换破坏性、无 migration，合并前清理开发库。

## Status

superseded by ADR-0048

## Considered Options

- 保留 Assignment，仅放宽 work 交付与模板：改动小，但继续维持与 Run 近乎同构的第三套生命周期，公开概念仍过重。
- 入口分数直接强制验收：实现简单，但与真实 trajectory 脱节，易误伤或漏验。
- 删除 Assignment 并以 Run 为边界，验收由「RunPlan 出现过 milestone」Latch：减少一等概念，硬约束跟执行事实走；入口分数只做规划建议。选此。

## Consequences

- 废除 `intake` 与 AssignmentStatus；旧 ADR 中以 Assignment 为主语的表述由本 ADR 与 `CONTEXT.md` 覆盖处为准。
- Objective 对 Scheduler 只依赖窄 facade（dispatch / wait / inject / pause-resume / tree），不经 mailbox 推进工作。
- `ObjectiveUserInput` 必须先入 Session 历史再写 ObjectiveLog；history_gap 停止空转重试。
- LLM 用量以 root Run / Outcome 边界汇总进 ObjectiveLog；attempt 明细留在 RunLog/trace。
- 实现前清空含 `assignment_id` 的开发状态；不写双读兼容层。
