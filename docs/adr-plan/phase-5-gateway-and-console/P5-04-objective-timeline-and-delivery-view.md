# P5-04：实现 Objective 时间线与最终交付视图

状态：done

## 目标

在 Session 页面中以 ObjectiveLog sequence 展示任务如何推进，并允许从 Assignment/Run 节点下钻 RunLog。ObjectiveDelivered 后，普通文本 report 成为主内容、文件 Artifact 作为附件，历史 trace 默认折叠；等待、暂停和失败状态仍显示过程与用户下一步动作。

## 对应决定

- ADR 0020：Objective 时间线与 Run 下钻。
- ADR 0038：milestone board、review cycles、carry_forward 共用 RunPlan 数据。
- ADR 0043：显示 append-only review 与实验评分。

## 依赖

- P5-01、P5-02 已完成。

## 实施步骤

1. 后端提供 Session 的 Objective 列表、当前 ObjectiveView、按 sequence 分页 facts、assignment_id/run_id 下钻链接。
2. 时间线主干只使用 ObjectiveLog sequence，不能用客户端到达时间、AgentState.updated_at 或多个 Run sequence 混排。
3. 节点至少覆盖：current_goal、ObjectiveUserInput、Contributions、Assignment、Decision、Artifact、handoff、budget、active window、drain/checkpoint、fault 和交付。
4. Assignment 节点显示 kind/status/outcome/run refs；Run 明细按需查询现有 RunLog/Trace，不复制到 Objective API。
5. Objective 未交付时，普通用户只看已提交 Objective 进度，不展示 Run token delta 或候选 report。
6. ObjectiveDelivered 后切换最终交付视图：普通文本主 report 在主内容区，文件 artifacts 作为附件列表；Timeline/Trace 默认折叠但可展开。
7. WAITING_USER、BUDGET_PAUSED、USER_PAUSED、DRAINING、FAILED 使用状态专属操作区，不误用最终交付布局。
8. WAITING_USER(reason=context_capacity_exceeded) 显示超限信息、可外置 input 列表和明确授权按钮；授权后时间线同时保留原始输入节点与 Artifact path/summary 关联，不表现为删除或摘要。
9. 开发模式下展示 finalization 调用 phase、固定 prompt、完整 messages、params、output、reasoning、parse errors、correction、usage 和实际成本来源。
10. milestone board 从当前 root Run 的 RunPlanUpdated 投影；completed/abandoned 默认折叠但可查。review cycle 使用同一 milestone identity，不建第二份计划数据。
11. trajectory review 展示 aligned/experience/tool_call_id/usefulness 和 experimental 标签；不显示模型无意义的存储 seq。
12. 所有异步状态有稳定占位和错误/重连行为，长文本/ID 不溢出。

## 主要改动位置

- `console/server/models/session.py`
- `console/server/models/view.py`
- `console/server/services/runtime/session_view_service.py`
- `console/server/services/runtime/runtime_observability.py`
- Objective API query services
- `console/web/src/app/sessions/[id]/page.tsx`
- `console/web/src/components/session-detail/`

## 测试计划

- ObjectiveLog sequence 排序和分页。
- Assignment/Run drilldown 引用正确且不复制 step。
- completed final delivery、waiting、budget pause、user pause、failed 五类视觉状态。
- final report 来自 ObjectiveDelivered 的普通文本，不从最后 assistant message 推断，也不把 report 当成 Artifact。
- finalization debug 完整可见但普通用户折叠。
- RunPlan 单一数据源和 trajectory score 标签。
- 前端窄屏/长文本快照与交互测试。

## 完成标准

- 用户能清楚区分正式交付与过程产出。
- Objective 时间线在刷新/重连后与 projector 一致。
- Run Trace 仍是开发下钻，不淹没主时间线。
- 无第二份 todos/plan 数据源。
- Console backend tests、frontend lint/test/build 通过。

## 风险与回退

不要把所有 RunLog 平铺进 Objective 页面。主线应解释“责任和决定如何推进”，执行细节只在用户主动下钻时加载。
