# P5：Objective Gateway 与 Console

本阶段把已经完整的 Objective 生命周期暴露给用户。Session 仍是用户看到的长期对话容器；Objective 是 Session 内可查看进度、预算和结果的工作项。普通用户路径从本阶段开始统一通过 ObjectiveService。

## 入口条件

- P2 至 P4 的正常、预算、恢复、故障和运行中用户输入主线均可通过 SDK 集成测试。
- ObjectiveLog 与 RunLog 的查询边界已经稳定。
- Console 尚未切流，现有 Session/Scheduler 路径仍可作为回归参照。

## 任务顺序

| ID | 任务 | 依赖 |
| --- | --- | --- |
| P5-01 | 异步 Objective HTTP API | P2-06、P3-05、P4-04 |
| P5-02 | 可重放 Objective SSE | P1-03、P5-01 |
| P5-03 | Session、Web 与渠道入口适配 | P5-01、P5-02 |
| P5-04 | Objective 时间线与最终交付视图 | P5-01、P5-02 |
| P5-05 | Assignment 模板管理界面 | P2-01 |
| P5-06 | Session 归档、恢复与 Fork | P3-05、P5-01、P5-03 |

P5-04、P5-05 可与 P5-03 并行开发；P5-06 必须在入口切换语义稳定后合并。

## 阶段出口

- `POST /objectives` 立即返回 objective_id；断开 HTTP/SSE 不影响 Objective 生命周期。
- Objective SSE 可以从 sequence 补发，不发送未验收的 Run token delta。
- `/sessions/{id}/input` 和 Feishu 普通消息不再直接调用 `Scheduler.route_root_input()`。
- 一个 Session 同时最多一个非终态 Objective；终态 Objective 不重开。
- Objective 完成后显示正式交付内容，过程时间线默认折叠；其他状态继续显示进度和所需用户动作。
- 模板可校验、预览、保存，并随默认 AgentConfigRecord 持久化。
- 普通删除变为可恢复归档；含活动 Objective 的 Session 先安全暂停。

