# P6：系统验收与发布收口

本阶段不再增加领域能力，而是证明前五个阶段在正常、并发、预算、暂停、故障、重启和用户交互下形成同一套一致系统，并完成发布所需的护栏与说明。

## 入口条件

- P0 至 P5 的阶段出口全部通过。
- Console 普通用户入口已经统一到 ObjectiveService。
- 所有旧配置名和旧计划事实名已删除。

## 任务顺序

| ID | 任务 | 依赖 |
| --- | --- | --- |
| P6-01 | 端到端状态与恢复测试矩阵 | P0 至 P5 |
| P6-02 | 观测、指标与 trajectory review 评估 | P0-02、P3、P4、P5-04 |
| P6-03 | 架构护栏、文档与发布收口 | P6-01、P6-02 |

## 阶段出口

- 45 份 ADR 的每一条核心不变量都有代码位置和自动化测试证据。
- ObjectiveLog、RunLog、AgentState 与 UI 投影在重启和重放后保持一致。
- prefix-cache、LLM 实际成本与并发越界、SSE 重连和 outbox 恢复具有专项回归测试。
- trajectory review 的成本、延迟和结果相关性可观测，但其评分仍只是 compaction 的可选实验输入。
- `AGENTS.md`、`CONTEXT.md`、API 文档、Console 文案与源码术语一致。
- `uv run python scripts/check.py pre-push` 和前端全部门禁通过。
