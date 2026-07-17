# P2：Assignment 可靠派发与正常主链

本阶段把 Objective 内核接到 Scheduler 和 Agent，先完成没有预算耗尽、没有基础设施故障的正常闭环。完成后，SDK 内部应能可靠地经历入口理解、工作、独立验收和最终交付。

## 入口条件

- P0、P1 阶段出口全部通过。
- ObjectiveStore 可以事务提交 AssignmentCreated 与 DispatchRequested。
- Agent 内部已具备 RunPlan、调用 phase 和用户输入来源。

## 任务顺序

| ID | 任务 | 依赖 |
| --- | --- | --- |
| P2-01 | Assignment 模板配置与渲染 | P1-01、P1-04、P0-03 |
| P2-02 | 稳定 Run identity 与内部执行请求 | P0-04、P1-01 |
| P2-03 | Outbox dispatcher 与幂等派发 | P1-03、P1-04、P2-02 |
| P2-04 | Assignment Input 与前缀安全上下文 | P0-03、P2-01、P2-02 |
| P2-05 | 计划门禁与系统收尾调用 | P0-01、P0-04、P2-02、P2-04 |
| P2-06 | 正常接力、验收与最终交付 | P1-02、P2-03、P2-05 |

P2-01 与 P2-02 可以并行；P2-03 和 P2-04 在各自依赖满足后也可并行。P2-05、P2-06 按顺序合并。

## 阶段出口

- ObjectiveService 创建 Assignment 时预分配唯一 run_id；重复派发不会产生第二个逻辑 Run。
- committed RunStarted 必须映射为幂等 `AssignmentExecutionStarted` Objective fact；只有该 fact 令 Objective/Assignment 进入 RUNNING。
- `Agent.start/run/run_stream` 公开签名保持不变。
- Assignment Input、模板 hash、最终渲染快照和实际模型输入都可追溯。
- root Run 的无 tool call 响应先经过 RunPlan 门禁，再由系统必然发起结构化收尾调用。
- 每个终态 Assignment 恰有一个 Outcome；暂停路径在本阶段仍不启用。
- `target=agent / verifier / user` 不包含具名 agent、config 或 pattern。
- 正常路径可以重放：intake -> work -> verification -> ObjectiveDelivered。
