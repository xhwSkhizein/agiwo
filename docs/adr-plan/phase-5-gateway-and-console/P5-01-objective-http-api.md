# P5-01：提供异步 Objective HTTP API

状态：planned

## 目标

为 ObjectiveService 提供统一的异步 HTTP 边界。创建请求在 Objective 与初始可靠派发提交后立即返回 objective_id，不阻塞到 workflow 结束；查询、输入、预算调整、pause 和 resume 都通过独立幂等 command 完成。

## 对应决定

- ADR 0021：Objective Gateway 使用统一异步 API。
- ADR 0030：八种状态决定允许的命令。
- ADR 0035：Session 是用户顶层容器，一个活动 Objective。

## 依赖

- P2-06、P3-05、P4-04 已完成。

## 范围

包含：Console backend router、request/response models、ObjectiveService adapter、错误映射和 command idempotency。

不包含：SSE、Session compatibility adapter 和前端时间线。

## 实施步骤

1. 新建 Objective router，但 HTTP 层只装配请求/响应，不读取 ObjectiveStore 或 Scheduler。
2. `POST /objectives` 接受 session_id、真实 UserMessage、ObjectiveBudget 和 request_id；容量可行时提交 ObjectiveCreated、初始 Assignment/outbox，单条输入已经超限时提交 ObjectiveCreated/ContextCapacityExceeded 并返回 WAITING_USER。
3. `GET /objectives/{id}` 返回 ObjectiveService 从 ObjectiveLog 投影的当前 view，不拼 AgentState。
4. `POST /objectives/{id}/inputs` 先持久化 input，再由 Service 按状态处理：RUNNING → 注入同一 root Run；WAITING_USER → 用户回复/后续 Assignment；终态 → 拒绝或提示创建新 Objective。
5. `POST /objectives/{id}/inputs/{input_id}/externalize` 只接受已认证用户的明确控制动作，幂等把原文物化到 `{agent_workspace}/sessions/<session_id>/artifacts/`、登记 Artifact（path/summary/hash）并写外置授权；容量重检通过后继续待执行 Assignment。普通自然语言消息不能伪造授权；ObjectiveLog 原始输入仍可查询。
6. `POST /objectives/{id}/pause` 与 `/resume` 使用 P3 checkpoint 语义，不调用 Scheduler.cancel。
7. 提供预算调整 command，显式区分 limit 更新和 active-time 继续确认；模型或普通消息不能伪造该控制命令。
8. 每个 command 要求 Idempotency-Key。scope 固定为 `session:{session_id}:objective:create` 或 `objective:{objective_id}:{command_kind}`；后端规范化语义 payload 并计算 hash，忽略连接级瞬时字段。
9. command receipt 与领域 facts/outbox/Session slot 在同一 ObjectiveStore 事务写入；同 scope/key/hash 返回首次持久化响应，同 key 不同 hash 返回 409 `idempotency_conflict`。create 因使用 Session scope，在 objective_id 产生前也能查重。
10. 状态冲突使用稳定 error code 和当前 view，例如 terminal_not_reopenable、objective_draining、budget_adjustment_required、context_capacity_exceeded。
11. 网络断开不取消 Objective；router task 只等待事务提交，不持有 workflow coroutine。
12. ObjectiveService/Store 异常与普通 agent fault 分开映射；只有领域不可恢复故障才显示 Objective FAILED。
13. API model 放 `console/server/models/` 或 view 边界规定位置，不新增 `schemas.py` 或 Console `domain/`。

## 主要改动位置

- `console/server/routers/objectives.py`
- `console/server/models/objective.py`
- `console/server/dependencies.py`
- `console/server/app.py`
- `agiwo/objective/__init__.py`
- `console/tests/test_objectives_api.py`

## 测试计划

- create 立即返回，不等待模型完成；初始 outbox 已可靠提交。
- get 投影覆盖八种状态、最终文本 report 与文件 Artifact refs。
- input 在 CREATED/RUNNING/WAITING/PAUSED/terminal 的状态矩阵。
- context overflow、externalize command、容量重检和原始输入仍可查询。
- externalize 后：Artifact 落在 `{agent_workspace}/sessions/<session_id>/artifacts/`；后续派发的 Assignment 上下文只有 path+summary；集成测试证明 agent 既有读文件工具能打开该 path 读回原文（与未外置内容一致）；禁止空引用或指向不可达路径。
- pause/resume/adjust budget 的幂等和冲突。
- create 与其他命令的 receipt 跨重启 replay，以及同 key 不同 payload 的 409。
- HTTP disconnect 不改变 Objective 状态。
- router 不直接 import ObjectiveStore/Scheduler internals 的架构测试。

## 完成标准

- 后端只有一套 Objective 生命周期 API，没有同步等待到终态的第二套实现。
- 所有写 command 可幂等重放。
- 当前 view 只来自 ObjectiveLog projector。
- cancel endpoint 未被复用为 pause。
- 外置输入的按需读取路径可验收（与 P2-04 / ADR 0045 一致）。
- Console backend tests 与 lint 通过。

## 风险与回退

不要在 create route 中消费 Run token stream 并把它当成请求生命周期。Console 可以在前端选择短暂等待，但后端创建语义始终异步。
