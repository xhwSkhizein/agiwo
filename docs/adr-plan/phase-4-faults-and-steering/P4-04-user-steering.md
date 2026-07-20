# P4-04：实现运行中用户输入注入

状态：done

## 目标

当 Objective 的活动 root Run 仍在执行时收到自然语言用户输入，先持久化 ObjectiveUserInput，再向**同一** root Run 注入一条系统提示型 user 消息（`is_user_provided=false`），使模型在当前 loop 中继续推理。不结束 Assignment、不写 Outcome、不创建新 Assignment/Run，也不进入 DRAINING。

## 对应决定

- ADR 0023：运行中输入注入同一 root Run。
- ADR 0042：注入消息不是用户气泡，不冒充 ObjectiveUserInput。
- ADR 0007：用户贡献与系统提示型消息分离。

## 依赖

- P2-04、P2-02 已完成；可与 P3 并行，但不依赖 DRAINING。

## 范围

包含：输入幂等、ObjectiveUserInput 写入、向 Scheduler/Agent 注入 false user、Console/SSE 展示边界、与 WAITING_USER 回复路径的分流。

不包含：可恢复中断（P3-04/P3-05）；WAITING_USER 创建新 Assignment（P5-01/P2-06）；finalization。

## 实施步骤

1. Gateway `POST /objectives/{id}/inputs`：Objective/Assignment/Run 均为进行中时，先写 ObjectiveUserInput（幂等键防重复），再请求向当前 root Run 注入消息。
2. 注入消息形态：`UserMessage`，`is_user_provided=false`，`origin=running_user_input`（或等价）；正文含 `<system-notice>`（说明用户发来新指令，请检查与目标一致性及规划是否需更新）+ 用户原文；不复制为第二条 ObjectiveUserInput。
3. 注入路径消费 **P2-02 已定义的** `inject_user_message`（或等价 facade 方法）：只追加到当前 root Run 的 llm context 尾部；不得重排已提交前缀；child Run 不单独接收用户输入；本任务不另起平行注入 API。
4. 明确不做：DRAINING、Steering Outcome Synthesis、kind=steering Assignment、结束旧 Assignment、写 AssignmentOutcome。
5. WAITING_USER 路径保持独立：用户回复可创建后续 Assignment；不得与运行中注入混淆。
6. 若 Objective 已处于 USER_PAUSED / BUDGET_PAUSED：拒绝“注入”语义，或定义为先 resume 再处理（策略在实现时钉死并测；默认建议：暂停中的输入先落 ObjectiveUserInput，resume 后按 WAITING/继续规则处理，不在 PAUSED 时注入死 Run）。
7. Timeline：用户气泡来自 ObjectiveUserInput；注入消息可在开发 Trace 中按系统提醒样式展示，不冒充聊天用户气泡。
8. 幂等：相同 idempotency key 不重复写 input、不重复注入；不同 key 的连续输入各自写 log 并各自追加注入（无全局 drain，故不需要 DRAINING 合并窗；渠道侧仍可用现有 Feishu batch 降低噪声）。

## 主要改动位置

- `agiwo/objective/service.py`
- `agiwo/objective/input.py`（或等价）
- `agiwo/scheduler/` steer / enqueue 路径
- `agiwo/agent/run_loop.py`（消费注入）
- `console/server/routers/` Objective inputs
- `tests/objective/test_running_user_input.py`

## 测试计划

- RUNNING 中输入：一条 ObjectiveUserInput + 一次 false user 注入；无 Outcome、无新 Assignment、无 DRAINING。
- 幂等 key 重复：只生效一次。
- 连续不同输入：按序追加多条注入，不创建多个 Assignment。
- child 运行中：提示进入 root，不直接写入 child context。
- WAITING_USER 回复不走本路径。
- PAUSED 行为符合步骤 6 的钉死策略。
- Console 不把 false user 显示为用户气泡。

## 完成标准

- ADR 0023 的注入语义可在集成测试中证明。
- 代码与文档中无 steering Assignment / steering_outcome phase / DRAINING(reason=user_steering)。
- objective/scheduler/agent tests 与 lint 通过。

## 风险与回退

若实现再次引入“先 drain 再开新 Assignment”，视为回归。注入必须走已提交 RunLog/消息路径，不能只改内存 queue 却不落可重放事实（若 Scheduler 当前 steer 仅内存，需补齐与 ObjectiveUserInput 的关联与可观测性）。
