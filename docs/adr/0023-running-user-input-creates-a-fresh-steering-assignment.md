# 运行中的用户输入注入同一 root Run

Objective 的活动 root Run 仍在执行时收到自然语言用户输入，Objective Gateway 先把完整输入作为 ObjectiveUserInput 写入 ObjectiveLog，再向**同一** root Run 注入一条系统提示型 user 消息（`is_user_provided=false`，可用 `<system-notice>` 包装说明「用户发送了新指令，请检查与目标的一致性及当前规划是否需要更新」），并带上用户原文，使模型在当前 loop 中继续推理。系统不结束 Assignment、不写 AssignmentOutcome、不创建新 Assignment 或新 Run，也不进入 DRAINING。

## Status

accepted（注入同一 root Run 仍有效；勿再包一层复述用户原文的 system-notice——见 ADR-0047 / CONTEXT.md）

## Considered Options

- 全局 DRAINING 后结束旧 Assignment、合成 Steering Outcome、再开新 Assignment：边界清晰，但状态机与交接成本高，MVP 阶段过重。
- 把输入只记日志、等自然 handoff：实现简单，但当前分支可能长时间按过时目标继续产生成本。
- 注入同一 Run（本决定）：复用现有 Scheduler steer 思路，改动面小，目标事实仍进入 ObjectiveLog。

## Consequences

- 本 ADR 取代此前「运行中输入必须创建全新 steering Assignment / Steering Outcome Synthesis」的决定。
- `POST /objectives/{id}/inputs` 在 Objective/Assignment/Run 均为进行中时：持久化 ObjectiveUserInput，再通过 Scheduler/Agent 内部路径向当前 root Run 追加注入消息；幂等键防止重复注入。
- 注入消息不是 ObjectiveUserInput，不冒充用户气泡；用户原文以权威 ObjectiveUserInput 为准，注入消息只服务模型上下文。
- child Run 不单独接收用户 steering；提示进入承担责任的 root Run。若需收敛 child，沿用现有委派/等待语义，而不是为 steering 新建 Assignment。
- Assignment kind 不再包含 `steering`；运行中输入不是新 kind。
- WAITING_USER 下的用户回复仍可创建后续 Assignment（例如继续 work），那是等待回复路径，不是本 ADR 的运行中注入。
- 用户暂停、预算暂停走可恢复中断（ADR 0029），与运行中输入注入分离。
