# Assignment 收尾调用由执行系统发起

当 Assignment 触发的 Run 产生没有 tool call 的 assistant step 时，执行系统先检查当前 root Run 的 `RunPlan`（见 ADR 0038）。若计划中仍有 `pending / active` 项，系统把该 assistant 消息正常写入 RunLog，再向同一个 Run 追加一条固定的 `role=user` 提醒，列出剩余计划项并要求继续执行；此时不发起收尾调用。只有计划中不存在未完成项（或 Run 未声明计划）时，系统才把无 tool call 的 assistant 消息保存为用户可见的普通文本 report（以及 `RunOutput.response`），并在同一个 Run 内必然追加一次收尾模型调用。

收尾调用不是模型自行选择的 tool，也不另开第二个 Run。为保住前缀缓存：收尾指令以新增的 `role=user`、`is_user_provided=false` 消息追加在历史尾部；本次 Provider 请求必须携带与当前 Run 常规 turn **相同的 tools 列表**（不得改成无 tools）。指令要求模型本次只输出结构化 JSON、不要发起 tool call；若仍返回 tool call，按解析失败处理并允许一次纠正。非 Assignment 触发的 Run 与 child Run 继续沿用无 tool call 即正常返回的原有语义。

## Status

superseded by ADR-0047

> 强制系统收尾调用产出 Decision JSON 已废止；下一跳由 NextAction（规则或控制工具）表达。

## Considered Options

- 通过 system prompt 要求所有 agent 自行输出终态结构：改动较小，但普通执行与 Assignment 执行无法可靠区分，模型也可能忘记协议。
- 暴露提交 outcome 的内部 tool：容易复用现有 tool schema，但是否调用的控制权仍在模型，不能保证收尾调用必然发生。
- 等 Run 返回后由 Scheduler 再调用一次 agent：执行边界清晰，但会产生第二个 Run，并失去同一 Run 上下文和统一成本记录。
- 收尾请求去掉 tools 以“强制只出 JSON”：实现简单，但会破坏与先前 turn 共享的 tools 前缀缓存。
- 派发时经 Scheduler 传递完整 finalization prompt/schema：灵活，但让 outbox 版本化大段 prompt，并模糊 Agent/Objective 边界。

## Consequences

- Assignment 身份必须成为 Run 的显式、类型安全上下文（`assignment_role=root`），不能依赖松散 metadata 猜测。root Assignment 启用计划门禁与系统收尾；child / direct Run 不启用。
- 收尾协议（固定 user 指令文本与 JSON schema）由 Agent 包内常量持有；Scheduler 只透传 identity，不携带 Spec。四种 Assignment kind 共用同一套收尾 schema；职责差异只体现在 Assignment Input。
- 计划提醒在模型协议中使用 `user` role，以便继续当前 agent loop，同时使用 `UserMessage.is_user_provided=false` 并记录 `origin=assignment_plan_guard`。它不是用户输入，不产生 ObjectiveUserInput 或 Objective 目标变化，也不在普通聊天界面冒充用户消息。
- 计划提醒明确要求 Agent 在信息不足或方向改变时直接通过计划工具修订当前计划，或在当前计划处理完后向用户请求必要输入；不能把缺少用户信息当作 handoff 给另一 Agent 的理由。
- 计划门禁读取当前 Run 本地 ledger 中的 `RunPlan`，是 finalization 路径的确定性检查，不查询 ObjectiveLog；该检查只对 Assignment 的 root Run 生效。
- 有未完成 `pending / active` 计划时不得进入收尾，不得产生 Decision/Outcome。
- 计划提醒及触发它的 assistant 消息完整写入 RunLog 和调试视图，并作为已经提交的 Session 历史保持原有顺序与内容，以维护跨 Run 前缀缓存。
- 收尾指令同样以尾部追加的 `user` 消息发送（`origin=assignment_finalization`），不得改写或重排已提交前缀；本次请求的 tools 与常规 assistant turn 一致。
- 收尾调用的完整输入、输出、推理内容、用量和解析结果必须写入 RunLog，并能在 Console 调试视图中展示；收尾 prompt 与结构化响应不进入后续普通业务对话上下文。
- 收尾调用不能覆盖用户可见的普通文本 report 或 `RunOutput.response`；report 不是 Artifact。
- 收尾输出由执行系统解析和校验；失败时不能从自由文本猜测 Decision，只允许既定的一次固定纠正调用，随后按最小系统 handoff 规则收束。
- Run 内不提供修改 Objective 的 runtime tool。模型产生的 Objective 修订、新 Contribution 与既有 Contribution 的 annotations 只能通过系统必然发起的 Assignment 收尾调用提交；ObjectiveService 校验后写入 ObjectiveLog。

### 收尾 JSON 线格式（Agent `RunFinalizationResult`）

字段彼此独立；`new_contributions` 与 `contribution_annotations` 不是同一概念。

```json
{
  "decision": {
    "target": "agent | verifier | user",
    "expects_reply": false,
    "reason": "optional string"
  },
  "artifact_refs": [
    { "artifact_id": "optional if already registered" },
    { "path": "sessions/<session_id>/artifacts/..." }
  ],
  "new_contributions": [
    {
      "content": "immutable discovery text",
      "summary": "optional short label"
    }
  ],
  "contribution_annotations": [
    {
      "contribution_id": "existing contribution id",
      "annotation": "supplement / supersede / deactivate reason",
      "deactivate": false
    }
  ],
  "objective_update": null
}
```

规则：

- `decision.target=user` 时必须显式给出 `expects_reply`；其他 target 禁止该字段或必须为省略。
- `artifact_refs` 只引用文件型 Artifact（ADR 0045）；可空。
- `new_contributions`：本 Assignment **新建**的贡献正文；系统分配 `contribution_id` 后原文不可变。可空列表。
- `contribution_annotations`：只针对**已存在**的 `contribution_id` 追加备注；`from`（提交该 annotation 的 assignment_id）与 `time` 由 ObjectiveService 在提交时填充，不由模型编造。`deactivate=true` 只表示不再进入活动上下文，不删除原文与历史 annotations。可空列表。
- `objective_update`：没有新的全局理解时必须为 `null`。非空时形状为：

```json
{
  "expected_revision": 3,
  "intent": "optional string",
  "scope": "optional string",
  "success_criteria": "optional string",
  "assumptions": ["optional"],
  "sources": [
    { "kind": "user_input | contribution | artifact | outcome | objective_fact", "id": "..." }
  ]
}
```

- `objective_update` 只允许意图 / 范围 / 成功标准理解 / 假设及其 `sources`；禁止 RunPlan、handoff、预算、状态字段。`expected_revision` 必须匹配当前 Objective 投影 revision，否则 Outcome 仍可提交，但该 update 以 first-class 拒绝/跳过 fact 记录（见审查项 R14 的后续确认）。
- ObjectiveService 把线格式映射为领域 `Decision`、`ObjectiveContribution`、annotations 与 `current_goal` 修订；不从自然语言猜路由。
- 文件 Artifact 的登记与引用规则见 ADR 0045。
