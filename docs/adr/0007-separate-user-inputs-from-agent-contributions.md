# 分开记录用户输入与 agent 贡献

Objective 分开保存权威的 `ObjectiveUserInput` 与派生的 `ObjectiveContribution`。每条 `is_user_provided=true` 的语义消息只保存一次，作为不可拆分、不可改写的 ObjectiveUserInput，并保留稳定 input id 和完整 UserMessage；pause、resume、预算调整等结构化控制命令不属于语义输入。agent 在 Assignment 收尾时提炼的验收建议、边界发现和经验进入 ObjectiveContribution，不能晋升或伪装成用户事实。

## Status

accepted

## Considered Options

- 在用户消息之外再建立一份要求模型：可以提供更短的验收清单，但必须依赖程序或模型判断自然语言中哪些片段算要求，既重复存储又引入新的权威性争议。
- 自动把 agent 贡献合并进用户侧事实：后续 agent 使用方便，但会让模型在无人授权时改变 Objective 的权威目标。
- 每条贡献都立即请求用户确认：权威边界最清楚，但会制造大量不必要的中断，破坏动态接力效率。
- 不记录 agent 贡献：模型简单，但后续执行会重复发现同一边界和经验，长链路也更容易丢失有效信息。

## Consequences

- ObjectiveUserInput 是 Objective 唯一的用户侧权威事实。系统不从自然语言中分句、分类或抽取更高等级的要求，也不存在候选确认或贡献晋升流程。
- ObjectiveUserInput 保存 `input_id`、完整 `UserMessage`，以及可选的 `in_reply_to_message_id / related_outcome_id` 等关联。agent 提问后的用户回复形成新的 ObjectiveUserInput；关联可让系统同时找到问题与回复，但不创造第二种领域对象。
- 活动 Objective 的每条用户输入必须在模型上下文中恰好出现一次（未外置保留完整原文，已授权外置只保留 path/summary）。正常路径下它们已在 Session 已提交历史中按出现顺序存在；跨 Assignment 只按 `input_id` 去重复用该前缀，新系统内容追加尾部，禁止把用户原文复制进 Assignment 模板，也禁止因「日志有、历史无」而向中段静默插入。若出现后者，按不变量故障处理（见 ADR 0036）。ObjectiveContribution 另行分区，不能混成一段权威摘要。
- 结构化 ObjectiveUserInput 只服务于系统存储、投影、关联和确定性校验。模型接收的是普通文本或 path/summary，不需要理解 input id、fact kind 或内部对象 schema。
- Context Optimization 在模型物理上下文允许时不得删除、摘要或改写 ObjectiveUserInput，正式 compaction 优先处理 assistant/tool 历史。若仅未外置的 ObjectiveUserInput 已超过模型上下文上限，系统在创建或派发下一 Assignment 前写 `ContextCapacityExceeded` 并令 Objective 进入 WAITING_USER，不能静默压缩用户原文。
- 用户可以通过引用 `input_id` 的幂等结构化命令授权外置。系统把原文写入 `{agent_workspace}/sessions/<session_id>/artifacts/` 下的稳定文件，登记 Artifact（path、summary、`source_input_id`、content_hash），并追加 `ObjectiveUserInputExternalized(input_id, artifact_id, authorized_at)`；ObjectiveLog 中的原始 ObjectiveUserInput 仍保留。控制命令不产生新的用户语义事实。
- 外置授权提交后，后续模型上下文在原输入位置只包含该 Artifact 的 path 与 summary；agent 需要全文时按 path 读取。系统重新执行容量检查；足够时才创建或派发待执行 Assignment，不足时继续 WAITING_USER。未经授权不得自动外置、摘要或选择替代内容。
- 接棒 agent 负责判断 ObjectiveContribution 的语义价值；ObjectiveService 记录并消费结构化决定，Scheduler 只把已经确定的 Assignment Input 派发给 Run。
- ObjectiveContribution 的原始内容不可变；接棒 agent 只能在收尾 JSON 的 `contribution_annotations` 中引用已有 `contribution_id` 追加备注（可选 `deactivate`），不能改写原文。系统在提交时填充 `from`（assignment_id）与 `time`。
- 新建贡献只能出现在收尾 JSON 的 `new_contributions[{content, summary?}]`；与 `contribution_annotations` 分开校验，不得混为同一数组。
- 停用只改变 Contribution 是否继续进入活动上下文；原始 Contribution 和全部备注仍保留以支持调试和复盘。
- 线格式细节见 ADR 0006；文件 Artifact 规则见 ADR 0045。
