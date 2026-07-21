# Assignment 以唯一结构化 Decision 结束

每个成功结束、且需要 agent 决定后续路线的 Assignment，必须提交普通文本 report（说明做了什么）、可选的文件型 Artifact 引用，以及一个结构化终态 `Decision`。Decision 承载下一步执行意图；Artifact 只索引工作目录中的独立文件。ObjectiveService 是校验并消费 Decision 的唯一 owner；Scheduler 只执行由 outbox 确定的 Run 派发。两者都不能从 Artifact、自然语言输出或流式文本中推断路由。

## Status

superseded by ADR-0047

## Considered Options

- 从 agent 的自然语言结尾推断下一步：对 agent 约束较少，但会把语义理解泄漏到 Scheduler，并产生不可重放的隐式控制流。
- 允许一次提交多个并列 Decision：表达能力更强，但会引入顺序、原子性和冲突处理问题；并行组合应由已有 pattern 原子表达。
- 让 Artifact 同时携带路由字段或把文本 report 建成 Artifact：对象较少，但工作内容、文件存储与执行控制无法独立校验、保存和演进。

## Consequences

- 一个 Assignment 可以经历等待、工具调用和多个 Run，但只有成功终结 Assignment 的 Run 提交终态 Decision。
- 每个结束的 Assignment 无论正常完成、中断或故障都必须提交唯一 AssignmentOutcome。成功的语义执行在 Outcome 中包含一个 Decision 与普通文本 report；由用户输入或故障策略确定下一步的中断 Outcome 可以携带机械 Decision，或由外部 command 直接驱动后续动作。
- Budget Pause 与其他 checkpoint 不结束 Assignment，因此暂停时不提交 Outcome；恢复后继续直到 Assignment 最终结束并产生 Outcome。
- 对于需要 agent 决定后续路线的成功 Outcome，缺失 Decision、结构不合法或提交多个冲突 Decision 都属于协议错误，而不是可供 ObjectiveService 或 Scheduler 猜测的模糊输出。
- Parallel 等 pattern 可以在 Assignment 内部执行；pattern 返回后，外层 Assignment 仍只提交一个终态 Decision。
- Decision 必须使用封闭、可校验的结构；HandoffDecision target 只允许 `agent / verifier / user`，其中 user 使用 `expects_reply` 区分等待输入与最终交付。
- 文件型 Artifact 的定义、落盘与按需读取见 ADR 0045。
