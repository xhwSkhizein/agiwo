# Trajectory review 追加纠偏并提供实验性有用性评分

Trajectory review 保持默认启用，但不再删除、隐藏或改写已经进入模型历史的 assistant/tool 消息。每次复盘只评价上一次复盘边界之后的工具调用，在消息尾部追加对齐判断、纠偏经验和以 `tool_call_id` 标识的 0–3 有用性评分；RunLog 可以自行补充 sequence 等存储坐标，但模型协议不暴露这些内部概念。评分是执行 Agent 的实验性自我判断，只能作为 compaction 的可选参考，不具有固定权重，也不能直接决定内容删除或保留。

## Status

accepted

## Considered Options

- 保留现有 step-back：把偏离区间的 tool result 改写成 experience，并隐藏 review tool call/result。上下文更短，但每次改写都会从变更位置打断 LLM 前缀缓存，复盘越频繁，缓存损失越大。
- 完全删除 trajectory review：没有额外调用和评分成本，但也失去当前仍需通过实际任务验证的复盘、纠偏和 compaction 辅助能力。
- 默认关闭实验功能：成本最保守，但无法在正常运行中持续积累是否有效的观测数据。
- 默认启用、只追加纠偏：保留实验覆盖面和完整历史，把真正的历史重建集中到必要的 compaction，同时承认 usefulness score 尚未证明能改善任务质量。

## Consequences

- `enable_goal_directed_review` 重命名为 `enable_trajectory_review`，默认值保持 `true`；不保留旧配置名兼容层。显式关闭时不装配 `review_trajectory`、不生成复盘提醒、纠偏消息或有用性评分。
- 关闭 trajectory review 不影响 `update_plan`、RunPlan、Assignment 完成门禁、carry_forward、RunPlanUpdated 或 Console 计划视图；introspect 不是这些能力的前置依赖。
- 每次复盘窗口由系统根据已提交事实确定，但给模型的结构化请求只列出 `tool_call_id` 与 tool name，不暴露 RunLog sequence、StepView id 或内部区间坐标。
- `review_trajectory` 结构化输出包含 `aligned`、必要时的精简 `experience`，以及本窗口内每个工具调用至多一项 `tool_usefulness`。评分含义固定为：`0` 有害或误导、`1` 低价值、`2` 有用、`3` 关键；执行成功与信息价值分别判断。
- 模型遗漏的工具调用记为 `unknown`，不能默认为 `0`；重复 id、窗口外 id 或非法分数只拒绝对应评分。完整解析结果写入 `IntrospectionOutcomeRecorded`，暂不为每个评分创建独立 RunLog fact。
- 复盘结果作为新的内部消息追加到历史尾部，不删除 `review_trajectory` 调用、不清理先前 notice，也不改写旧 tool result。已经提交的消息保持原有顺序与内容，以维护前缀缓存。
- Compaction 不依赖 trajectory review。存在评分时，只把它作为明确标记为 experimental/unverified 的可选输入；不设置固定权重，不允许程序按阈值确定性删除内容。compaction 必须结合 Objective、RunPlan 和原始证据重新判断，并且不得摘要、改写或删除 ObjectiveUserInput。
- 只有正式 compaction 可以通过 `MessagesRebuilt` 重建模型历史并打断既有前缀。原始消息、复盘结果、评分和 compaction 输入输出继续完整保存在 RunLog 与开发 Trace 中。
- 每次复盘额外产生的模型调用、token、成本、延迟、aligned 比例、评分分布和后续 Objective 结果必须可观测，以便通过实际任务或 eval 判断该实验能力是否值得保留或调整。
