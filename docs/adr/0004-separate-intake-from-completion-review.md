# 入口判断与最终验收使用独立 Assignment

`Objective Gateway` 是无语义判断的用户边界：它创建 Objective，并在 Objective 到达可交付终态后返回结果。Objective 开始时创建入口指派，由普通 agent 形成第一项语义决定；任何执行者认为工作完成时只能提交完成提议，系统随后创建独立的验收指派，由同一 Session agent identity 下的一次新 Run 对照原始目标、硬约束和候选结果作出通过、返工或交还用户决定的结构化决定。

## Status

accepted

## Considered Options

- 为入口与验收使用不同 agent identity：上下文更干净，但会失去 Session 已有的对话历史，并偏离当前 persistent agent 的运行方式。
- 允许最后一个工作 agent 直接结束 Objective：延迟最低，但无法建立不可绕过的独立验收边界。
- 由程序规则判断语义完成度：结果确定，但规则无法可靠理解开放式用户目标是否真正得到满足。

## Consequences

- agent 的一次 Run 正常结束只说明当前 Assignment 完成，不等于 Objective 完成。
- 完成提议必须经过验收指派；只有验收产生的通过决定才能令 Objective 进入可交付终态。
- 入口指派与验收指派使用不同 Run，不继续旧 Run；二者复用 Session 的 agent identity 和可见消息历史。
- 当前方案统一使用系统默认 AgentConfig 构建 Session persistent agent；入口、工作和验收的差异完全来自 Assignment kind、ObjectiveView 与固定输入模板，不建立专用 worker/verifier 配置。
- 共享 Session 历史意味着验收不再通过隔离上下文消除自我审查偏差；独立性来自新的 Assignment 职责、固定验收输入和不可绕过的结构化 Decision。
- 验收默认只读取原始目标、硬约束、候选结果和确定性校验结果；只有判断依据不足时才按需读取中间事实。
