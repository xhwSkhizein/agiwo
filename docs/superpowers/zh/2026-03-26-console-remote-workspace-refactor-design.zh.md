# 2026-03-26 Console 远程工作台重构设计

## 摘要

这份设计将当前的 Console 重构重新定义为一次**结构与产品模型对齐**的工作，而不是单纯的功能扩张。当前阶段的直接目标，是让 Console 真正成为 SDK 开发者可用的远程 agent 操作工作台。核心问题不是表层功能不够多，而是当前的 Console 代码结构与交互模型，还没有和目标产品语义形成清晰对应。

因此，这次重构首先聚焦两个问题：

1. 讲清围绕 session、task、run 的状态模型
2. 让 Console 与 Feishu 共享同一套交互语义

最终结果应当是：Console 作为 SDK 执行事实之上的一层轻量产品投影，而不是另起一套并行执行真相的第二系统。

## 问题陈述

当前仓库包含一个 agent SDK 和一个 Console 控制台，并支持 Feishu 作为远程交互入口。基础能力虽然已经存在，但整体实现仍然过于接近 demo 状态：

- Console 的交互模型还没有与真实使用方式充分对齐
- Web 端体验还不够强，难以支撑长期使用的开发者工作台定位
- Console 与 Feishu 暴露的是相近能力，但抽象层不统一
- 当前 Console 的实现复杂度，相比底层产品模型的清晰度来说过高

当前最紧迫的事情不是继续加功能，而是先降低 Console 侧复杂度，并让实现重新对齐它想承载的远程操作台产品模型。

## 目标

### 主要目标

将 Console 重构为一个面向 SDK 开发者、可维护的远程工作台：产品模型以 session 和 task 为核心，运行模型统一通过 scheduler 路由执行。

### 目标用户

本阶段的首要用户是 SDK 开发者，他们会用 Console 来：

- 远程与 agent 交互
- 检查执行过程中到底发生了什么
- 在多个进行中的工作上下文之间切换
- 通过 Console 或 Feishu 连续地推进同一类工作

### 范围内结果

- 定义一套 Console 与 Feishu 共享的统一交互模型
- 让 Session / Task / Run 的语义变得明确且一致
- 让 Console 成为 SDK 层执行事实的投影视图
- 降低当前 Console 实现的架构复杂度
- 保留远程执行能力，并为后续观测能力增强留下空间

### 范围外事项

- 与本次重构无关的大范围新功能扩张
- 在这份 spec 中直接改造 SDK 的执行模型
- 在这份文档里直接决定是否完全移除 SDK Trace
- 在默认主流程里暴露大量底层执行细节

## 产品模型

### 核心领域对象

#### Session

Session 是一次完整对话上下文，是远程工作台模型中的主容器。用户应当可以跨入口创建、切换、恢复 session。

#### Task

Task 是 Session 中的一个工作单元。Task 是一等领域对象，但在默认工作流里，它不应当表现为一个需要用户频繁手动操作的一等动作。

正常情况下：

- 用户进入或切换到某个 session
- 用户发送第一条消息
- 系统为该 session 隐式创建当前 task
- 后续消息默认追加到当前活跃 task

因此，虽然 Task 在数据模型里是显式对象，但用户默认感受到的心智模型仍然应该保持简单。

#### Run

Run 是 task 工作在执行层面的落地产物。Console 应当把 Run 视为一种由 SDK 执行事实派生出来的执行视图，而不是一个由 Console 独立维护的业务真相对象。

## 默认交互语义

### Session-First 工作流

Console 与 Feishu 的统一工作流应当是：

1. 选择或创建一个 session
2. 在该 session 中发送消息
3. 如果当前 session 还没有相关活跃 task，则隐式创建 task
4. 统一通过 scheduler 路由执行
5. 将 SDK 产出的执行事实投影回 UI，形成 task/run 视图

### 默认一 Session 一 Task

默认预期应当是：

- 一个 session 对应一个 task

这样可以让产品心智保持简单，并符合大多数实际使用场景。

### 多 Task 仅用于强相关串行工作

只有当工作在上下文上强连续、且明显属于同一条对话脉络的串行任务时，一个 session 才允许承载多个 task。这应当是例外，而不是主模型。

重构应避免让普通用户在日常使用中感觉自己在操作一个 workflow engine。

## Fork 模型

Fork 是一项关键产品能力。

当用户正在处理 task A，过程中发现了一个相邻但相对独立的 task B 时，系统应当支持把当前上下文 fork 到一个新的 session，而不是鼓励把异质工作不断塞进同一个 session。

### Fork 行为

- 创建一个新的 session
- 在新 session 中创建一个新的 task
- 将选定的上下文从源对话中拷贝过去
- 保持新 session 与源 session 在运行时上完全独立
- 仅保留源与 fork 后分支之间的弱关联引用

### 弱关联规则

Fork 可以保留如下 lineage 元数据：

- source session id
- source task id
- fork 时使用的摘要或选定上下文快照

但 fork 不应引入跨 session 的共享 live runtime state。它是一次上下文拷贝，不是一次运行态合并。

## 统一入口架构

目标运行交互路径应为：

**Console / Feishu → Session application layer → Scheduler → Agent**

### 这条架构意味着什么

这表示：

- Console 和 Feishu 都只是入口适配器
- 产品语义不应由入口本身定义
- 通过 scheduler 执行应当成为统一且默认的执行路径
- Console 不应再保留“直接对 agent 对话”作为其主要产品语义

这种统一是必要的，因为当前 Console 与 Feishu 之间的产品语义不一致，正是状态模型混乱与实现复杂度升高的重要来源之一。

## 投影边界

这次重构有一条非常关键的设计规则：

**Console 只构建 SDK 层事实之上的视图，不创建第二套执行真相模型。**

### SDK 事实来源

当前相关的 SDK 执行事实主要有：

- RunStep
- Trace

对于这次重构，Console 应基于 SDK 已经提供的执行记录做投影，而不是自己发明新的 authoritative runtime object。

### RunStep-First 方向

当前设计应优先采用 RunStep-first 的投影策略。

这意味着：

- 面向回放的 message/timeline 视图应主要基于 RunStep 构建
- task 与 run 的摘要应当是 SDK 执行状态上的投影
- 低层执行细节在产品表面应保持为次级、可选信息

### Trace 决策暂缓

Trace 与 RunStep 之间可能存在足够大的重叠，未来完全有可能进一步收敛，甚至让 Trace 被削弱或移除。

但这份文档不直接把它定成结论，而是记录如下决策：

- 本次 Console 重构不能依赖“Trace 必然是长期主真相源”这个前提
- 是否可以让 RunStep 完全覆盖当前 Trace 用途，需要单独做一次技术调研

这样做的好处是：它能让这次重构与未来可能的简化方向保持兼容，但不会把尚未验证的问题提前绑定成架构承诺。

## UI 暴露原则

默认的用户界面应突出产品模型，而不是执行内部机制。

### 默认视图

用户主要应看到：

- sessions
- 当前 session 下的当前 task
- task 状态
- 当前或最近结果摘要
- 继续推进工作或 fork 为新 session 的能力

### 次级调试视图

只有在用户主动查看时，才展示更细的执行机制，例如：

- run timeline
- step 序列
- tool 活动
- 类似 trace 的诊断细节

这样既能保留 SDK 开发者需要的调试能力，又不会让主路径被底层执行细节淹没。

## 服务边界方向

重构后，Console 侧应收敛为一组更小、更明确的服务边界。

### 目标职责

#### SessionService

负责 session 生命周期相关能力，例如 create、switch、resume、lookup。

#### ConversationService

在 session 上下文中接收用户输入，决定对应的 task 语义，并统一通过 scheduler 路由执行。

#### TaskProjectionService

基于 SDK 执行事实与 session/task 状态，构建面向用户的 task 视图。

#### RunProjectionService

构建 run 级别的回放与诊断视图，但自身不能演变为第二套执行状态机。

#### ChannelAdapter

Console 与 Feishu 都作为各自的入口适配器存在，负责把入口特有的协议和身份语义翻译到共享 application model 中。

### 边界约束

- entry adapter 不应拥有产品语义
- projection service 不应变成权威运行时 owner
- scheduler 仍然是统一编排入口
- Console 应继续对齐 SDK 合约，而不是在外层并行重写执行逻辑

## 对当前 Console 结构的影响

这次重构应有意识地摆脱这样一种现状：channel-specific code path、路由决策和执行语义分散在多个部分重叠的 service 中。

目标不只是“挪文件”，而是让以下事情成立：

- session-based remote interaction 只有一条清晰主路径
- Console 与 Feishu 的差异仅存在于 adapter/protocol 边缘
- Session / Task / Run 从 API 层到 projection 层保持一致表示
- 执行真相继续留在 SDK 数据与 orchestration 组件中

## 测试策略

这次重构应在三个层次上验证。

### 1. 领域与服务语义

测试应验证：

- session 切换行为
- 隐式 task 创建行为
- 一 session 一 task 的默认行为
- fork 的 lineage 行为与独立性
- scheduler 作为默认执行路径的路由语义

### 2. 投影正确性

测试应验证：

- task 视图能正确从 SDK 执行事实投影出来
- run 回放视图与 RunStep 支撑的执行数据保持一致
- 详细视图仍是可选的，不会泄漏到默认主流程模型中

### 3. Channel 一致性

测试应验证：

- Console 与 Feishu 产出等价的 session/task 语义
- channel-specific 行为仍然被限制在 adapter 级逻辑中
- 用户切换 session 或持续推进工作时，identity 与 session binding 保持正确

## 风险与权衡

### 风险：在 UI 里过度显式化 Task

因为 Task 是一等领域对象，容易出现过度暴露它的倾向，导致默认工作流变重。

缓解方式：

- 在模型层保持 task 显式
- 在默认交互流中让 task 大体保持隐式
- 只有在调试或工作流控制确实有帮助时，才暴露更强的 task 操作能力

### 风险：重构再次造出一套并行运行模型

如果 Console 侧的 projection logic 开始承载权威执行状态，这次重构就会重演它想解决的复杂度问题。

缓解方式：

- 把 SDK 执行记录视为 source of truth
- 明确所有 Console 侧 run/task 视图都是 derived
- 让执行编排继续收口在 scheduler 与 SDK runtime 中

### 风险：过早决定移除 Trace

未来把 Trace 收敛进 RunStep 也许是正确方向，但如果在缺乏证据前先锁死，会扭曲本次重构。

缓解方式：

- 让 Console 重构兼容 RunStep-first 的未来
- 把 Trace vs. RunStep 的取舍单独拆成技术调研

## 推荐实现顺序

1. 在 Console domain model 与 service contract 中固化 Session / Task / Run 语义
2. 统一 Console 与 Feishu 的入口语义，收敛到 session-first、scheduler-mediated interaction
3. 简化路由，让 ConversationService 成为主要的 input-to-scheduler path owner
4. 基于 SDK 事实引入 projection-oriented 的 task/run view 构建
5. 把 fork 支持作为显式的跨 session 工作流能力加入
6. 加强围绕 session/task 语义和 channel 一致性的测试
7. 单独调研是否可以用 RunStep-only projection 替代 Trace

## 最终设计决策

这次重构应被视为一次**产品模型对齐型重构**，而不是一次功能冲刺。

重构后的 Console 应成为一个面向 SDK 开发者的远程工作台：以 session-first 交互、隐式 task 创建、scheduler 统一执行、fork 到新 session 的分支能力，以及基于 SDK 执行事实的投影视图为核心。