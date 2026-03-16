# Agent 构造与 child 派生问题阶段性总结

> 更新时间：2026-03-14
>
> 本文只总结当前阶段已经确认的问题、本质判断和处理思路，不覆盖全部 review 细节。逐条问题确认可参考根目录的 `review_confirm.md`。

## 背景

这一轮讨论的起点，是围绕 `Agent` 的公开构造、`derive_child()` 的语义、`close()` 的资源释放，以及 `system_prompt` 的运行时语义展开。

最初看起来像是几个分散的问题：

- `Agent` 公开 API 与内部装配路径混在一起
- child agent 派生时共享父 agent 的部分运行时对象
- `close()` 的语义与共享对象冲突
- `Agent.system_prompt` 这个运行时属性与真实 prompt 生命周期不一致

继续往下分析后，结论逐渐收敛到同一个核心：`Agent` 当前把不同层次的东西混在了一个构造和生命周期模型里。

## 当前聚焦的问题

### 1. 公开构造与内部装配没有分清

当前 `Agent` 仍然被当作公开主入口导出，但构造函数已经更像“内部已装配对象”的签名。这会导致：

- 用户从公开 API 的角度看，`Agent(...)` 的心智模型不稳定
- 内部为了支持 `derive_child()`，开始引入额外的组装分支
- `create_agent()` 和 `Agent(...)` 容易形成双入口和双语义

当前阶段结论：

- `create_agent()` 不应该保留
- `Agent(...)` 应恢复为唯一公开入口
- 不为兼容保留双入口

### 2. `Agent.system_prompt` 运行时属性语义错误

当前的 prompt 真正由 `DefaultSystemPromptBuilder` 持有并懒构建、自动刷新；而 `Agent.system_prompt` 这个同步属性既不是 source of truth，也不能稳定反映当前有效 prompt。

当前阶段结论：

- 删除 `Agent.system_prompt` 运行时属性
- 删除 `Agent._system_prompt`
- 保留构造参数里的 `system_prompt`，因为那是输入配置，不是运行时状态
- 统一通过 `await agent.get_effective_system_prompt()` 获取当前有效 prompt

### 3. child 问题表面上是“要不要共享父资源”，本质上是资源分层不清

最开始的争论点是：

- child 是否应该共享父 agent 的 storage / trace / prompt / skill 等对象
- 如果共享，`close()` 怎么办
- 如果不共享，行为会不会变化

当前阶段结论：

- 核心问题不是 child 是否共享父 agent
- 核心问题是：哪些东西是 agent-local 状态，哪些东西是可共享基础设施
- 只要这两层没有拆开，child 派生、关闭、重建都会持续变复杂

## 当前阶段的本质判断

`Agent` 当前至少混合了三类不同语义的对象：

### 1. 纯配置

例如：

- `name`
- `description`
- `model`
- `options`
- 输入态 `system_prompt`

这类数据可以复制、派生、序列化，不应承载运行时副作用。

### 2. agent-local 可变状态

例如：

- `hooks`
- `DefaultSystemPromptBuilder`
- `tools` 列表本身
- 与当前 agent 实例绑定的缓存、包装、运行时改写

这类对象不能简单共享同一个实例。否则会出现 child 改写 parent、wrapper 污染 sibling、状态缓存串用等问题。

### 3. 可共享的基础设施资源

例如：

- `run_step_storage`
- `session_storage`
- `trace_storage`
- 更广义上的 SQLite / Mongo runtime、连接池、共享 client

这类资源“底层可以共享”，但“不应该通过共享同一个 store 对象实例来表达共享”。

更合理的语义应该是：

- 每个 agent 拿到自己的 resource handle
- handle 背后连接共享的 runtime / pool
- `close()` 释放当前 handle
- 真实连接在 ref-count 归零时再关闭

## 关于 child 不共享资源的阶段性结论

让 child 不共享父 agent 的 resource handle，从语义上更干净，但不是只改 `derive_child()` 一处就够了。

### 可以成立的前提

- 对 `sqlite` / `mongodb` 这类持久化 backend，child 拿自己的 storage handle，功能通常仍然成立
- child 自己使用独立 runtime session，本来就与 parent 的历史隔离

### 需要同步接受和处理的影响

#### 1. `memory` backend 的行为会变化

SDK 默认 `run_step_storage` 仍是 `memory`。如果 child 拥有自己的内存 storage：

- child 运行期间的历史仍然连续
- 但 child 完成并被移出注册表后，这些内存数据会跟对象一起消失

因此：

- `memory` backend 下 child 的详细 step / compact / trace 更像临时态
- 这不是 bug，但属于语义变化，需要明确接受

#### 2. scheduler cleanup 必须显式 close child

当前 scheduler 对 finished child 主要是 `unregister`，并不会 `close()`。

如果 child 改为拥有自己的 storage handle：

- SQLite / Mongo 的 pool ref-count 会随着 child 创建而增加
- 如果 child 完成后不 close，就会变成资源引用泄漏

因此：

- 一旦 child 不再共享父 handle，scheduler 在 child 完成/失败 cleanup 时就应显式 `await child.close()`

#### 3. 不能简单把“已装配好的 tools”再次走公开装配

当前 child 派生直接基于父的 runtime tools 列表。如果改成 `derive_child()` 直接重走公开构造，必须先解决：

- 哪些 tools 是用户输入的原始配置
- 哪些 tools 是装配阶段自动注入的
- skill tool 是否会被重复追加

否则会出现“已装配对象再次装配”的问题。

因此：

- child 重建不能粗暴地拿 `self.tools` 再跑一遍完整装配流程
- 需要先把“原始配置”和“运行时装配结果”拆开

## 当前建议的处理思路

### 第一阶段：先把 API 语义拉直

目标：先消除公开 API 和运行时状态中最明显的混乱。

处理建议：

- 删除 `create_agent()`
- 恢复 `Agent(...)` 为唯一公开入口
- 删除 `Agent.system_prompt`
- 删除 `Agent._system_prompt`
- 仓库内所有读取当前 prompt 的地方统一改用 `await agent.get_effective_system_prompt()`

### 第二阶段：拆开 agent-local 状态 与 可共享基础设施

目标：让 child 派生和资源关闭不再依赖“共享同一个 Python 对象”的隐式语义。

处理建议：

- `hooks`、`prompt_builder`、`tools` 列表始终视为 agent-local
- storage / trace / session 不再通过“父子共用同一个对象实例”表达共享
- 共享应下沉到 runtime/pool 层，而不是停留在 `Agent` 实例层

### 第三阶段：重写 `derive_child()` 的依赖来源

目标：child 的派生规则简单、稳定、无隐式污染。

处理建议：

- child 复用 parent 的配置语义，而不是复用 parent 的 live object
- child 派生时至少重新构建：
  - `hooks`
  - `DefaultSystemPromptBuilder`
  - resource handles
- child 是否共享底层连接由 infra/runtime 决定，而不是由 parent store object 决定

### 第四阶段：把 `close()` 语义收口为“只释放自己”

目标：消除“共享对象导致 close 语义不清”的问题。

处理建议：

- root / child 都只关闭自己持有的 handle
- scheduler 对 child 的 cleanup 补上显式 `close()`
- 不引入额外的 ownership patch 逻辑，前提是 parent / child 不再共享同一个 closable 对象

## 当前不准备做的事

- 不为了兼容保留旧入口或 deprecated alias
- 不继续扩展 `Agent.system_prompt` 的同步缓存语义
- 不在当前阶段优先处理 scheduler 的重复 `_update_status_and_notify()` 逻辑
  - 这一点仍然是问题，但本轮先忽略

## 当前阶段的收敛结论

目前已经达成的阶段性共识可以压缩成 4 句话：

1. `Agent(...)` 应恢复为唯一公开入口，`create_agent()` 应删除。
2. `Agent.system_prompt` 这个运行时属性应直接删除，当前有效 prompt 统一走异步方法。
3. child 问题的本质不是“要不要共享父 agent”，而是 `Agent` 没有分清 local state 和 shared infra。
4. 真正合理的方向是：local state 永远重新构建，shared infra 通过底层 runtime/pool 合理共享，`close()` 永远只释放自己。

## 后续落地时需要重点验证的点

- `Agent(...)` 直接构造是否重新成为稳定可用的公开 API
- `derive_child()` 是否仍然会重复注入 skill / builtin tool
- child 在 `memory` backend 下的历史保留语义是否符合预期
- child 完成后 scheduler cleanup 是否确实调用了 `close()`
- SQLite / Mongo pool 的 ref-count 是否会随着 child 生命周期正确归零

