# Trade-Off 与迁移策略

## 1. 总体判断

这层代码可以明显变得更清晰，也可以减少不少维护代码。

但前提是：

- 不要把“重构”做成“再造一个更抽象的框架”
- 不要为了保持 100% bug-for-bug compatibility 去继承明显错误
- 不要同时追求“分布式调度”“真正消息队列”“多消费者流订阅”等超出当前产品边界的能力

## 2. 关键 Trade-Off

| 选择 | 收益 | 代价 | 我的结论 |
| --- | --- | --- | --- |
| 继续保留 polling-only tick | 简单 | 延迟感强，fast path 分散 | 不选 |
| 改为 event-driven nudge + periodic sweep | 保持简单，同时降低延迟 | loop 实现稍复杂 | 选 |
| 保留 `AgentRunMode + AgentRunSpec` | 兼容现状 | 可读性差，布尔矩阵难扩展 | 不选 |
| 改成 `DispatchAction` | planner / executor 边界清晰 | 需要一次性重写 runner 主干 | 选 |
| 继续用可变 `AgentState` | 看起来改动少 | 后端行为不一致，refresh dance 很多 | 不选 |
| copy-on-write state snapshot | 迁移后更稳、更易测 | 初期要改很多 transition call site | 选 |
| 保留当前单进程 owner | 范围受控，风险低 | 不能直接横向扩容 | 选 |
| 顺手升级为 distributed lease | 长远看更强 | 需求外，复杂度会立刻飙升 | 不选 |
| 保留单槽 `pending_input` | 对齐当前产品语义 | 不是真正多消息队列 | 选 |
| 顺手做真正 inbox queue | 行为更完整 | 明显改变语义与 Console 路由 | 不选 |

## 3. 最难处理的兼容点

真正困难的地方不是状态机，而是下面这些“看起来像行为，实际上是实现瑕疵”的边界。

### 3.1 要不要兼容 `QUEUED` steer 丢消息

我的判断：

- 这不属于值得保留的功能；
- 它只会制造线上歧义；
- 它已经和 Console 的直觉使用方式冲突。

所以我会兼容“公开 API 的存在”，但不会兼容这个丢消息行为本身。

### 3.2 要不要兼容多 store 后端的 mutable alias 差异

不会。

理由很简单：

- 这是内部实现偶发差异；
- 它让 memory backend 和 SQLite/Mongo 的测试意义不同；
- 长期一定会继续制造“测试过了，线上不一样”的问题。

### 3.3 要不要把 `shutdown(RUNNING root)` 的 no-op 也当兼容目标

不会。

这属于 API 语义不成立，而不是“某种用户依赖的功能”。

## 4. 我会明确保留的 Trade-Off

有些限制虽然不完美，但在当前阶段是合理的。

### 4.1 单进程 owner

我会把它显式写进设计，而不是假装现在已经支持多实例。

原因：

- 当前 coordinator 明显是进程内 owner；
- store 没有 lease / CAS / version；
- 这不是“补几行判断”能安全升级的事。

### 4.2 root 下一轮输入继续保持单槽

当前 `enqueue_input()` 的语义本来就不是队列，而是“下一轮输入槽位”。

我会继续保留这个设计，因为：

- 它和 Console 当前路由兼容；
- 它比“半套队列”更容易解释；
- 如果未来真要做多消息队列，应该是单独的产品与 API 设计，不该偷偷塞进这次重构。

### 4.3 stream 先保持单 owner，但要显式化

当前实现已经近似单 owner，只是没有写清楚。

我会二选一：

1. 显式只允许一个 stream subscriber，重复打开直接报错
2. 或者真正支持多 subscriber fanout

如果目标是“更少代码”，我会选 1。

## 5. 迁移步骤

我建议按下面四步做，而不是一次性大爆改。

```mermaid
flowchart LR
    A["补齐行为测试与边界用例"] --> B["引入 runtime/service/executor/planner 新骨架"]
    B --> C["迁移状态迁移与 dispatch 逻辑"]
    C --> D["删除旧模块并收敛文档"]
```

### 第一步：补齐测试，先冻住真正要保留的行为

必须先补的不是“更多 happy path”，而是这些边界测试：

1. `QUEUED` 时再来一条用户输入应该怎样处理
2. `shutdown(RUNNING root)` 的预期行为
3. `custom_child_id` 冲突处理
4. 多 waiter / 多 stream consumer 的预期
5. memory / sqlite store 的等价行为

### 第二步：先搭新骨架，但 facade 不变

先保留 `Scheduler` facade 原样，把内部替换为：

- `SchedulerService`
- `SchedulerRuntime`
- `TickPlanner`
- `SchedulerExecutor`

这一步的重点是：

- 先让 owner 收束；
- 暂时不追求进一步压缩 tool 层。

### 第三步：迁移状态迁移和 dispatch

这一步具体会做：

1. 用 `DispatchAction` 替换 `AgentRunMode + AgentRunSpec`
2. 用 model transition helpers 替换 `state_ops.py`
3. 用 planner 替换 `tick_ops.py + selectors.py`
4. 把树操作收回 `service.py`

### 第四步：删除旧模块

在所有测试过完之后，删除：

- `state_ops.py`
- `tick_ops.py`
- `tree_ops.py`
- `selectors.py`
- `wake_messages.py`
- `formatting.py`

如果 executor 已经稳定，`runner.py` 会被等价替换掉。

## 6. 我预计会减少哪些代码

可以直接删掉或明显压缩的代码主要是这几类：

1. 布尔 spec/模式映射
2. 重复的 state refresh 模板
3. selector + tick dispatch 的双层胶水
4. 树操作独立文件
5. wake message 和 formatting 的拆分胶水
6. agent preparation / child materialization 的重复样板

保守估计：

- top-level scheduler 文件数会减少一截；
- 核心 owner 数会更少；
- 代码行数不会是“腰斩”，但 15% 左右的净减少是现实目标。

## 7. 哪些诱人的事，这次一定不要做

这次重构最容易失控的地方，是顺手把不属于这次范围的能力也带进来。

我会明确拒绝：

1. 顺手做分布式调度。
2. 顺手做真正 inbox 队列。
3. 顺手做多租户 stream hub。
4. 顺手做通用 workflow engine。
5. 顺手把 scheduler tools 重写成 meta framework。

## 8. 最终判断

如果只允许我给一个最重要的判断，那就是：

> 这层代码最需要的不是“更多抽象”，而是“更少 owner、更硬边界、更显式的 dispatch 模型”。

只要坚持这条线：

- 当前功能能保住；
- 代码会更少；
- 扩展成本会比现在低得多。
