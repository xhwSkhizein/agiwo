# Objective 使用 DRAINING 收敛到可恢复中断

用户暂停/停止或预算触顶需要停下整个 Objective 时，执行系统先短暂进入 `DRAINING`：立即关闭新模型调用、tool、handoff、Assignment 创建和 outbox 派发，允许已经开始的在途操作完成，再使活动 Run 进入 `PAUSED`，Objective 投影为 `USER_PAUSED` 或 `BUDGET_PAUSED`。DRAINING 不用于运行中用户输入（见 ADR 0023）。

## Status

accepted

## Considered Options

- 保持 RUNNING 直到分支自行结束：状态少，但无法阻止新派发与新调用。
- 立即写终态或走现有 cancel：快，但会失败化 Run，无法按可恢复中断恢复。
- 为预算与用户暂停分两套过渡状态：名称直观，但机制相同，徒增枚举。

## Consequences

- DRAINING 只服务可恢复中断的收敛，不服务 steering。
- ObjectiveDrainStarted 记录 reason（封闭：`budget` / `user_pause` / `user_archive`）、触发来源、当时活动 run ids；重复触发幂等合并。`user_archive` 与 `user_pause` 共用可恢复中断 barrier，稳定后投影为 `USER_PAUSED`；归档标记本身不属于本 ADR，由 Session 归档流程在 pause 之后提交。
- 进入 DRAINING 后拒绝新的 Assignment/outbox/dispatch；LLM/tool/spawn 启动前检查可推进状态。
- 在途操作完成后，活动 Run 进入 PAUSED（不写 Outcome、不结束 Assignment）；barrier 以 Run PAUSED 为准。
- 运行中用户输入直接注入 root Run，不经过本状态。
- 不引入 ObjectiveActionLease；接受极窄残余竞态（每分支至多一个已越过检查点的额外动作）。
