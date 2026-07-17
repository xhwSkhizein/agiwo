# Assignment 模板随默认 AgentConfig 持久化

Console 对 `intake`、`work`、`verification` 模板的保存不是仅修改进程内 runtime override。模板作为系统默认 AgentConfigRecord 的一部分写入 AgentRegistryStore；持久化成功后，Console 再刷新内存中的有效配置。进程重启时，AgentRegistry 中已保存的默认 record 优先于环境默认值。持久化能力服从选用的 AgentRegistryStore：SQLite 等持久化后端跨重启保留，显式使用 in-memory backend 时只在进程生命周期内保留。

## Status

accepted

## Considered Options

- 仅使用 RuntimeConfigService 的内存 override：改动最少，但 Console 显示“保存”后重启即丢失，不符合配置预期。
- 为模板建立独立持久化与版本仓库：职责清楚、历史完整，但第一版增加额外存储模型和管理界面。
- 把模板塞入 AgentOptions：可以复用现有 JSON 字段，但 Assignment 输入模板不是 Agent.run 行为选项，会污染 SDK AgentOptions 语义。

## Consequences

- AgentConfigRecord 增加类型化 assignment_templates，固定包含四种 Assignment kind；runtime SDK AgentConfig 不需要把模板传给 Agent.run，Objective 输入装配层在实例化前读取它们。
- AgentRegistryStore 的 memory 与 SQLite 实现同步读写该字段。SQLite schema 直接更新；按项目约定不提供 migration，旧开发数据不兼容时要求清理重建。
- Console 保存流程先校验固定占位符并预渲染示例，再持久化 AgentConfigRecord；只有持久化成功才切换内存有效配置。
- 如果持久化成功而内存刷新失败，持久化 record 仍是真相源；服务报告刷新失败并从 Registry 重新加载，不能反向覆盖已保存配置。
- 默认 agent 尚未持久化时继续使用环境/代码 fallback；首次从 Console 保存会在 Registry 中创建同 id record，后续启动优先使用它。
- 配置修改只影响新创建的 Assignment。运行中 Assignment 使用创建时保存的 config/template hash 与渲染输入。
- 模板修订不单独保存历史版本；AgentConfigRecord.updated_at 与内容 hash 标识当前版本，历史执行通过 ObjectiveLog/RunLog 快照自证。
