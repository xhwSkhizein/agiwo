# P2-01：实现 Assignment 模板配置与渲染

状态：planned

## 目标

为 intake、work、verification 三种 Assignment kind 提供固定、可校验、可持久化的输入模板。模板只负责把类型化 ObjectiveView 渲染为本次 root Run 的 UserMessage；共享 planning policy 继续放在默认 AgentConfig.system_prompt 中，不复制到每个模板。

## 对应决定

- ADR 0025：四种职责共享同一默认 AgentConfig。
- ADR 0026、0027：模板可配置并随默认 AgentConfigRecord 持久化。
- ADR 0037：模板消费 current_goal 及其权威依据。
- ADR 0042：系统渲染输入使用非用户来源。

## 依赖

- P0-03、P1-01、P1-04 已完成。

## 当前源码现状

- `AgentConfigRecord` 目前只有 system_prompt、tools、skills、options 和 model params。
- memory/SQLite AgentRegistryStore 都持久化这组字段。
- RuntimeConfigService 有进程内 override，但 ADR 明确要求模板保存后可按 Registry backend 持久化。
- Console 还没有模板校验或预览界面；P5-05 实现 UI。

## 范围

包含：模板 DTO、默认模板、占位符校验、渲染器、AgentConfigRecord/Store 字段、运行时加载优先级和历史快照。

不包含：Console 表单和预览页面、Assignment 上下文历史选择、真正启动 Run。

## 实施步骤

1. 在 Objective public DTO 中定义 `AssignmentTemplateSet`，固定包含 intake/work/verification；Objective 包不导入 Console 的 AgentConfigRecord。
2. 模板只允许固定占位符：current_goal、objective_contributions、objective_budget、assignment_outcomes、assignment_kind。用户输入不属于模板字段；P2-04 按 input_id 复用稳定历史前缀中的 canonical `UserMessage`，不向中段补回。模板不得另行渲染全部或最新用户输入。
3. 实现安全 formatter：不支持 Jinja 循环、条件、属性访问或代码执行；未知占位符和每种 kind 缺失的必需占位符在保存前失败。
4. 为三种 kind 提供代码默认模板；模板只描述本次职责，不复制 planning policy，不允许指定具名 agent、config 或 pattern。
5. `AgentConfigRecord` 增加类型化 assignment_templates；memory 和 SQLite store 同步序列化。按项目约定直接更新 schema，不写 migration。
6. 默认 agent 加载顺序固定为：Registry 已保存 record 优先，未保存时使用环境/代码 fallback。
7. Console 保存服务先校验并用示例 ObjectiveView 预渲染，再持久化 record，成功后刷新内存有效配置。失败时不得只更新内存。
8. ObjectiveService 构造或配置刷新时接收 TemplateSet DTO，不读取 Console store 内部对象。
9. 每次 Assignment 创建时保存 template kind、AgentConfigRecord.updated_at、内容 hash、最终渲染 UserMessage 和相关 Objective revision。
10. 渲染内容只要经过模板包裹就标记 `is_user_provided=false`；若 intake 明确直接使用未经包裹的原始 UserMessage，则保留 true，并在 Assignment 记录中引用同一用户输入 fact。
11. planning policy 更新默认 system prompt：说明动态选择直接执行/Agent/Parallel/Pipeline/pattern，以及 RunPlan 只覆盖当前责任。

## 主要改动位置

- `agiwo/objective/` 的 template DTO 与 renderer
- `console/server/services/agent_registry/models.py`
- `console/server/services/agent_registry/store/memory.py`
- `console/server/services/agent_registry/store/sqlite.py`
- `console/server/services/agent_registry/defaults.py`
- `console/server/services/runtime/agent_factory.py`
- `console/server/services/runtime_config.py`
- `console/server/config.py`

## 测试计划

- 四种模板的必需字段、未知字段和安全 formatter 测试。
- 示例 ObjectiveView 预览与真实渲染完全一致。
- AgentConfigRecord memory/SQLite round-trip 和重启优先级。
- 持久化失败时内存不切换；持久化成功但刷新失败时可从 Registry 重载。
- Assignment 创建时保存的模板/AgentConfig 快照供该 Assignment 全程（含 PAUSED resume）使用；后续 Registry 编辑只影响新 Assignment（ADR 0033）。
- planning policy 不出现在四份模板中，模板不含 executor 选择字段。

## 完成标准

- Assignment 模板不是 AgentOptions 字段，也没有独立模板数据库。
- Registry 保存后，SQLite backend 重启不丢失；memory backend 按其既有语义丢失。
- Objective SDK 不导入 Console。
- 模板修改只影响之后创建的 Assignment。
- SDK/Console backend 测试与 lint 通过。

## 风险与回退

模板属于 Console 持久配置，但 renderer 属于 Objective SDK。用公开 TemplateSet DTO隔开两者；不要为了共享模型让 SDK 反向导入 `server.*`。若 schema 变更失败，清理开发数据库重建，不增加旧列兼容。
