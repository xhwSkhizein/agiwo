# Assignment 计划复用 RunPlan

Objective 不引入独立的 Assignment todo 模型。现有 milestone 计划能力统一命名为 `RunPlan`，由 `agiwo.agent.plan` 通过内建系统工具 `update_plan` 管理，并以 `RunPlanUpdated` RunLog fact 记录；`agiwo.agent.introspect` 只读取当前 RunPlan 作为轨迹复盘锚点。它就是当前 Assignment root Run 的工作计划。Objective 层不再拥有第二份清单，只作为该计划的读侧消费者增加两个行为：完成门禁与中断时的 `carry_forward` 快照。系统中始终只有一个计划模型、一个计划工具、一条计划事实流，自省（trajectory review）、完成门禁和 Console 计划视图消费同一份数据。

## Status

superseded by ADR-0046

> 只保留 RunPlan；不存在 Assignment Plan 别名聚合。

## Considered Options

- 引入 Objective 层独立的 Assignment todos（本 ADR 的前一版本）：门禁与交接语义清楚，但与既有 milestone 机制形成两套"agent 自述计划"。模型面对两个语义重叠的工具会更新其一而遗漏其二，`<system-review>` 锚定的 milestone 与门禁检查的 todo 互不相识，Console 出现两个不一致的计划视图。
- objective-managed Run 中禁用 milestone/introspect，只保留 todos：消除重叠最直接，但 Objective 管理的恰恰是最长、最容易跑偏的执行，等于在最需要的地方拆掉已经建好的轨迹自省与上下文修复护栏。
- 把计划管理并入曾考虑过的 Objective runtime update tool、在 objective-managed Run 中停用既有 milestone 计划工具：工具归属集中到 Objective 层，但会迫使 agent 层理解 Objective 语义，破坏 `objective -> scheduler -> agent` 依赖方向；非 Objective 路径的普通 Run 也会失去计划工具。
- 保留两套模型并在文档中约定用法：改动最少，但两份真相的漂移无法靠约定阻止，长期维护成本最高。

## Consequences

### 所有权与依赖方向

- 计划属于正在执行的 Run，归 `agiwo.agent` 所有；Objective 层不在 ObjectiveLog 中维护逐条计划变更，符合 ADR 0018"ObjectiveLog 通过 run_id 引用 Run、不复制 RunLog 内容"的既有规则。
- `agiwo.agent.plan` 拥有 `RunPlan` 更新规则与 `UpdatePlanTool`；纯数据模型收口在 `agiwo.agent.models`。`agiwo.agent.introspect` 拥有 `ReviewTrajectoryTool`、复盘触发、append-only 纠偏与实验性有用性评分，并只通过 RunPlan 的内部接口读取 active milestone；历史重建仍由正式 compaction 负责。两者都不依赖 Scheduler。
- Agent 在构建 Run runtime 时自行装配 `update_plan`，并在 `enable_trajectory_review=true` 时装配 `review_trajectory`；该配置默认保持 `true`。二者都是不受 `allowed_tools` 过滤的内部系统工具。Scheduler 只注入 spawn、fork、sleep、query、cancel、list 等调度控制工具，不再定义、构造或派生计划与复盘工具。
- root 与 child Run 都可使用 `update_plan`；是否启用 Assignment 完成门禁由显式 Assignment root 身份决定，不由工具是否存在推断。非 Objective 路径的普通 Run 也可维护 RunPlan，但默认保留无 tool call 即返回的既有结束语义。
- 计划的唯一管理入口是 `update_plan(changes=[...])`；它同时承担首次声明与后续修订，不再保留 `declare_milestones` 旧名。每个 change 以稳定 milestone id 为键：新 id 必须提供 description，已有 id 只修改本次明确提供的字段，未出现在 changes 中的计划项保持不变。Objective 在 Run 内没有 runtime update tool，其全局修订只发生在用户输入和 Assignment 收尾边界（见 ADR 0039）。
- 一次 `update_plan` 中的全部 changes 原子应用；任一 change 非法时整批拒绝，不产生部分计划或 `RunPlanUpdated`。成功时，模型可见的 tool result 只返回稳定格式的 plan revision 与状态计数，不重复计划描述；assistant tool call 保留本次增量参数，规范化后的完整当前计划写入内部 ToolResult output 与 `RunPlanUpdated` fact。失败结果必须保留可操作的字段级错误，帮助 Agent 修正调用。
- 计划项沿用 `Milestone` 模型：稳定 id、description 和 `pending / active / completed / abandoned` 状态。不提供物理删除，也不增加 `blocked`、`cancelled` 状态；信息缺失或计划变化通过修订计划本身或向用户请求输入表达。`abandoned` 表示该项经 root Agent 判断不再需要执行，视为已处理。
- `completed / abandoned` 在同一个 Run 内可以重新改为 `pending / active`，继续使用原 milestone id；跨 Assignment 的新 Run 不继承该 identity。修改描述直接更新原项目，不通过删除重建。
- `Milestone.status` 是当前 active 状态的唯一事实来源；`RunPlan.active_milestone_id` 只作为从唯一 `status=active` 项推导的便利属性，不单独持久化。`IntrospectionTriggered` 等历史事实仍可记录触发当时的 active id，因为它们表达历史锚点，不是当前状态副本。
- 规范化后的非空未解决计划必须恰有一个 active：一次 changes 明确声明多个 active 时整批拒绝；明确激活新项时，未被完成或放弃的旧 active 自动退回 pending；active 被完成或放弃且未明确指定下一项时，系统按稳定计划顺序激活第一个 pending；首次计划全为 pending 时同样自动激活第一项；全部项目均为 completed/abandoned 时不保留 active。
- 新 milestone 按同一次 changes 中的出现顺序追加，已有 milestone 保持原顺序。第一版不提供重排操作；Agent 通过显式激活目标项改变当前焦点，程序不判断项目的语义优先级。
- 每次计划变更以 `RunPlanUpdated` first-class fact 写入 RunLog；当前快照保留全部计划项和状态，历史 facts 保留完整审计轨迹。Objective 时间线需要计划演进过程时，经 Assignment/Run 节点下钻 RunLog 查看，不在 ObjectiveLog 中复制。Console 默认折叠 `completed / abandoned`，但不能从历史和 review cycle 中隐藏其 identity。

### 完成门禁

- 完成门禁是 `agiwo.agent` finalization 路径的通用可配置能力：root Run 产生无 tool call 的 assistant step 时，系统确定性检查当前 `RunPlan`；存在 `pending / active` 项则不进入收尾调用，按 ADR 0006 追加内部来源的固定 `role=user` 提醒并继续当前 agent loop。
- Objective 管理的 Assignment root Run 必然开启该门禁；非 Objective 路径的 Run 默认保持"无 tool call 即正常返回"的现有语义，可按需开启。门禁读取的是当前 Run 本地 ledger 中的 `RunPlan`，不查询 ObjectiveLog，不产生跨层读取。
- 计划是可选的执行管理能力：简单 Assignment 不要求先创建形式化清单；只有 Run 已声明计划时门禁才生效。门禁只校验 Agent 自己声明的计划，不替 verifier 判断 Objective 是否真的完成。
- 该门禁不能只依靠 prompt 要求模型自律；它是 finalization 路径的确定性检查。

### 交接与边界

- Assignment 边界即计划边界：`RunPlan` 与轨迹自省状态严格按当前 `run_id` 重放。暂停恢复同一个 Run 时恢复原计划；每个新 Assignment 使用新 Run，并从空 `RunPlan` 开始。后继 Assignment 不继承旧计划项的 identity；它读取 Outcome 中的 `carry_forward`，依据自己的责任决定是否声明新计划。
- 为保持 LLM 前缀缓存，已经进入 Session 历史的旧 Run `update_plan` assistant/tool 消息不因新 Assignment 而删除或改写。新 Assignment Input 在稳定历史尾部声明当前 `run_id` 的计划尚未建立、旧计划仅是历史；程序的完成门禁和轨迹自省始终只读取按当前 `run_id` 重放的权威 `RunPlan`。
- Assignment 以 `INTERRUPTED` 结束时，收口流程把 `RunPlan` 中 `pending / active` 项快照进唯一 AssignmentOutcome 的 `carry_forward`，由 Objective 层写入 ObjectiveLog；不能因中断静默丢弃计划。
- 用户 pause 或预算暂停不结束 Assignment；计划状态随 RunLog 事实自然保留，恢复同一 Run 后继续维护，无需额外机制。
- 若计划项因用户信息缺失或方向变化不再合理，root Agent 直接修订计划，并选择继续执行或在当前计划处理完后通过 Decision 请求用户输入；不得仅因缺少用户信息而 `target=agent`，否则多个 Agent 会在没有新事实的情况下循环 handoff。
- `RunPlan` 只覆盖当前 root 在下一次 Decision 前承担的责任。计划项可以由 child 执行，但必须由 root 接收、验证并负责；不得把后继 Agent、Verifier 或 User 才能完成的工作放入当前计划。当前 root 可以计划“准备验证材料”或“识别所缺信息”，不能计划“Verifier 验收通过”或“等待用户回复”。

### 实施注意点

- **门禁只对 Assignment 的 root Run 生效。** child Run 可以通过 `update_plan` 管理自己的 `RunPlan`（各自 ledger 独立，服务各自的轨迹自省），但完成门禁和 `carry_forward` 只读取 root Run 的 `RunPlan`；child 的计划随委派结果返回 root 后自然消亡。实现时不得把门禁装配到所有 Run 上。
- **计划粒度和责任范围必须在 planning policy 中显式约束。** milestone 同时充当自省参照物与完成清单，最佳粒度是当前责任内“可验证的阶段性结果”，不是操作步骤或未来接棒者的工作。共享 system prompt（ADR 0026）必须写明该要求，避免模型产出十几条细碎待办或把 handoff 后的责任塞入当前计划；自省继续只锚定 active 项。
- **一份数据、三个视图。** Console 的 milestone board、review cycles 与 Objective 时间线的遗留项都由 `RunPlanUpdated` facts（及 Outcome 快照）投影，不得出现第二份计划数据源。

### 命名

- `GoalState` 统一改名为 `RunPlan`，`RunLedger.goal` 改为 `RunLedger.plan`。
- `GoalUpdate` 与 `GoalUpdateReason` 分别改名为 `RunPlanUpdate` 与 `RunPlanUpdateReason`。
- `GoalMilestonesUpdated` 改名为 `RunPlanUpdated`，`declare_milestones` 改名为 `update_plan`；不保留旧名称的兼容别名。

本决定取代本 ADR 前一版本"Assignment Todo 由当前 root Agent 直接管理"中由 Objective runtime tool 管理独立 todos 的设计；前一版本中关于门禁语义、carry_forward、root 独占管理和不增加额外状态的不变量全部保留，仅数据归属与工具载体改变。ADR 0006、0012、0024、0031、0039、0041、0042 中的 todo 表述随本决定同步修订为计划表述。
