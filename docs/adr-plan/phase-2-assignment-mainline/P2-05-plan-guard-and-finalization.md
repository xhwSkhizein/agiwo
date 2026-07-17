# P2-05：实现计划门禁与系统收尾调用

状态：planned

## 目标

当 Assignment root Run 产生无 tool call 的 assistant 消息时，由系统先检查当前 RunPlan，再必然发起一次结构化收尾调用。模型不能选择是否提交 Outcome。非 Assignment Run 和 child Run 保持现有结束语义。

## 对应决定

- ADR 0005、0006：唯一 Decision、系统控制收尾、线格式与前缀安全收尾请求。
- ADR 0007、0037、0039：Contribution / annotations / objective_update 边界。
- ADR 0012：max_steps_per_run 的有限收口。
- ADR 0024：每个终态 Assignment 都有 Outcome。
- ADR 0038：root RunPlan 门禁。

## 依赖

- P0-01、P0-04、P2-02、P2-04 已完成。

## 关键边界

- Agent 不能导入 `agiwo.objective`。
- **不经 Scheduler 传递 FinalizationSpec。** `assignment_role=root` 即启用 Agent 包内置收尾协议（固定 user 指令 + JSON schema + 一次纠正）；Scheduler 只透传 identity。
- 三种 kind 共用同一套收尾 schema；职责差异只来自 Assignment Input。
- ObjectiveService 接收 `RunFinalizationResult` 线格式，校验后写入领域 Outcome/Decision/Contribution/objective_update。

## 实施步骤

1. 在 `agiwo.agent` 定义 `RunFinalizationResult` 线格式（字段以 ADR 0006 为准），以及包内常量：收尾 user 指令文本与 JSON schema。不定义跨 Scheduler 的 Spec DTO。
2. RunLoop：仅当 `assignment_role=root` 时启用计划门禁与收尾。child / direct Run 行为不变。
3. root 出现无 tool call 的 assistant 后，检查当前 ledger RunPlan：
   - 存在 pending/active：提交该 assistant step，追加 `is_user_provided=false`、`origin=assignment_plan_guard` 的计划提醒，继续同一 loop；**不收尾、不写 Outcome**。
   - 计划已空或从未声明：该 assistant 文本定为 report / `RunOutput.response`，进入收尾。
4. 收尾请求：
   - 在历史**尾部**追加一条 `role=user`、`is_user_provided=false`、`origin=assignment_finalization` 的指令消息（前缀缓存优先：不改写、不重排已提交前缀）。
   - Provider 请求必须携带与当前 Run 常规 turn **相同的 tools 列表**；禁止改成无 tools。
   - 指令要求本次只输出 ADR 0006 的 JSON、不要发起 tool call。
   - phase=`assignment_finalization`；计入 `max_steps_per_run` 与 Objective 成本检查。
5. 若模型返回 tool call 或 JSON 非法：记解析失败；允许一次 `finalization_correction`（同样保留 tools、尾部追加纠正 user 消息）。再次失败则生成最小机械 handoff Result，不猜自由文本。
6. 完整 messages、tools、output、reasoning、usage、解析错误写入 RunLog debug facts；不 append 进后续普通业务对话历史。
7. `RunOutput.response` 始终是收尾前的 report 原文；Result 不得覆盖它。
8. Result 作为类型化执行结果返回（非松散 metadata）。ObjectiveService：
   - 映射 `decision` → 领域 Decision；
   - `new_contributions` → 新建 Contribution（系统分配 id）；
   - `contribution_annotations` → 对已有 id 追加 annotation（填充 from/time）；
   - `objective_update` → 校验 expected_revision/sources/字段后写入或拒绝该 update；
   - `artifact_refs` → 文件 Artifact 引用（ADR 0045）。
9. pause / budget pause 不执行收尾。
10. **`max_steps_per_run` 阈值收口（ADR 0012，硬约束）：**
    - 达到阈值且仍有 pending/active 计划项（或阈值截断需接棒）时：Run/Assignment → `INTERRUPTED`；Outcome 必须含普通文本 report、剩余计划项状态，以及需后继继续的项标记为 `carry_forward`。
    - Decision **系统保证** `HandoffDecision(target=agent)`。若模型 JSON 给出其他 target（verifier/user）或省略 Decision，**覆盖为** `target=agent` 并记 first-class fact（reason=`max_steps_per_run_mechanical_handoff`）；不得让模型把机械保险丝改成验收或问用户。
    - 收尾/纠正解析两次失败时的最小机械 handoff 同样强制 `target=agent`。
    - 本路径仍消耗 ObjectiveBudget.handoffs（限额检查在 P3；本阶段先写计数预备 fact）。
11. `carry_forward` 项的序列化进入 Outcome，供 P2-06 装配下一 work Assignment Input；本任务用 fixture 锁住 Outcome 字段形状。

## 主要改动位置

- `agiwo/agent/models/` 的 finalization 线格式与常量
- `agiwo/agent/run_loop.py`
- `agiwo/agent/models/log.py`
- `agiwo/agent/runtime/state_writer.py`
- `agiwo/agent/termination/`
- `agiwo/objective/service.py`
- `agiwo/objective/finalization.py`
- `tests/agent/`、`tests/objective/` 相关用例

## 测试计划

- 未声明计划 / 已清空计划：直接收尾；有 pending/active：只 reminder，不收尾。
- guard/收尾只作用于 root；child/direct 无变化。
- 收尾请求 tools 与前一常规 turn 的 tools 列表字节级一致；历史前缀与收尾前完全相同，仅尾部多一条 user 指令。
- 去掉 tools 的收尾路径不得存在。
- report 不被 Result 覆盖；`new_contributions` 与 `contribution_annotations` 分开校验。
- `objective_update=null`、合法 update、revision 不匹配（Outcome 仍提交、update 被拒绝并记 fact）三条路径。
- JSON 成功、一次纠正成功、两次失败走最小 handoff。
- debug facts 可重放但不进入下一 Run 普通 messages。
- max_steps 触发：INTERRUPTED + Outcome 含 carry_forward；Decision 恒为 `target=agent`。
- 模型收尾 JSON 试图 `target=verifier` 或 `target=user` 时被覆盖为 `agent`，并有可查询 fact。
- 两次 finalization 解析失败 → 最小机械 `target=agent`，无自由文本猜测。

## 完成标准

- 模型没有「提交 Outcome」的 tool；收尾由系统必然触发。
- Scheduler 不解析收尾 JSON，也不携带 Spec。
- root 在未解决计划存在时不能 COMPLETED（reminder 路径）。
- max_steps 收口路径 Decision 不可被模型改成非 agent。
- 每次 finalization/correction attempt 有 phase、ordinal、usage；tools 列表非空且与常规 turn 一致（若该 Run 本身配置了 tools）。
- agent/objective 集成测试与 lint 通过。

## 风险与回退

模型在「仍带 tools」时可能违规发起 tool call——按解析失败 + 一次纠正处理，不要为此改回无 tools 请求。若内置 schema 与 Objective 领域校验漂移，用同一份 fixture 锁两边测试。
