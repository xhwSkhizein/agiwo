# adr-plan Review：遗漏与风险

对照 `CONTEXT.md`、`docs/adr/0001`–`0044` 与 `docs/adr-plan` 的执行方案做缺陷审查。本文件只记录问题；确认后再改方案与 ADR。

审查基准日：2026-07-17。

## 使用方法

1. 下面每个问题按编号逐项确认。
2. 对每一项，用户给出：`确认 / 部分确认 / 驳回`，并可补充约束。
3. 确认后的项再改 `docs/adr-plan` 与相关 ADR；未确认项保持原状。

状态约定：`open` → `confirmed` / `partial` / `rejected` → `resolved`。

**本轮审查结论（2026-07-17）：** R1–R16 均已 confirmed/rejected 并落到文档；无残留 open 项。后续以 `docs/adr-plan` 任务实施为准。

---

## R1 — ObjectiveActionLease 是错误决定，应改为调用前成本上界检查

- **状态：** confirmed → resolved
- **严重度：** high（已按确认修复）
- **用户确认：** ObjectiveActionLease 废弃。最新决定是用「调用前检查」替代「调用前预留」：BEFORE_LLM 检查 `used + call_cost_ceiling <= limit`，超则 pause；不预留、不结算、不返还。最坏超支被一次调用成本上界封顶（并行时 × 分支数）。
- **已做修复：**
  - `CONTEXT.md`：以 Call Cost Ceiling / LLM Preflight Cost Check 取代 Objective Action Lease；DRAINING / 成本记账同步。
  - ADR 0010、0011、0014、0023、0028、0029、0044：删除 lease 准入，改为上界检查 + 状态门禁。
  - adr-plan：README、P3-02、P3-05、P3-06、P4-01、P4-04、P6-01 同步。
- **残余说明：** DRAINING 接受极窄 check-then-act（每并发分支至多一个已越过检查点的额外动作）；由 Run checkpoint/Outcome barrier 收敛，不再用租约集合。

## R2 — ObjectiveActionLease 持久化表无人认领

- **状态：** rejected（superseded by R1）
- **严重度：** medium
- **说明：** 租约概念整体废弃后，不再需要 action-lease 表；本项不再作为独立缺陷跟踪。

## R3 — Artifact 收窄为 Session 工作目录中的文件索引

- **状态：** confirmed → resolved
- **严重度：** high（已按确认修复）
- **用户确认：**
  1. Artifact 只管理文件型产出（图片、PDF、日志、大文本等）与超长输入外置；收尾/故障 report 一律普通文本。
  2. 文件落在 `{agent_workspace}/sessions/<session_id>/artifacts/`；对象含 path、可选小 content、summary；大文件按需读 path。
- **已做修复：**
  - 新增 ADR 0045；修订 0005/0006/0007/0011/0012/0015–0017/0020–0025/0023/0024/0030/0036/0041 等。
  - `CONTEXT.md` 与 adr-plan（P1-01/02、P2-04/05/06、P4、P5-01/04、P6-01、总 README）同步。
- **对 R4 的影响：** 外置后上下文提供 path + summary，agent 按 path 按需读取；R4 的实现验收已写入 P2-04 / P5-01 并关闭。

## R4 — 超长输入外置后，模型缺少可访问原文的路径

- **状态：** confirmed → resolved
- **严重度：** medium（契约早由 R3 关闭；实现验收已写入计划）
- **用户确认：** 保持 ADR 0045 契约；在 P2-04 / P5-01 完成标准与测试中明确按需读取验收后关闭本项。
- **已做修复：**
  - P2-04：外置后 path+summary + 既有读文件工具可打开并读回原文；禁止空引用。
  - P5-01：externalize 集成测试覆盖可达 path 与原文一致。
- **明确不做：** 在上下文中内联全文，或引入第二套专用「读 Artifact」领域工具（除非现有读文件能力无法覆盖 workspace path）。

## R5 — 破坏性改名不写兼容层，开发数据清理重建

- **状态：** confirmed → resolved
- **严重度：** high（已按确认修复）
- **用户确认：** 需要清理重建；不要在代码里搞兼容逻辑；MVP 不承载历史债务。
- **已做修复：**
  - 总方案原则改为：无 migration、无兼容读取、无旧 kind 探测；合并前清理本地数据。
  - P0 README 入口条件要求先清 `.agiwo` / Console SQLite 等开发状态。
  - P0-01、P6-03 同步：只保留当前契约，发布说明复述清理范围。
- **明确不做：** 不为旧 RunLog/枚举写 fail-closed 探测或友好迁移路径；旧数据直接不兼容，靠开发者清库。

## R6 — RunFinalizationSpec 不经 Scheduler 传递；root 启用 Agent 内置收尾

- **状态：** confirmed → resolved
- **严重度：** medium（已按确认修复）
- **用户确认：**
  1. 派发只传 identity；`assignment_role=root` 启用 Agent 内置收尾；不传 Spec。
  2. 四种 kind 共用一套收尾 schema；report 为收尾前普通文本。
  3. 有未完成 RunPlan 则只提醒、不收尾、不写 Outcome。
  4. `new_contributions` 与 `contribution_annotations` 分开；`objective_update` 有明确字段（见 ADR 0006）。
  5. 收尾指令为尾部 user message；请求必须保留与常规 turn 相同的 tools（前缀缓存优先）；禁止无 tools 收尾。
- **已做修复：** ADR 0006/0007/0037/0039、CONTEXT、P2-02、P2-05；review 本项关闭。顺带采纳 revision 不匹配时「Outcome 仍提交、update 拒绝并记 fact」（原 R14 建议），R14 可在后续确认时标 resolved。

## R7 — 协作式 pause/checkpoint 进入 Agent run loop 的信号不足

- **状态：** confirmed → resolved
- **严重度：** medium（已按确认修复；设计已推翻旧 steering 收口模型）
- **用户确认：**
  1. 运行中用户输入：写 ObjectiveUserInput，向同一 root Run 注入 `is_user_provided=false` 系统提示型消息；无 Outcome、无新 Assignment、无 DRAINING。
  2. 用户暂停/停止与预算触顶：统一为可恢复中断；DRAINING 后 Run → PAUSED；按消息末项恢复（assistant+tools / assistant / tool result）。
  3. FAILED 仅真正不可恢复故障；普通停止不得复用 cancel→CANCELLED/FAILED。
- **已做修复：**
  - 重写 ADR 0023 / 0028 / 0029 / 0033；同步 0012、0024–0027、0030–0032、0035–0036、0042。
  - 重写 P3-04 / P3-05 / P4-04 与 P4 README；总 README、P0-04、P1/P2/P5/P6 相关表述去掉 steering Assignment / steering_outcome。
  - run loop 内部信号收窄为 `pause` vs `force_fail`；不再需要 `finalize_for_steering`。
- **对 R9 的影响：** 运行中输入不再全局 drain，R9「合并进一次 steering Assignment」被 supersede。

## R8 — “前缀缓存优先”与“用户输入按原始顺序补回”可能冲突

- **状态：** rejected（confirmed invalid under current design）
- **严重度：** medium
- **用户确认：** 下一个 Assignment 的上下文应已完整，不存在「缺失用户输入再中段补回 / 重新拼装」的正常路径；R8 是旧 steering 防守文案与前缀规则叠成的假问题。
- **已做修复：**
  - ADR 0007 / 0026 / 0036：删除「缺失输入按原序补回」；改为按 `input_id` 去重复用稳定前缀，新内容只追加尾部；日志有而历史无 → 不变量故障。
  - P2-01 / P2-04、总 README 原则同步；运行中输入仍走同一 Run 注入（ADR 0023），保证进入历史后再谈跨 Assignment 复用。
- **残余说明：** 外置授权会把**已在历史中的那条**用户表示改为 path/summary，这是对该条的表示更新，不是跨 Assignment 中段插入。

## R9 — Steering 无合并窗口，渠道连发会反复全局 drain

- **状态：** rejected（superseded by R7）
- **严重度：** medium
- **说明：** 运行中输入改为直接注入同一 root Run，不再 DRAINING / synthesis / 新 Assignment；连发只追加多条注入。渠道侧既有 Feishu batch 仍可降低噪声。原「合并进一次 steering Assignment」不再适用。

## R10 — ObjectiveStore 只规划 memory/SQLite，与 ADR“集合/Mongo”表述及后端跟随规则未闭合

- **状态：** confirmed → resolved
- **严重度：** medium（已按确认修复）
- **用户确认：** MVP 明确只支持 memory/SQLite；其他 backend fail-closed；去掉暗示已支持集合/Mongo 的说法。
- **已做修复：**
  - ADR 0040 重写：跟随 = 同一 wiring/同一 sqlite 或 memory；非 memory/sqlite 构造 fail-closed。
  - ADR 0018 / 0019、CONTEXT、P1-03、AGENTS.md 同步：Scheduler state / RunLog / ObjectiveStore 均以源码为准，MVP 为 memory/sqlite；未实现 backend fail-closed。
- **明确不做：** 不为 Objective 提交半成品 Mongo 实现，也不在配置不可用时 silent 降级为 memory。

## R11 — 配置/模板热更与 checkpoint hash 校验未约定

- **状态：** confirmed → resolved
- **严重度：** medium（已按确认修复）
- **用户确认：** resume **钉住快照**——必须用 checkpoint/Assignment 创建时钉住的 AgentConfig 与模板；编辑只影响之后新建的 Assignment。
- **已做修复：**
  - ADR 0033：与 live Registry hash 不同是预期，不得因此拒绝；快照缺失/自检失败才保持 PAUSED。
  - P3-04 / P5-05：创建时持久化快照；编辑不得改写已钉住字节；交叉测试覆盖 PAUSED 改模板后仍可 resume。
- **明确不做：** 因后台改了默认配置就拒绝 resume，或静默改用新 prompt/schema 恢复。

## R12 — Scheduler facade 的查询/checkpoint 面无单一 owner

- **状态：** confirmed → resolved
- **严重度：** medium（已按确认修复）
- **用户确认：** 在 P2-02 一次性定义 facade 扩展契约；后续任务只实现/消费，不再各自发明 API。
- **已做修复：**
  - P2-02：冻结派发、run/树查询、recoverable pause/resume、inject、可选 SSE 订阅句柄等机械面；可 stub，不得平行分叉。
  - P2-03 / P3-04 / P3-05 / P3-06 / P4-04 改为引用该契约。
- **明确不做：** 后续任务再写「Scheduler facade 增加所需查询能力」式开放扩面。

## R13 — `user_archive` drain reason 落在状态机任务之外

- **状态：** confirmed → resolved
- **严重度：** low（已按确认修复）
- **用户确认：** 选 1——P3-05 起就枚举并测试 `user_archive`，与 budget/user_pause 一并拥有 barrier。
- **已做修复：**
  - P3-05：reason 封闭为 `budget` / `user_pause` / `user_archive`；`user_archive` → USER_PAUSED；三种 reason 均有 barrier 测试。
  - ADR 0028 / 0041、P5-06：P5 只消费枚举，不再扩展。
- **明确不做：** 把 `user_archive` 留到 P5 才加入枚举。

## R14 — finalization 的 `expected_revision` 与并发用户输入

- **状态：** confirmed → resolved（随 R6/ADR 0006、0037、0039 一并落地）
- **严重度：** low
- **决定：** `expected_revision` 不匹配时，Outcome 与 Decision 仍可提交；该次 `objective_update` 被拒绝并以 first-class fact 记录原因；不静默丢弃、也不因过期 update 阻断 Outcome。

## R15 — adr-plan 与 `docs/eval-draft` 未对齐范围

- **状态：** confirmed → resolved
- **严重度：** low（已按确认修复）
- **用户确认：** `docs/eval-draft` **不在**本次 Objective 重构范围；执行 adr-plan 时不考虑、不依赖、不跟进该目录内容。
- **已做修复：**
  - 总方案 README 增加明确排除说明。
  - P6-02 限定为 Objective/trajectory-review 观测与指标，不实现通用 benchmark eval 框架。
  - `docs/eval-draft/README.md` 顶部注明与本次重构无关。
- **明确不做：** 为 eval-draft 补依赖边、共享 P6-01 夹具，或在本计划中实现 evaluation core。

## R16 — max_steps 中断收尾的 Decision 与 carry_forward 注入在计划中偏弱

- **状态：** confirmed → resolved
- **严重度：** low（已按确认修复）
- **用户确认：** P2-05/P2-06 显式对齐 ADR 0012：机械默认 `target=agent`；`carry_forward` 必须进入下一 Assignment Input。
- **已做修复：**
  - P2-05：max_steps 收口强制 `target=agent`（覆盖模型改路由）；Outcome 含 carry_forward；测试覆盖。
  - P2-06：装配规则 + 双保险校验；E2E 断言下一 work Input 含全部 carry_forward 项。
- **明确不做：** 允许模型把阈值保险丝改成 verifier/user；只记录 carry_forward 却不注入下一 Assignment。

---

## 审查备注（非独立问题，供确认时参考）

- ADR 0044 对早期 Objective/Scheduler 所有权冲突的统一，在计划依赖与边界叙述上基本一致。
- Planning Policy 文本：P0-01 / P2-01 / P5-05 有装配与编辑入口；若确认需“默认政策正文”的单一作者任务，可并入 R 系列或单独追加。
- 主动时间无 heartbeat 的极端计时问题，总方案 §9 已显式排除，不记为缺陷。
