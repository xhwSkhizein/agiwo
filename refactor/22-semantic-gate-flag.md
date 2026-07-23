# Task 22：处理 `enable_semantic_completion_gates` no-op 开关

Phase 2 · 独立任务 · 修复 review 问题 #8（配置项看似生效实则无效）

## 现状与问题

调用链：

1. `AgentOptions.enable_semantic_completion_gates`
   （`agiwo/agent/models/config.py:86`，默认 `False`）；
2. `run_loop.py:362-371` `_evaluate_completion_gates` 每次现场构造
   `CompletionGates(enable_semantic=...)`，**不传 `semantic=` 参数**；
3. `CompletionGates.__init__`（`completion_gates/__init__.py:20-27`）
   兜底为 `DisabledSemanticCompletionGate`；
4. `DisabledSemanticCompletionGate.evaluate` 恒返回 `None` →
   `CompletionGates.evaluate` 恒 `AllowComplete()`。

结论：**开关置 `True` 后行为与 `False` 完全一致**。配置项已经出现在
公开的 `AgentOptions` 上，用户开启后会误以为语义门在工作。

## 决策

**推荐：移除配置项，保留 seam。**（G-v1a 的忠实执行）

理由：

- ADR 0049 G-v1a 明确"ship mechanical gates only; keep the semantic-gate
  seam but do not turn it on in v1"——seam 指 `SemanticCompletionGate`
  Protocol + `CompletionGates` 的组合点，**不要求暴露用户可见开关**；
- 无效配置项是最差的一种 API：它撒谎。fail-closed 原则下，
  没实现的能力就不该可配置；
- 语义门真正落地时（需要 LLM caller、SessionIntent 读取、复杂度信号），
  开关的形态大概率不是一个 bool（可能是触发条件配置），现在保留
  bool 反而预设了错误形状。

备选（不推荐，如团队决定提前打通注入链路时用）：
在 `CompletionGateContext` 增加 `semantic_gate: SemanticCompletionGate | None`
字段，由 MainAgent 构造时注入，`run_loop` 透传给 `CompletionGates`。
该方案的问题是引入了一个当前无人实现的注入点，属于 speculative
generality；记录于此备查。

## 任务拆分

1. **删除配置项**：`AgentOptions.enable_semantic_completion_gates`
   （`models/config.py:86`）。
2. **简化 run_loop**：`_evaluate_completion_gates`（`run_loop.py:362-371`）
   改为 `CompletionGates()`（机械门 only）；`enable_semantic` 形参与
   `semantic_enabled` 属性保留在 `CompletionGates` 上（它们是 seam 的
   一部分，供未来 MainAgent 注入时使用，且有 E 系列测试覆盖）。
3. **全库清扫**：
   ```bash
   grep -rn "enable_semantic_completion_gates" agiwo console tests docs
   ```
   逐一处理：Console 的 `agent_config` 模型 / registry 如果透传了该字段
   （核对 `console/server/models/agent_config.py` 本次 diff 新增的字段），
   一并删除；相关测试断言更新。
4. **配置热更核对**：`docs/config-hot-reload.md` 与 registry 的
   config snapshot 逻辑若列举了该字段，同步移除。
5. **文档同步**：`CONTEXT.md` / `AGENTS.md` 若提及语义门开关，改为
   "语义门 seam 保留（`SemanticCompletionGate` Protocol），v1 无
   用户可见开关"。

## 验收标准

- `grep -rn "enable_semantic_completion_gates"` 全库零命中；
- `tests/agent/test_completion_gates_e01/e02.py` 全绿
  （机械门行为不变；若有用例直接构造
  `CompletionGates(enable_semantic=True, semantic=stub)` 属于 seam
  测试，保留）。

## 风险

- 若 Console 前端（`console/web/`）已渲染该配置项，需同步删除表单字段；
  执行时 `grep -rn "semantic" console/web/src` 核对。
- 本任务是纯删除，无行为变化（因为开关本来就是 no-op），风险极低。
