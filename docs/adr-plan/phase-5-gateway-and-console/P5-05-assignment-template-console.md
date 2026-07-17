# P5-05：实现 Assignment 模板管理界面

状态：planned

## 目标

在 Console 后台为默认 AgentConfig 提供三种 Assignment 输入模板的编辑、校验和预览。保存必须先持久化 AgentRegistryStore，再刷新内存；运行中或 PAUSED 的 Assignment 使用创建时钉住的快照，resume 仍用该快照，不受即时修改影响。

## 对应决定

- ADR 0026：模板可在 Console 配置和预览。
- ADR 0027：模板持久化到默认 AgentConfigRecord。

## 依赖

- P2-01 已完成。

## 实施步骤

1. 在默认 agent 设置或 Agent 表单中增加三个清楚分区的模板编辑器；不把字段藏进通用 options JSON。
2. 显示允许占位符清单和每种 kind 的必需项；不提供可执行模板语法。
3. 前端调用后端校验/preview API，使用固定示例 ObjectiveView 展示最终 UserMessage 文本和 provenance。
4. 保存时提交完整 TemplateSet 与 expected AgentConfig updated_at/hash，避免覆盖并发修改。
5. 后端按 P2-01 流程校验 -> Registry persistence -> memory refresh；任一失败返回明确阶段和当前有效版本。
6. 成功后显示新的 content hash/updated_at；不假装维护模板历史版本。
7. 明确提示：修改只影响之后新创建的 Assignment；运行中/PAUSED 的 Assignment 使用已钉住快照，resume 不改用最新 Registry（ADR 0033）。
8. default record 尚未持久化时显示 fallback 来源；首次保存创建同 ID record，重启后 Registry 优先。
9. system planning policy 与三种模板分开编辑，避免重复和漂移。
10. 编辑路径不得改写、删除或覆盖已有 Assignment 已钉住的快照字节；只能写入默认 record 供后续 Assignment 使用。

## 主要改动位置

- `console/server/routers/agents.py` 或 config router
- `console/server/models/agent_config.py`
- `console/server/services/agent_registry/`
- `console/web/src/components/agent-form.tsx`
- `console/web/src/app/settings/page.tsx`
- API client/types/tests

## 测试计划

- 四模板加载、编辑、校验、preview 和保存。
- 未知/缺失占位符的字段级错误。
- SQLite 重启后保留；memory backend 仅进程内。
- persistence 失败不刷新内存；refresh 失败能重新加载持久 record。
- 并发 expected hash 冲突。
- 修改后旧 Assignment input/hash 不变，新 Assignment 使用新模板；PAUSED Assignment resume 仍用旧快照成功。

## 完成标准

- 常用模板配置是可见表单，不要求用户编辑 JSON。
- 保存语义与持久化事实一致，不出现“界面成功、重启丢失”。
- planning policy 只有一份。
- 后端 Console tests 和前端 lint/test/build 通过。

## 风险与回退

不要让前端自行实现另一套模板 parser。预览和最终渲染必须调用同一后端 renderer，否则保存成功的模板可能在运行时得到不同文本。

