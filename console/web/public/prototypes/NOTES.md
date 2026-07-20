# Session / Trace 巡检页重设计 — 原型笔记

**问题：** 当前 Console 的 Sessions / Session detail / Traces 无法让人快速理解 agent 执行轨迹与关键状态。

**原型：** [`session-trace-redesign.html`](./session-trace-redesign.html)

**打开方式：**

```bash
open console/web/public/prototypes/session-trace-redesign.html
```

## 设计结论（已落地到 React）

1. **Session 详情主舞台：Agent Runs + Steps**（`runs-steps-panel.tsx`）
2. **Outcome 顶栏 + Assignment 标签**（`outcome-hero.tsx` / `assignment-tags.tsx`）
3. **Objective 过程摘要侧栏**（`process-summary.tsx`；完整 facts 折叠）
4. **Trace：可点时间条 + 右侧分层详情**（`trace-flame-explorer.tsx`）
5. **Sessions 列表以 Objective 状态为主信号**（并行拉取 objectives）

已用本地 `console/.agiwo` 数据验证，例如：

- Session `855e4bcb-dd3e-4b86-af58-dfed7b4126be`：COMPLETED + intake + 25 steps / 13 tools
- Trace `8c9ac2bf-5f3f-4630-93ec-ebb03afff372`：时间条点选 web_search → Tools 层展示 IO
