# P5-03：接入 Session、Web 与渠道入口

状态：done

## 目标

让现有 Session 输入和 Feishu 等渠道消息统一进入 ObjectiveService。Session 仍是用户可见的长期对话容器；没有活动 Objective 时创建新 Objective，有活动 Objective 时把输入交给该 Objective 的状态机。

## 对应决定

- ADR 0021：Console/渠道复用同一 Objective Gateway use case。
- ADR 0035：一个 Session 同时最多一个非终态 Objective。
- ADR 0036：复用 Session persistent agent identity。
- ADR 0042：外部用户输入来源固定为 true。

## 依赖

- P5-01、P5-02 已完成。

## 当前源码现状

- `POST /sessions/{id}/input` 直接通过 `SessionRuntimeService`/Scheduler 执行并返回 Agent stream。
- Feishu inbound handler 解析消息后也进入 Scheduler 路径。
- Session.id 已作为 root persistent scheduler state id，Agent factory 使用稳定 config.id。

## 实施步骤

1. 建立一个 Console application adapter：输入 session_id/UserMessage，查询该 Session 的活动 Objective，并调用 ObjectiveService create 或 submit_input。
2. Session 没有活动 Objective时创建新 Objective；存在 CREATED/RUNNING/DRAINING/WAITING/BUDGET_PAUSED/USER_PAUSED 时按命令和输入类型交给当前 Objective。
3. COMPLETED/FAILED Objective 不重开；下一条普通用户 query 创建新 Objective，并可按相关性引用旧正式交付/Artifact。
4. 保留 `POST /sessions/{id}/input` 兼容入口，但响应包含 objective_id，并使用 Objective SSE/当前 view 表达进度；内部不直接 route_root_input。
5. Web chat 可以对很快完成的 Objective 自动等待一小段时间，但这只是客户端行为，不建立同步后端路径。
6. Feishu inbound 继续负责批处理、消息解析和 current Session 指针；解析后的 UserMessage 统一调用同一 adapter。
7. 渠道文本、图片、文件和 ChannelContext 全部保留，`is_user_provided=true` 由可信 server boundary 设置；客户端 payload 不允许改为 false。
8. root agent_id/session state 继续稳定复用；每个 Objective/Assignment 使用新 IDs，不重建 Session identity。
9. 现有 `Scheduler.route_root_input()` 保留 SDK/debug 能力；正常 Web/渠道 code path 加架构测试禁止直接调用。
10. 更新 delivery：普通运行进度来自 Objective events，最终回复来自 ObjectiveDelivered Artifact；不发送未验收 candidate report。
11. 并发两个 Session inputs 依赖 ObjectiveService 事务，只能创建一个活动 Objective，另一个成为同 Objective 的运行中注入/input 或得到确定性冲突。

## 主要改动位置

- `console/server/services/runtime/session_runtime_service.py`
- 新的 Console Objective adapter（放 `services/` 具名模块）
- `console/server/routers/sessions.py`
- `console/server/channels/feishu/inbound_handler.py`
- Feishu delivery/command helpers
- `console/web/src/hooks/use-chat-stream.ts` 或现有 chat hook
- Console channel/session tests

## 测试计划

- no active -> create；active running -> 注入同一 root Run；waiting user -> response Assignment；terminal -> new Objective。
- Web 与 Feishu 对同一输入产生相同 Objective facts。
- multimodal/ChannelContext/provenance 全链路。
- concurrent inputs 不产生两个活动 Objectives。
- grep/architecture test 证明正常用户路径不直接调用 Scheduler。
- debug scheduler route 保持可用。

## 完成标准

- Session 是唯一用户顶层导航对象，Objective 是其中的工作项。
- Web/渠道共享相同状态和预算语义。
- 外部输入不能伪造系统来源。
- 终态 Objective 永不重开。
- Console backend、channel 和前端 chat tests 通过。

## 风险与回退

切流应在单个提交或明确 feature gate 中完成，避免同一普通入口同时写 Scheduler direct Run 与 Objective。回退时恢复旧 adapter 路由，但不能让一次请求双写两个生命周期。

