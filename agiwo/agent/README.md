# Agent Runtime

`agiwo.agent` 的 public surface 保持在包根；内部实现已经收口为两层：

- `lifecycle/`
  - `definition.py`: `AgentDefinitionRuntime`、`ResolvedExecutionDefinition`、`AgentCloneSpec`
  - `resource_owner.py`: run/session/trace 资源与 active root executions
  - `session.py`: `AgentSessionRuntime`
  - `orchestrator.py`: `ExecutionOrchestrator`
  - `assembly.py`: runtime owner 组装入口
- `engine/`
  - `context.py`: `AgentRunContext`
  - `state.py`: `RunState`
  - `recorder.py`: `RunRecorder`
  - `engine.py`: `ExecutionEngine`
  - `llm_handler.py` / `tool_runtime.py` / `termination.py` / `message_assembler.py`
  - `compaction/`: compact 相关实现

## Runtime Ownership

- `Agent` 是 facade，只负责构造 owner 和暴露 `start/run/run_stream`
- `AgentDefinitionRuntime` 负责 root/child/scheduler child definition materialization
- `AgentResourceOwner` 负责 run-step storage、session storage、trace storage、active root execution lifecycle
- `ExecutionOrchestrator` 只负责 root/child session wiring、task、handle、active execution 注册与关闭
- `ExecutionEngine` 负责单次 run 的完整 pipeline：prepare -> before_run -> loop -> after_run -> finalize
- `RunRecorder` 是唯一的 run/step lifecycle write owner，负责 storage、trace、hook、observer、stream fanout

## Key Contracts

- `Agent.start(...)` 同步返回 `AgentExecutionHandle`
- `Agent.run(...)` 只支持 root run；嵌套 child 执行只能走 `Agent.run_child(...)`
- `ChildAgentSpec` 是 public override input；`AgentCloneSpec` 是 scheduler child materialized output，不合并
- `RunRecorder` 不再允许双阶段 state 绑定
- `RunState` 的结构性字段更新必须收口到 `agiwo/agent/engine/` 内部的方法调用
- `scheduler/` 与 `console/server/` 不得依赖 `agiwo.agent.engine.*` / `agiwo.agent.lifecycle.*`

## Test Focus

当前 agent 层测试重点覆盖：

- run contracts：handle wait/stream/steer/cancel
- child/session contracts：sequence、parent/depth、shared session runtime
- recorder contracts：state/hook/observer/publish 顺序
- engine limits：max steps/timeout/max input/max output/max run cost/tool termination
- definition contracts：child materialization 与 scheduler child clone parity

## Maintenance Notes

- `agiwo/agent/inner/` 已删除，不要恢复兼容层
- 新增 agent 内部实现时，优先评估应该放 `lifecycle/` 还是 `engine/`
- 如果需要从包外访问 agent 行为，优先走 `Agent`、`AgentExecutionHandle`、`scheduler_port.py` 这些 public surface
