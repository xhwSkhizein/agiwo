# Agent Layer — Revised Design (Post-Review)

> 基于 GPT review 和对当前代码的二次审读，修订原方案。  
> 核心修正：**优先解决真实耦合点，而不是为了压文件数去合并已有明确职责的 owner**。

---

## 0. 原方案哪里判断错了

### 错误 1：Runner + Executor 应该合并

**原判断**：调用深度 8 层太深，合并为 ExecutionEngine 可以减少 30% 代码。

**修正**：重新审读代码后，Runner 和 Executor 确实有不同的生命周期语义：

```
Runner.start_root()  [同步]
  → 创建 SessionRuntime、AgentRunContext、asyncio.Task、AgentExecutionHandle
  → 必须同步返回 handle 给调用者

Runner._execute_workflow()  [异步，在 task 内]
  → 创建 RunRecorder
  → 前置 hooks + memory retrieval
  → 创建 user step
  → 委托给 Executor
  → 后置 hooks + finalize

Executor.execute()  [异步]
  → prepare_execution → RunState
  → while loop → LLM → tools → compaction → termination
```

Runner 的 `start_root()` 是同步的 handle 工厂，不能塞进 async 的执行引擎。`_execute_workflow()` 管理的是 **run lifecycle**（recorder 创建、hooks 调度、异常处理），Executor 管理的是 **execution loop**（LLM 调用、工具执行、限制检查）。这两个职责确实不同。

合并它们不会消除生命周期语义，只会把两种关注点塞进同一个类。

### 错误 2：AgentCloneSpec 应该合并进 ChildAgentSpec

**原判断**：4 种 child 类型太多，合并到 2 种。

**修正**：`ChildAgentSpec` 和 `AgentCloneSpec` 语义完全不同：

```python
# ChildAgentSpec — "我想要什么样的 child"（输入规格）
@dataclass(frozen=True)
class ChildAgentSpec:
    agent_id: str
    agent_name: str
    instruction: str | None          # 覆写请求
    system_prompt_override: str | None
    exclude_tool_names: frozenset[str]
    metadata_overrides: dict

# AgentCloneSpec — "构造一个新 Agent 需要什么"（物化产物）
@dataclass(frozen=True, slots=True)
class AgentCloneSpec:
    agent_id: str
    config: AgentConfig               # 完整配置
    hooks: AgentHooks                  # 已解析的 hooks
    tools: tuple[RuntimeToolLike, ...] # 已解析的 tools
```

一个是 **输入规格**（"我要排除哪些工具"），一个是 **构造产物**（"这是最终的 config + hooks + tools"）。
合并会把"请求"和"结果"搅在同一个类型里，违反信息隐藏。保留两者。

### 错误 3：RunRecorder 应该拆出 EventBus

**原判断**：用 EventBus 解耦 side effects。

**修正**：当前 `commit_step()` 的执行顺序有**明确的行为契约**，被测试显式验证：

```python
# test_runtime_contracts.py:150
async def test_run_recorder_tracks_state_before_hooks_and_observers():
    ...
    assert seen == [
        ("hook", 1, 0, user_step.id),      # state 已更新，hook 看到 steps_count=1
        ("observer", 1, 0, user_step.id),   # observer 也看到 steps_count=1
        ("hook", 2, 1, assistant_step.id),   # state 已更新，message 已 append
        ("observer", 2, 1, assistant_step.id),
    ]
```

顺序是：**state.track_step → storage.save_step → trace → hook → observer → publish**。

这是 sequential pipeline，不是 pub-sub。如果拆成 EventBus，必须保持同步顺序执行，那它本质上就不是 "event bus"，而是一个 ordered sink coordinator。命名为 "EventBus" 会误导——暗示解耦和异步，但实际必须同步串行。

不建议引入新抽象。RunRecorder 的 `commit_step()` 虽然做了多件事，但它们的执行顺序是业务需求，不是偶然耦合。

---

## 1. 原方案哪里判断对了

### ✅ RunState 封装化

`executor.py` 中大量直接修改 state 属性：

```python
state.current_step += 1                              # L232
state.termination_reason = TerminationReason.CANCELLED # L121
state.termination_reason = reason                     # L198, L216, L257, L265
state.messages = result.compacted_messages            # L358
state.messages = modified                             # L236
```

任何持有 `state` 引用的代码都能随意改写。封装后可以：
- 限制变更入口
- 在变更时加验证/日志
- 避免 `state.current_step += 1` 被遗忘

### ✅ ChildDefinitionInputs 可以删除

`ChildDefinitionInputs` 是 `_build_child_inputs()` 的返回类型，被两个调用者立刻解构：

```python
# snapshot_child_definition() 解构后用 inputs.base_prompt, inputs.tools, inputs.hooks, inputs.options
# build_scheduler_child_clone() 同样解构
```

4 个字段的 frozen dataclass 作为内部函数的返回值，是过度建模。可以用 tuple 或直接在两个调用者中内联共享逻辑。

### ✅ RunRecorder.attach_state() 是 Runner/Executor 边界的真正痛点

当前流程：

```
Runner._execute_workflow():
  recorder = RunRecorder(context, hooks, observers)  # 无 state
  recorder.start_run(run)
  user_step = recorder.create_user_step()
  recorder.commit_step(user_step)                     # state 为 None，track_step 跳过
  ...
  executor = AgentExecutor(..., run_recorder=recorder)

Executor.execute():
  prepared = prepare_execution(..., run_recorder=recorder)
    → state = RunState(...)
    → new_recorder = recorder.attach_state(state)     # 创建新 RunRecorder 实例！
  # 从此用 new_recorder，原 recorder 废弃
```

问题：
1. `attach_state()` 创建了**全新的 RunRecorder 实例**，拷贝了 hooks/observers/run 引用
2. Runner 创建的原始 recorder 在 attach 后被丢弃
3. 两阶段 recorder（有 state vs 无 state）是隐式协议，不明显

---

## 2. 修订后的重构计划

### Phase 1：RunState 封装化

**目标**：消灭 executor.py 中对 RunState 属性的直接修改。

**变更范围**：`run_state.py` + `executor.py` + `termination_runtime.py`

```python
# run_state.py — BEFORE
@dataclass
class RunState:
    current_step: int = 0
    termination_reason: TerminationReason | None = None
    messages: list[dict]
    ...

# run_state.py — AFTER
@dataclass
class RunState:
    _current_step: int = 0
    _termination_reason: TerminationReason | None = None
    _messages: list[dict] = field(default_factory=list)
    ...

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def termination_reason(self) -> TerminationReason | None:
        return self._termination_reason

    @property
    def messages(self) -> list[dict]:
        """Read-only access. Use set_messages() or append_message() to modify."""
        return self._messages

    def advance_step(self) -> None:
        self._current_step += 1

    def terminate(self, reason: TerminationReason) -> None:
        self._termination_reason = reason

    def set_messages(self, messages: list[dict]) -> None:
        self._messages = messages

    def apply_compaction(self, compacted_messages: list[dict]) -> None:
        self._messages = compacted_messages
```

**executor.py 调用者更新**：

```python
# BEFORE                              # AFTER
state.current_step += 1               state.advance_step()
state.termination_reason = reason      state.terminate(reason)
state.messages = modified              state.set_messages(modified)
state.messages = result.compacted_messages  state.apply_compaction(result.compacted_messages)
```

**设计决策**：
- `messages` property 返回 mutable list 引用（允许 `MessageAssembler` 和 `apply_steering_messages` 原地操作）。如果要彻底不可变，改动范围太大且性能有问题。当前先用方法守住"结构性变更"（替换整个列表），允许"内容追加"（append）通过 `track_step` 走。
- `track_step()` 保持不变——它已经是封装好的入口。

### Phase 2：删除 ChildDefinitionInputs

**目标**：消灭纯中转类型，减少一层概念负担。

**变更范围**：`definition_runtime.py`

**方案**：将 `_build_child_inputs` 返回的 frozen dataclass 改为内部 helper，两个调用者直接消费返回的 tuple：

```python
# BEFORE
@dataclass(frozen=True, slots=True)
class ChildDefinitionInputs:
    options: AgentOptions
    tools: tuple[RuntimeToolLike, ...]
    base_prompt: str
    hooks: AgentHooks

def _build_child_inputs(self, spec: ChildAgentSpec) -> ChildDefinitionInputs:
    ...
    return ChildDefinitionInputs(options=options, tools=tools, base_prompt=base_prompt, hooks=hooks)

# AFTER — 删除 ChildDefinitionInputs 类，用 tuple 返回
def _resolve_child_overrides(
    self, spec: ChildAgentSpec
) -> tuple[AgentOptions, tuple[RuntimeToolLike, ...], str, AgentHooks]:
    options = self._config.options.model_copy(deep=True)
    options.enable_termination_summary = True
    tools = tuple(
        tool for tool in self._effective_tools()
        if tool.get_name() not in spec.exclude_tool_names
    )
    base_prompt = self._append_instruction(
        spec.system_prompt_override or self._config.system_prompt,
        spec.instruction,
    )
    hooks = copy.deepcopy(self._hooks)
    return options, tools, base_prompt, hooks
```

调用者解构：

```python
async def snapshot_child_definition(self, *, model, spec):
    options, tools, base_prompt, hooks = self._resolve_child_overrides(spec)
    prompt_runtime = self._create_prompt_runtime(
        base_prompt=base_prompt, agent_id=spec.agent_id, ...
    )
    return ResolvedExecutionDefinition(
        ..., options=options, tools=tools, hooks=hooks,
        system_prompt=await prompt_runtime.get_system_prompt(),
    )

def build_scheduler_child_clone(self, *, child_id, ...):
    spec = ChildAgentSpec(agent_id=child_id, ...)
    options, tools, base_prompt, hooks = self._resolve_child_overrides(spec)
    child_config = copy.deepcopy(self._config)
    child_config.system_prompt = base_prompt
    child_config.options = options
    return AgentCloneSpec(agent_id=child_id, config=child_config, hooks=hooks, tools=tools)
```

**净效果**：删除 `ChildDefinitionInputs` 类（4 行定义 + 导入），调用者更直接。

### Phase 3：RunRecorder.attach_state() 改为原地绑定

**目标**：消灭 "创建新 RunRecorder 实例" 的隐式协议。

**变更范围**：`run_recorder.py` + `execution_bootstrap.py`

**现状问题**：

```python
# run_recorder.py
def attach_state(self, state: RunState) -> "RunRecorder":
    recorder = RunRecorder(          # 创建全新实例
        context=self.context,
        hooks=self.hooks,
        step_observers=self._step_observers,
        state=state,
    )
    recorder._run = self._run        # 手动拷贝 _run 引用
    return recorder
```

这个方法创建了新实例来避免突变原对象。但审读代码发现：
- AgentExecutor 是 `_execute_workflow()` 中的局部变量，每次 workflow 单独创建
- RunRecorder 也是 `_execute_workflow()` 中的局部变量
- `attach_state` 后，原 recorder 再也不被使用

因此 "防御性新建" 没有实际价值。改为原地绑定：

```python
# run_recorder.py — AFTER
def attach_state(self, state: RunState) -> None:
    """Bind execution state to this recorder. Must be called exactly once."""
    if self._state is not None:
        raise RuntimeError("state_already_attached")
    self._state = state
```

```python
# execution_bootstrap.py — AFTER
async def prepare_execution(..., run_recorder: RunRecorder, ...) -> PreparedExecution:
    ...
    state = RunState(...)
    run_recorder.attach_state(state)       # 原地绑定，不再返回新 recorder
    return PreparedExecution(
        state=state,
        run_recorder=run_recorder,         # 同一个实例
        compactor=...,
    )
```

**净效果**：
- 消灭了 `RunRecorder.__init__` 中的 `state: RunState | None = None` 可选参数
- 消灭了隐式的"两阶段 recorder"协议
- Runner 和 Executor 共享同一个 RunRecorder 实例，生命周期更清晰
- `commit_step` 中 `if self._state is not None:` 检查保持不变（user step 在 state 绑定前提交时 track_step 跳过，这是正确行为）

### Phase 4（可选）：评估是否继续

前 3 个 phase 完成后，重新审视：
- Runner/Executor 边界是否还觉得别扭？如果 `attach_state` 痛点消除后不再别扭，**不合并**。
- RunRecorder 是否还是 God Object？如果 `commit_step` 的顺序执行管线是业务需求，**不拆**。
- `scheduler_port.py` 是否值得动？124 行稳定 adapter，**暂不动**。
- `inner/` 是否需要改名/拆目录？如果痛点已解决，**不做结构变更**。

---

## 3. 对 GPT Review 的逐点回应

### GPT 说对了的

| Point | GPT 判断 | 我的修正 |
|---|---|---|
| 不合并 Runner + Executor | ✅ 正确 | 撤回合并方案。`start_root()` 的同步 handle 工厂语义不能塞进异步执行引擎 |
| 不合并 AgentCloneSpec 进 ChildAgentSpec | ✅ 正确 | 撤回合并方案。输入规格 ≠ 构造产物 |
| 不做松散 pub-sub | ✅ 正确 | 撤回 EventBus 方案。commit_step 的顺序是行为契约 |
| scheduler_port.py 优先级低 | ✅ 正确 | 移出重构范围 |
| RunState 封装最值得动 | ✅ 正确 | 作为 Phase 1 |
| ChildDefinitionInputs 是纯中转 | ✅ 正确 | 作为 Phase 2 |
| attach_state() 是真正的痛点 | ✅ 正确 | 作为 Phase 3 |

### GPT 判断中我有补充的

| Point | GPT 判断 | 我的补充 |
|---|---|---|
| "先把文档重写成 2026-03 代码版本" | 合理 | 本文就是重写版。01-03 是历史快照，04 是修订版 |
| "第三阶段再评估 RunRecorder 内部 sink" | 偏保守 | Phase 3 改 `attach_state()` 是最小可行改动，不需要拆 sink |

### 原方案中真正值得保留的洞察

1. **DefinitionRuntime 确实偏重**（314 LoC）——但不是现在的瓶颈。工具组装/hooks 构建逻辑已经被提取为模块级函数 (`_build_effective_hooks`, `_build_sdk_tools`, `_build_prompt_runtime`)，DefinitionRuntime 本身只是把它们串起来。如果未来膨胀再拆。

2. **Storage 层的样板代码**——SQLite 515 LoC / MongoDB 420 LoC 确实冗长，但不影响执行层的设计。正交问题，独立解决。

3. **runtime/ 包的数据模型设计好**——保持不动。

---

## 4. 修订后的 Summary Matrix

| Phase | Change | Files | Risk | LoC Impact |
|---|---|---|---|---|
| **1** | RunState 封装化 | run_state.py, executor.py, termination_runtime.py, compaction/runtime.py | Low | +30 / -0 (style) |
| **2** | 删除 ChildDefinitionInputs | definition_runtime.py | Low | -8 |
| **3** | attach_state() 原地绑定 | run_recorder.py, execution_bootstrap.py | Low | -10 |
| ~~4~~ | ~~Runner+Executor 合并~~ | — | — | ~~撤回~~ |
| ~~5~~ | ~~EventBus 拆分~~ | — | — | ~~撤回~~ |
| ~~6~~ | ~~inner/ 目录重构~~ | — | — | ~~撤回~~ |

**总体变更量**：~6 个文件，+30/-18 LoC。这不是一个"大重构"，而是**3 个精确的耦合点修复**。

---

## 5. 与原方案的对比

| Dimension | 原方案 (02) | 修订方案 (04) |
|---|---|---|
| 文件数变化 | 63 → 45 (-28%) | 63 → 63 (不变) |
| LoC 变化 | ~5800 → ~4200 (-28%) | ~5800 → ~5800 (微调) |
| 新增抽象 | ExecutionEngine, StepEventBus, recording/ | 无 |
| 删除抽象 | Runner, Executor, ChildDefinitionInputs, AgentCloneSpec | ChildDefinitionInputs |
| 核心理念 | 压扁层次 + 解耦 side effects | 封装裸状态 + 消灭中转类型 + 修复边界 |
| 风险等级 | Medium（大面积文件移动和重命名） | Low（精确修改 6 个文件） |

一句话：**原方案想通过结构变更来解决耦合问题，修订方案在保持现有结构不变的前提下直接修复耦合点**。
