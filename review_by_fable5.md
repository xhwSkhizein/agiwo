# Agiwo 代码库可读性 Review（聚焦 [agiwo/]）

先给你一张地形图，再逐个指出复杂度的洼地。整个 SDK 约 **43,000 行**，重量分布是：

- agent/ 13.6k、objective/ 9.1k、tool/ 6.3k、[scheduler/](scheduler:0:0-0:0) 6.2k。架构骨架是三层事件溯源——agent 层的 RunLog、objective 层的 ObjectiveLog、scheduler 层的 AgentState——**主要的认知负荷不来自单个坏函数，而来自每一层都在手工重复同一套"事实流"仪式**：手工分配序列号、手工复制身份字段、手工写完再手工投影、手工实现幂等回执。下面按投入产出比排序。

---

## 一、[ObjectiveService](objective/service.py:161:0-1696:9)：七次复刻的幂等仪式（最高收益）

`@agiwo/objective/service.py:1-1666` 有 1666 行，是全库最大文件。每个命令方法（[create_objective](objective/service.py:290:4-449:21)、[submit_user_input](objective/service.py:451:4-573:21)、[externalize_user_input](objective/service.py:618:4-760:21)、[pause](objective/service.py:762:4-857:21)、[resume](objective/service.py:859:4-944:21)、[adjust_budget](objective/service.py:1026:4-1124:21)）都手工复刻同一段五步仪式：

1. [get_view](objective/service.py:264:4-268:66) + 存在性/terminal 校验
2. 拼 `request_payload` + [_canonical_hash](objective/service.py:152:0-154:58)
3. `get_receipt` → 撞 hash 抛 `IdempotencyConflict`，命中则 replay
4. 构造 facts（手工递增 seq）
5. 拼 `CommandReceipt` + `commit_command`

`IdempotencyConflict` 这段 15 行的三元判断出现了 **7 次**，`CommandReceipt(...)` 构造出现 8 次。这不只是行数问题：**幂等契约散落在每个方法里，新增命令时漏掉任何一步都会静默破坏幂等语义**。

**建议**：提取一个模板方法，把仪式收口成一处：

```python
async def _run_command(self, *, scope, idempotency_key, request_payload,
                       handler) -> CommandResult:
    # receipt 查询 / hash 冲突 / replay / commit 全在这里
    ...
```

各命令退化为纯业务函数，只返回 `(facts, result, slot_mutation, outbox_records)`。预计 [service.py](objective/service.py) 直接瘦身 400+ 行，且幂等语义变成"改一处、处处生效"。

---

## 二、手工序列号簿记：一条随时会踩的暗雷

[service.py](objective/service.py) 里 facts 的 sequence 是人肉维护的，`seq += 1`、`seq += len(close)` 与 `facts.append(...)` 交错出现，典型如：

```@agiwo/objective/service.py:1490-1493
            close = facts_close_active_window(view, start_sequence=seq + 1, now=now)
            facts.extend(close)
            seq += len(close)
            seq += 1
```

而 `@agiwo/objective/projection.py:126-167` 的投影器会严格校验序列——**写方任何一次 off-by-one 都会变成运行时的 `ProjectionError`**。同时 `@agiwo/objective/log.py:104-120` 的 28 个 `fact_*` 构造器每个都要显式收 `objective_id / sequence / fact_id / occurred_at` 四件套。

**建议**：引入一个小的 `FactBatch` builder——持有 `objective_id`、起始 seq、`now`，[batch.add(...)](agent/hooks.py:134:4-136:47) 自动分配 sequence 和 fact_id。所有 `fact_*` 构造器不再接收 sequence，签名减半；[service.py](objective/service.py) 里所有 `seq +=` 消失，一类 bug 从类型上灭绝。

---

## 三、[RunStateWriter](agent/runtime/state_writer.py:49:0-538:9)：950 行的参数搬运走廊

`@agiwo/agent/runtime/state_writer.py:50-539` 有 **23 个 `record_*` 方法**，每个都是同一模式：

- `await allocate_sequence()`
- 调对应的 `build_*_entry(...)`，把 `session_id / run_id / agent_id` 从 state 抄进 entry
- 再把 caller 的 10+ 个参数原封不动穿透过去（[record_llm_call_completed](agent/runtime/state_writer.py:278:4-311:9) 收 12 个参数、抄 12 个参数）
- [append_entries([entry])](agent/runtime/state_writer.py:55:4-58:28)

下半个文件又是一排 `build_*_entry` 函数做同样的字段搬运。**这是纯粹的参数穿透样板，读者要在三层签名里核对同一批字段**。

**建议**：让 writer 只干两件事——补全身份 + 分配序列：

```python
async def emit(self, entry_cls, **fields):
    entry = entry_cls(
        sequence=await self._allocate(),
        session_id=..., run_id=..., agent_id=...,
        **fields,
    )
    return await self.append_entries([entry])
```

调用方直接写 `await writer.emit(LLMCallStarted, logical_call_id=..., ...)`。23 个 `record_*` 和全部 `build_*_entry` 大半可删，预计 950 行 → 300 行以内。真正需要副作用的少数几个（[commit_step](agent/runtime/state_writer.py:348:4-357:70)、[record_termination_decided](agent/runtime/state_writer.py:359:4-377:9) 等要同步改 ledger）保留为显式方法。

---

## 四、"写完必须手工投影"的成对咒语

`@agiwo/agent/run_loop.py:74-81` 定义了 [_project_entries](agent/run_loop.py:73:4-80:9)，随后**在同一文件里被成对调用了 12 次**——每次 `await self.writer.xxx(...)` 后必须紧跟 [await self._project_entries(entries)](agent/run_loop.py:73:4-80:9)。忘记一处，stream/trace 就静默丢事件；`@agiwo/agent/llm_caller.py:51` 还为此把 `ProjectEntries` 回调穿透了四层函数签名。这是教科书式的时间耦合（temporal coupling）。

**建议**：把投影回调注入 [RunStateWriter](agent/runtime/state_writer.py:49:0-538:9) 构造函数，[append_entries](agent/runtime/state_writer.py:55:4-58:28) 落盘后自动投影。[run_loop.py](agent/run_loop.py) 少 12 段成对代码，[llm_caller.py](agent/llm_caller.py) 的 `project_entries` 参数彻底消失，"写了不投"从此不可能发生。

---

## 五、`run_loop` 的双层异常控制流

`@agiwo/agent/run_loop.py:135-155` 中 `RunPausedExit` 和 `RunBlockingFaultError` 在内层 try 和外层 try **各捕获一次**——因为暂停/故障是靠异常从 [_maybe_pause](agent/run_loop.py:343:4-347:43)、[_commit_pause_and_exit](agent/run_loop.py:349:4-361:71) 一路穿透多层抛上来的，而 [_finalize_run](agent/run_loop.py:261:4-290:21) 阶段也可能再抛。读者必须推演两层 catch 的先后关系才能确认行为。

**建议**：让 [_run_loop](agent/run_loop.py:292:4-341:17) 返回显式结果（如 `LoopExit.COMPLETED / PAUSED(checkpoint) / FAULT(fault)`），[execute_run](agent/run_loop.py:713:0-780:5) 单点分派。异常只留给真正的意外错误，控制流回到返回值上，双层 catch 消失。

---

## 六、Scheduler：命名错位与故事碎片化

三个具体问题：

- **文件名与职责错位**。[Scheduler](scheduler/engine.py:65:0-911:23) facade 类住在 `@agiwo/scheduler/engine.py:66-67`，而真正的编排循环在 [_tick.py](scheduler/_tick.py)。你的 `AGENTS.md` 写的是"`scheduler.py` 是 facade，[engine.py](scheduler/engine.py) 是唯一编排 owner"——**文档与现实已经漂移**，两边至少要改一边。
- **[_stream.py](scheduler/_stream.py) 与 [stream.py](scheduler/stream.py) 并存**（2.2k vs 2.3k 字节），名字只差一个下划线，读者每次都要打开确认哪个是哪个。
- **一次 dispatch 的完整故事横跨 8+ 个文件**：`engine.py → _tick.py → runner.py → runner_completion.py / runner_output.py / runner_events.py → runtime_state.py / runtime_facts.py`。[runner_output.py](scheduler/runner_output.py)（60 行）和 [runner_events.py](scheduler/runner_events.py)（70 行）这种碎片建议合回 [runner.py](scheduler/runner.py)——切分的收益低于跳转的成本。

另有一个局部坏味道：`@agiwo/scheduler/engine.py:789-806` 的 [prepare_resume](agent/agent.py:545:4-563:33) 里，为找一个 agent 做了**四段式 fallback**（`agents` 按 key 查 → `canonical_agents` 按 key 查 → 线性扫 `agents` 比对 [.id](agent/agent.py:172:4-174:23) → 线性扫 `canonical_agents`）。这暴露了 `RuntimeState` 双注册表的身份规则太隐晦。建议在 `RuntimeState` 上收口一个 `find_agent(agent_id)`，把查找规则写成一处有名字的方法。

---

## 七、[hooks.py](agent/hooks.py)：一个 phase 的合同拆在五张表里

`@agiwo/agent/hooks.py:75-111` 用五个类级集合/字典（`_TRANSFORM_PHASES`、`_DECISION_SUPPORT_PHASES`、`_CRITICAL_PHASES`、`_TRANSFORM_ALLOWLISTS`、`_DECISION_SUPPORT_ALLOWLISTS`）描述 hook 能力。想知道"`BEFORE_LLM` 允许什么"，要横跨五张表拼答案。

**建议**：按 phase 倒转成单张表：

```python
PHASE_SPECS: dict[HookPhase, PhaseSpec]  # PhaseSpec: 能力集合 + 各能力的字段白名单 + 是否允许 critical
```

一行看全一个 phase 的完整合同，[_validate_registration](agent/hooks.py:138:4-154:13) 和 [_merge_hook_result](agent/hooks.py:156:4-189:21) 也随之简化。

---

## 八、三份雷同的 SQLite 存储 + 65 处手写序列化

- `objective/store/sqlite.py`(791 行)、[agent/storage/sqlite.py](agent/storage/sqlite.py)(567 行)、`scheduler/store/sqlite.py`(384 行) 是三份结构雷同的 append-only/CRUD 实现。`utils/storage_support/` 已有共享 runtime，可以再上一层：提供通用的 "append-only log 表 + 序列校验 + 分页查询" 基建，三处只声明表名和 codec。
- 全库有 **65 个手写 [to_dict](objective/log.py:75:4-83:9)/[from_dict](objective/log.py:85:4-100:9)**，其中 `datetime.fromisoformat(raw) if isinstance(raw, str) else raw` 和 enum 恢复逻辑反复出现（如 `@agiwo/objective/log.py:86-101`）。不必引入 pydantic，一个 20 行的 `codec` helper（处理 datetime/enum/嵌套 dataclass）就能砍掉每个 [from_dict](objective/log.py:85:4-100:9) 里一半的防御代码。

---

## 九、可直接清理的死代码与遗留双入口

- **死参数**：`@agiwo/objective/service.py:183-189` 的 `auto_start_dispatcher` 收下后只有一句 `pass` 和注释"caller 仍需自己 await start_dispatcher()"——参数无效，删掉或实现。
- **遗留双入口**：`@agiwo/agent/llm_caller.py:190-213` 的 [stream_assistant_step](agent/llm_caller.py:189:0-212:28) 自注"Prefer [execute_model_call](agent/llm_caller.py:77:0-186:63)"，生产代码已无外部调用（仅一处测试引用），可移入测试或删除。
- **一行包装**：[_user_message_to_dict(message)](objective/service.py:157:0-158:28) 就是 [message.to_dict()](objective/log.py:75:4-83:9)（`@agiwo/objective/service.py:157-158`），内联即可。
- **幽灵签名**：`@agiwo/scheduler/engine.py:816-818` 的 [release_resume_barrier(*args, **kwargs)](scheduler/engine.py:815:4-826:36) 收任意参数又立刻 `del`，掩盖了真实契约，改成无参。

---

# 优先级建议

| 顺序 | 改动 | 预估收益 |
| --- | --- | --- |
| 1 | ObjectiveService 幂等模板方法 + `FactBatch` | -500 行，幂等/序列两类 bug 源头消失 |
| 2 | `RunStateWriter.emit()` 泛型化 + 投影回调内置 | -700 行，消除时间耦合 |
| 3 | `run_loop` 显式 `LoopExit` 结果 | 控制流可读性质变 |
| 4 | hooks 单表 `PHASE_SPECS`、`RuntimeState.find_agent` | 局部认知负荷下降 |
| 5 | scheduler 文件重命名/合并 + AGENTS.md 对齐 | 导航成本下降 |
| 6 | 存储基建共享、codec helper、死代码清理 | 长尾收益 |

这些改动全部是**内部重构**，不触碰 `agiwo.agent` / `agiwo.objective` / `agiwo.scheduler` 的公开 API 边界，与 AGENTS.md 记录的架构决策（三层事件溯源、深模块 facade）方向一致——做的是"把仪式收进机器"，不是推翻架构。