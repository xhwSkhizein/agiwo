# PR 49 Review

**PR 标题**: refactor: tools & skills use global manager to load

**Branch**: `refactor/agent-tools-skills-config`

---

## 🔴 严重问题

### 1. `_CHILD_EXCLUDED_TOOLS` 定义但未使用

**位置**: `agiwo/agent/agent.py:36`

```python
_CHILD_EXCLUDED_TOOLS: frozenset[str] = frozenset({"spawn_agent"})
```

这个常量只在 PR 中定义，但在 `create_child_agent` 方法中过滤 spawn_agent 时使用的是内联的 `{"spawn_agent"}`，没有使用这个常量。

**问题**: 重复定义，维护困难，容易 drift。

**修复建议**: 删除常量或在 `create_child_agent` 中使用。

---

### 2. `Agent._extra_tools` 的继承逻辑设计缺陷

**位置**: `agiwo/agent/agent.py:221-231`

```python
parent_extra = (
    list(self._extra_tools)
    if inherit_all_extra_tools
    else self._get_inheritable_extra_tools()
)
merged_extra = parent_extra + list(extra_tools or [])
```

**问题**:
- `inherit_all_extra_tools=True` 时会继承**所有** extra_tools，包括 AgentTool (可能产生循环引用风险)
- 没有验证继承的 tools 与子 agent 的 `allowed_tools` 是否有冲突
- `merged_extra` 可能出现重复工具（parent_extra 和 extra_tools 可能包含同名工具）

---

### 3. `validate_child_subset` 的 `None` 语义不一致

**位置**: `agiwo/agent/definition.py:28-30`

```python
def validate_child_subset(child, parent, label):
    if child is None or parent is None:
        return  # 直接返回，不验证
```

**问题**: 
- `child=[]` (空列表) 和 `child=None` (未限制) 的语义被混淆
- 如果 parent 是 `[]` (明确禁用所有 tools)，child 传 `None` 应该被限制，但当前逻辑允许通过

---

## 🟡 中等问题

### 4. `ToolManager._tool_cache` 的缓存键设计风险

**位置**: `agiwo/tool/manager.py:240-250`

```python
def _get_or_create_tool(self, name: str, tool_cls: Type[BaseTool]) -> BaseTool:
    if not tool_cls.is_stateless:
        return self._create_tool_instance(name, tool_cls)
    if name not in self._tool_cache:
        self._tool_cache[name] = self._create_tool_instance(name, tool_cls)
    return self._tool_cache[name]
```

**问题**: 缓存只按 `name` 索引，但同一 tool 可能因 `citation_store_config` 不同需要不同实例。当前设计不支持这种区分。

---

### 5. `allowed_tools` 中 `agent:` 前缀的处理位置分散

- 验证在 `ToolManager.normalize_allowed_tools()`
- 解析在 `ToolCatalog.list_available_tools()` 和其他地方
- AgentTool 的实际创建在 `build_agent()`

**问题**: 跨越多层的分散处理会导致后续维护困难，容易产生不一致。

---

### 6. `fork` 模式的参数覆盖逻辑过于隐式

**位置**: `agiwo/scheduler/runtime_tools.py:106-112`

```python
fork = bool(parameters.get("fork", False))
instruction = None if fork else parameters.get("instruction")
system_prompt = None if fork else parameters.get("system_prompt")
allowed_skills = None if fork else parameters.get("allowed_skills")
```

**问题**: 
- 调用方传入的参数被静默覆盖，无警告/错误
- 用户可能困惑为什么 `instruction` 和 `system_prompt` 没生效
- 应该在 `gate()` 中提前拒绝带参数的 fork 请求，或至少记录 warning

---

### 7. `skills_enabled()` 的文档和实现有微小不一致

**位置**: `agiwo/skill/allowlist.py:19-28`

```python
def skills_enabled(allowed_skills):
    """
    - None: All skills are allowed (skills system is enabled)
    - [] (empty list): No skills are allowed (skills system is effectively disabled)
    """
    normalized = normalize_allowed_skills(allowed_skills)
    return normalized is None or bool(normalized)
```

**问题**: 注释说 `[]` 是 "effectively disabled"，但函数返回 `True` 表示 enabled。语义混淆。

---

## 🟢 轻微问题 / 代码整洁

### 8. `ToolManager` 单例初始化存在 race condition

**位置**: `agiwo/tool/manager.py:30-35`

```python
def get_global_tool_manager(citation_store_config=None):
    global _GLOBAL_TOOL_MANAGER
    if _GLOBAL_TOOL_MANAGER is None:
        _GLOBAL_TOOL_MANAGER = ToolManager(citation_store_config=citation_store_config)
    return _GLOBAL_TOOL_MANAGER
```

**问题**: 非线程安全，虽然 Python GIL 下风险较低，但在 async 环境下仍可能多次初始化。

---

### 9. `build_fork_task_notice` 的 `<system-notice>` 格式与其他地方不一致

**位置**: `agiwo/scheduler/formatting.py:165-175`

```python
_FORK_NOTICE = (
    "<system-notice>\n"
    "You are a forked child agent..."
)
```

**问题**: Retrospect 模块也使用 `<system-notice>`，但格式可能不同。建议统一用工厂函数创建。

---

### 10. 测试文件 `test_tool_manager.py` 过大

367 行新增测试全部在一个文件，且有很多重复 setup。

**建议**: 按功能拆分为多个 test class 或文件。

---

## 📋 修复优先级

| 优先级 | 问题 | 修复建议 |
|--------|------|----------|
| P0 | `_CHILD_EXCLUDED_TOOLS` 未使用 | 删除常量，或在 `create_child_agent` 中使用 |
| P0 | `validate_child_subset` None 语义 | 明确区分 `None` (继承) vs `[]` (禁用) 的处理 |
| P1 | `fork` 参数覆盖 | 在 `gate()` 中拒绝非法组合，或添加 warning log |
| P1 | `Agent._extra_tools` 继承 | 验证继承的 tools 与 allowed_tools 的一致性 |
| P2 | `ToolManager` 缓存键 | 考虑加入 citation config 到缓存键 |
| P2 | `skills_enabled` 语义 | 修复注释或函数实现，保持一致 |
