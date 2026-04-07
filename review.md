# Code Review: Tool System Refactoring


# Issue 01: Skill Tool 处理过于底层

## 问题描述

在 `Agent.__init__` 中（`agiwo/agent/agent.py:105-145`），手动遍历工具列表来查找、比较、添加/替换/移除 skill tool：

```python
# 当前实现（问题代码）
initial_tools = list(tools) if tools else []
skill_manager = get_global_skill_manager()
new_allowed_skills = (
    frozenset(self._config.allowed_skills)
    if self._config.allowed_skills
    else None
)

# Find existing skill tool - 手动遍历
existing_skill_idx = None
for i, tool in enumerate(initial_tools):
    if tool.name == "skill":
        existing_skill_idx = i
        break

if new_allowed_skills is not None:
    # Skills enabled: add or replace skill tool
    if existing_skill_idx is not None:
        existing_tool = initial_tools[existing_skill_idx]
        # Replace if allowed_skills differ - 手动比较 frozenset
        if existing_tool._allowed_skills != new_allowed_skills:
            skill_tool = skill_manager.create_skill_tool(
                self._config.allowed_skills
            )
            initial_tools[existing_skill_idx] = skill_tool
    else:
        # No existing skill tool, add new one
        skill_tool = skill_manager.create_skill_tool(
            self._config.allowed_skills
        )
        initial_tools.append(skill_tool)
elif existing_skill_idx is not None:
    # Skills disabled (empty list): remove existing skill tool
    initial_tools.pop(existing_skill_idx)

self._tools = tuple(initial_tools)
```

### 问题

1. **违反单一职责**: `Agent` 不应该关心 skill tool 是如何被添加/移除的
2. **复杂度过高**: 35 行代码处理一个简单的"确保 skill tool 存在且配置正确"的需求
3. **暴露内部细节**: 需要访问 `existing_tool._allowed_skills`（私有属性）
4. **与 `ToolManager.get_tools()` 重复**: `ToolManager` 已经有添加 skill tool 的逻辑，但 `Agent` 在做另一套

---

# Issue 02: None vs [] 语义混淆

## 问题描述

`allowed_tools` 和 `allowed_skills` 都使用 `None` vs `[]` 来表达不同的语义：

| 值 | `allowed_tools` 语义 | `allowed_skills` 语义 |
|----|---------------------|----------------------|
| `None` | 使用默认 builtin tools | 所有 skills 都允许 |
| `[]` | 没有 builtin tools | skills 系统禁用 |
| `["item"]` | 仅指定 tools | 仅指定 skills |

### 问题代码

```python
# agiwo/tool/manager.py:170-175
def get_tools(self, allowed_tools: list[str] | None = None, ...):
    if allowed_tools is None:
        # Default: include all DEFAULT_TOOLS
        selected_tool_names = set(DEFAULT_TOOLS.keys())
    else:
        # Explicit allowlist: only include specified tools
        selected_tool_names = set(allowed_tools)  # 包括空集！
```

```python
# agiwo/skill/allowlist.py:18-24
def skills_enabled(allowed_skills: list[str] | tuple[str, ...] | None) -> bool:
    """Semantic meanings:
        - None: All skills are allowed (skills system is enabled)
        - [] (empty list): No skills are allowed (skills system is effectively disabled)
    Returns True if skills should be enabled (None or non-empty list).
    """
    normalized = normalize_allowed_skills(allowed_skills)
    return normalized is None or bool(normalized)
```

### 问题

1. **容易误用**: 开发者可能误以为 `[]` 表示"使用默认值"
2. **文档依赖**: 必须阅读文档才能理解区别
3. **API 不一致风险**: `AgentConfig(allowed_tools=[], allowed_skills=[])` 实际行为完全不同
4. **序列化歧义**: JSON `null` vs `[]` 在边界处容易混淆

---


# Issue 03: 子集验证逻辑重复

## 问题描述

`allowed_tools` 和 `allowed_skills` 的子集验证逻辑在多处重复，几乎相同的代码写两遍：

### 位置 1: `agiwo/agent/definition.py`

```python
# 验证 allowed_tools 是父集的子集（约 15 行）
if child_allowed_tools is not None:
    if parent_allowed_tools is not None:
        parent_allowed = set(parent_allowed_tools)
        disallowed = [
            name for name in child_allowed_tools if name not in parent_allowed
        ]
        if disallowed:
            tool_list = ", ".join(disallowed)
            raise ValueError(
                "child_allowed_tools must be a subset of the parent's "
                f"allowed_tools: {tool_list}"
            )

# 验证 allowed_skills 是父集的子集（约 15 行）
if child_allowed_skills is not None:
    if agent.config.allowed_skills is not None:
        parent_allowed = set(agent.config.allowed_skills)
        disallowed = [
            skill for skill in effective_allowed_skills if skill not in parent_allowed
        ]
        if disallowed:
            skill_list = ", ".join(disallowed)
            raise ValueError(
                "child_allowed_skills must be a subset of the parent's "
                f"allowed_skills: {skill_list}"
            )
```

### 位置 2: `agiwo/scheduler/tool_control.py`

```python
# 再次重复类似逻辑
if allowed_tools is not None:
    if parent_agent is not None and parent_agent.config.allowed_tools is not None:
        parent_allowed = set(parent_agent.config.allowed_tools)
        disallowed = [
            tool for tool in allowed_tools if tool not in parent_allowed
        ]
        if disallowed:
            tool_list = ", ".join(disallowed)
            raise ValueError(
                "Child allowed_tools must be a subset of the parent's "
                f"allowed_tools: {tool_list}"
            )
```

### 问题

1. **重复代码**: 同一逻辑写 3 遍，维护困难
2. **不一致风险**: 修改时可能漏掉某处
3. **错误消息不统一**: "child_allowed_tools" vs "Child allowed_tools" 大小写不一致
4. **难以测试**: 分散在各处，需要多处测试覆盖

---


# Issue 04: spawn_agent 硬编码过滤

## 问题描述

`spawn_agent` 工具的过滤逻辑在多处重复硬编码：

### 位置 1: `agiwo/agent/agent.py:260-273`

```python
def create_child_agent(...):
    builtin_tool_names = set(tool_manager.list_available_tool_names())
    parent_custom_tools = [
        tool
        for tool in self._tools
        if tool.name not in builtin_tool_names and tool.name != "spawn_agent"
    ]
```

### 位置 2: `agiwo/scheduler/runner.py:116-132`

```python
async def _activate_child(...) -> Agent:
    # Child agents get scheduling tools (spawn_agent filtered out to prevent grandchildren)
    scheduling_tools = [
        t for t in self._ctx.scheduling_tools if t.name != "spawn_agent"
    ]
    child = await parent.create_child_agent(
        child_id=state.id,
        child_allowed_tools=child_allowed_tools,
        extra_tools=list(scheduling_tools),  # 已经过滤过的
    )
```

### 位置 3: `agiwo/scheduler/engine.py:704-710`

```python
async def _create_child_agent_for_persistent(...) -> Agent:
    return await agent.create_child_agent(
        child_id=state_id,
        system_prompt_override=agent.config.system_prompt,
        child_allowed_tools=list(agent.config.allowed_tools) if ... else None,
        extra_tools=list(self._scheduling_tools),  # spawn_agent 应该在这里被过滤？
    )
```

### 问题

1. **魔法字符串**: `"spawn_agent"` 多次硬编码
2. **不一致风险**: 某处可能忘记过滤
3. **难以维护**: 如果要更改"子 Agent 不能 spawn"的策略，需要修改多处
4. **职责不清**: 是调用者负责过滤，还是被调用者负责过滤？

---

# Issue 06: agent: 前缀解析分散

## 问题描述

`agent:` 前缀（用于引用其他 Agent 作为工具）的解析逻辑分散在 API 层，而不是集中在 Tool 系统中：

### 当前实现

```python
# console/server/services/agent_registry/models.py:45-55
@model_validator(mode="before")
@classmethod
def _validate_allowed_tools(cls, data: dict) -> dict:
    # ...
    if normalized.get("allowed_tools") is not None:
        tool_manager = get_global_tool_manager()
        allowed_tools = list(normalized.get("allowed_tools"))

        # 分离 builtin 工具名和 agent 工具引用
        builtin_tools = [t for t in allowed_tools if not t.startswith("agent:")]
        agent_refs = [t for t in allowed_tools if t.startswith("agent:")]

        # 验证 agent 引用
        for ref in agent_refs:
            agent_id = ref[len("agent:") :].strip()
            if not agent_id:
                raise ValueError(f"Invalid tool reference: {ref!r}")

        # 仅验证 builtin 工具名
        if builtin_tools:
            validated_builtin = tool_manager.normalize_allowed_tools(builtin_tools)
            normalized["allowed_tools"] = list(validated_builtin) + agent_refs
        else:
            normalized["allowed_tools"] = agent_refs
    return normalized
```

### 问题

1. **职责错位**: API 层（`agent_registry/models.py`）不应该了解 `agent:` 前缀的实现细节
2. **重复风险**: 如果其他地方需要解析 `allowed_tools`，需要重复这个逻辑
3. **扩展困难**: 未来添加 `skill:` 或 `workflow:` 前缀时，需要修改多处
4. **测试分散**: 验证逻辑散落在 API 测试和工具测试中

---

