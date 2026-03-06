# Agiwo SDK

Agiwo 是一个 Python AI Agent SDK，提供完整的 Agent 构建、调度、观测和工具扩展能力。

## 目录结构

```
agiwo/
├── agent/                    # Agent 核心实现
│   ├── agent.py              # Agent 类：唯一公开入口 (run / run_stream)
│   ├── execution_context.py  # ExecutionContext + SessionSequenceCounter
│   ├── hooks.py              # AgentHooks：生命周期钩子 (dataclass, 全部可选)
│   ├── options.py            # AgentOptions + 存储配置
│   ├── schema.py             # 核心数据模型: Run, StepRecord, StreamEvent, RunOutput 等
│   ├── stream_channel.py     # StreamChannel：asyncio.Queue 事件流通道
│   ├── inner/                # 内部实现 (不对外暴露)
│   │   ├── executor.py       # AgentExecutor：LLM + Tool 执行循环
│   │   ├── event_emitter.py  # EventEmitter：纯事件发射
│   │   ├── llm_handler.py    # LLM 流式响应处理
│   │   ├── message_assembler.py  # 消息组装 (system prompt + history + memory)
│   │   ├── run_state.py      # 运行状态追踪
│   │   ├── storage_sink.py   # StorageSink：事件流中间件，消费事件完成持久化
│   │   ├── step_builder.py   # StepRecord 构建
│   │   ├── summarizer.py     # 终止摘要生成
│   │   └── system_prompt_builder.py  # System Prompt 动态构建
│   ├── memory_hooks.py       # 默认记忆钩子实现
│   └── storage/              # Run/Step 持久化
│       ├── base.py           # RunStepStorage ABC + InMemoryRunStepStorage
│       ├── factory.py        # StorageFactory：根据配置创建存储实例
│       ├── session.py        # SessionStorage 会话隔离层
│       ├── sqlite.py         # SQLite 实现
│       └── mongo.py          # MongoDB 实现
├── llm/                      # LLM Provider 抽象层
│   ├── base.py               # Model ABC + StreamChunk
│   ├── openai.py             # OpenAIModel (OpenAI 兼容 Provider 的基类)
│   ├── anthropic.py          # AnthropicModel (独立实现)
│   ├── deepseek.py           # DeepseekModel (继承 OpenAIModel)
│   ├── nvidia.py             # NvidiaModel (继承 OpenAIModel)
│   └── helper.py             # 工具函数: JSON 解析, usage 标准化
├── tool/                     # Tool 系统
│   ├── base.py               # BaseTool ABC + ToolResult + ToolDefinition
│   ├── agent_tool.py         # AgentTool + as_tool()：Agent 作为 Tool 嵌套
│   ├── executor.py           # ToolExecutor：批量并行执行 + 缓存
│   ├── cache.py              # ToolResultCache
│   ├── builtin/              # 内置工具 (装饰器自动注册)
│   │   ├── registry.py       # @builtin_tool + @default_enable 装饰器
│   │   ├── bash_tool/        # Bash 命令执行
│   │   ├── http_client.py    # HTTP 请求
│   │   ├── html_extract.py   # HTML 内容提取
│   │   ├── llm_client.py     # LLM 调用工具
│   │   ├── web_reader/       # 网页阅读工具
│   │   ├── web_search/       # 网页搜索工具
│   │   ├── retrieval_tool/   # RAG 检索工具
│   │   └── config.py         # 内置工具配置
│   ├── permission/           # 工具权限管理
│   └── storage/              # 工具存储相关
├── scheduler/                # Agent 调度系统
│   ├── models.py             # AgentState, WakeCondition, SchedulerConfig 等
│   ├── executor.py           # SchedulerExecutor：Agent 执行 + 输出事件发射
│   ├── store.py              # AgentStateStorage ABC + 实现
│   ├── guard.py              # TaskGuard：集中式护栏 (spawn/wake 限制, 超时检测)
│   ├── tools.py              # SpawnAgentTool, SleepAndWaitTool, QuerySpawnedAgentTool
│   └── scheduler.py          # Scheduler：编排层入口
├── skill/                    # Skill 系统
│   ├── manager.py            # SkillManager：发现、加载、热重载
│   ├── registry.py           # SkillRegistry + SkillMetadata
│   ├── loader.py             # SkillLoader：SKILL.md 解析
│   ├── skill_tool.py         # SkillTool：将 Skill 暴露为 Tool
│   └── exceptions.py         # Skill 相关异常
├── observability/            # 可观测性
│   ├── base.py               # BaseTraceStorage ABC
│   ├── trace.py              # Trace + Span 模型 (兼容 OpenTelemetry)
│   ├── collector.py          # TraceCollector：从 StreamEvent 流构建 Trace
│   ├── otlp_exporter.py      # OTLP 导出器
│   ├── store.py              # MongoDB TraceStorage
│   └── sqlite_store.py       # SQLite TraceStorage
├── config/
│   └── settings.py           # AgiwoSettings (pydantic-settings, env prefix: AGIWO_)
└── utils/
    ├── abort_signal.py       # AbortSignal：优雅取消
    ├── logging.py            # structlog 日志配置
    ├── retry.py              # 异步重试装饰器
    ├── sqlite_pool.py        # SQLite 连接池
    └── tojson.py             # JSON 序列化工具
```

## 核心设计

### 1. Agent 架构

Agent 是**具体类** (非 ABC)，负责配置持有和执行生命周期。两个公开方法：

- `run(user_input, ...)` -> `RunOutput` — 阻塞执行
- `run_stream(user_input, ...)` -> `AsyncIterator[StreamEvent]` — 流式执行

Agent 不做 LLM 调用和 Tool 执行，委托给 `AgentExecutor`。

### 2. 事件驱动架构 (Event Pipeline)

核心执行通过 `EventEmitter` 发射事件到 `StreamChannel`，所有副作用在事件流下游处理：

```
Executor → EventEmitter → Channel → StorageSink(持久化) → TraceCollector(Trace) → User
```

- **StorageSink**：消费事件完成 Run/Step 持久化，支持嵌套 Agent 多 Run 追踪
- **TraceCollector**：消费事件构建 Trace + Span，兼容 OpenTelemetry，支持 OTLP 导出

### 3. Agent 嵌套 (Agent as Tools)

通过 `AgentTool` 或 `as_tool()` 将 Agent 包装为 Tool，实现 **Main Agent + Agent as Tools** 模式：

```python
from agiwo import Agent, as_tool
from agiwo.llm import DeepseekModel

research_agent = Agent(
    name="researcher",
    description="Research specialist",
    model=DeepseekModel(id="deepseek-chat"),
)

main_agent = Agent(
    name="main",
    model=DeepseekModel(id="deepseek-chat"),
    tools=[as_tool(research_agent)],
)
```

内置深度限制和循环引用检测。

### 4. Scheduler 异步编排

`Scheduler` 是 Agent 之上的编排层，管理 Agent 的 spawn/sleep/wake/completion 生命周期：

- **三种 API 模式**：
  - `run()` — 阻塞等待
  - `submit()` + `wait_for()` — 非阻塞，获取最终结果
  - `submit_and_subscribe()` — 输出流，逐条接收 Agent 文本产出

- **WakeCondition 类型**：
  - `WAITSET` — 等子 Agent
  - `TIMER` — 一次性延迟
  - `PERIODIC` — 周期唤醒
  - `TASK_SUBMITTED` — 新任务提交

- **TaskGuard**：集中式护栏，所有 spawn/wake 限制检查收敛于此

### 5. LLM Provider 抽象

`@dataclass` + ABC。子类必须实现 `arun_stream(messages, tools)`：

- **OpenAI 兼容 API**：继承 `OpenAIModel`，仅覆写 `_resolve_api_key` / `_resolve_base_url`
- **非兼容 API (如 Anthropic)**：直接继承 `Model`，完整实现 `arun_stream`

### 6. Tool 系统

ABC 基类，子类实现 5 个方法：

- `get_name()` — 工具名称
- `get_description()` — 工具描述
- `get_parameters()` — 参数 JSON Schema
- `is_concurrency_safe()` — 是否可并发执行
- `execute()` — 执行逻辑，返回 `ToolResult`

### 7. 配置治理

- **SDK 能力配置**放 `agiwo/config/settings.py` (env prefix: `AGIWO_`)
- **每个配置项必须有 owner**，新增配置前先判断归属
- **外部 Provider 键保留标准名**：如 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`
- **Console 不承载通用 SDK 运行参数**：Agent 行为参数应存入 `agent_configs.options/model_params`

## 使用方法

### 基础 Agent 使用

```python
import asyncio
from agiwo import Agent, AgentOptions
from agiwo.llm import DeepseekModel

async def main():
    agent = Agent(
        name="assistant",
        description="A helpful assistant",
        model=DeepseekModel(id="deepseek-chat"),
        system_prompt="You are a helpful assistant.",
    )
    
    # 阻塞执行
    result = await agent.run("Hello, how are you?")
    print(result.output)
    
    # 流式执行
    async for event in agent.run_stream("Tell me a story"):
        if event.type == "CONTENT":
            print(event.delta.content, end="")

asyncio.run(main())
```

### 使用 Tools

```python
from agiwo.tool import BaseTool, ToolResult
from agiwo.agent.execution_context import ExecutionContext
from agiwo.utils.abort_signal import AbortSignal

class CalculatorTool(BaseTool):
    def get_name(self) -> str:
        return "calculator"
    
    def get_description(self) -> str:
        return "Perform arithmetic calculations"
    
    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }
    
    def is_concurrency_safe(self) -> bool:
        return True
    
    async def execute(self, parameters, context, abort_signal=None):
        result = eval(parameters["expression"])
        return ToolResult(
            tool_name=self.name,
            tool_call_id=parameters.get("tool_call_id", ""),
            input_args=parameters,
            content=str(result),
            output=result,
            start_time=time.time(),
            end_time=time.time(),
            duration=0,
        )

agent = Agent(
    name="math-assistant",
    model=DeepseekModel(id="deepseek-chat"),
    tools=[CalculatorTool()],
)
```

### Scheduler 编排

```python
from agiwo import Scheduler

scheduler = Scheduler()

# 提交 Agent 异步执行
state = await scheduler.submit(agent, "Research Python async patterns")

# 等待完成
result = await scheduler.wait_for(state.id)
print(result.output)

# 或使用订阅模式实时获取输出
async for output in scheduler.submit_and_subscribe(agent, "Long task"):
    print(output.chunk, end="")
```

## 开发指南

### 环境设置

```bash
# 克隆仓库
git clone <repo-url>
cd agiwo

# 使用 uv 安装依赖
uv pip install -e ".[dev]"

# 运行测试
pytest tests/ -v
```

### 添加新的 LLM Provider

继承 `OpenAIModel` (如果是 OpenAI 兼容 API)：

```python
from agiwo.llm.openai import OpenAIModel

class MyProviderModel(OpenAIModel):
    provider = "myprovider"
    
    def _resolve_api_key(self) -> str | None:
        return os.getenv("MYPROVIDER_API_KEY")
    
    def _resolve_base_url(self) -> str | None:
        return "https://api.myprovider.com/v1"
```

或者继承 `Model` (如果是非兼容 API)：

```python
from agiwo.llm.base import Model, StreamChunk

class MyCustomModel(Model):
    async def arun_stream(self, messages, tools=None):
        # 实现自己的流式调用逻辑
        async for chunk in self._call_api(messages, tools):
            yield StreamChunk(content=chunk.text)
```

### 添加新的 Tool

```python
from agiwo.tool.base import BaseTool, ToolResult

class MyTool(BaseTool):
    cacheable = True  # 启用缓存
    timeout_seconds = 60
    
    def get_name(self) -> str:
        return "my_tool"
    
    def get_description(self) -> str:
        return "Description for LLM to understand when to use this tool"
    
    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string"}
            },
            "required": ["param1"]
        }
    
    def is_concurrency_safe(self) -> bool:
        return True  # 是否可以并行执行
    
    async def execute(self, parameters, context, abort_signal=None):
        # 实现执行逻辑
        return ToolResult(
            tool_name=self.name,
            tool_call_id=parameters.get("tool_call_id", ""),
            input_args=parameters,
            content="Result for LLM",
            output=raw_result,
            start_time=start,
            end_time=end,
            duration=end - start,
        )
```

### 内置工具注册

使用装饰器自动注册内置工具：

```python
from agiwo.tool.builtin.registry import builtin_tool, default_enable

@builtin_tool
@default_enable  # 默认启用
class MyBuiltinTool(BaseTool):
    ...
```

### 代码规范

- **Python 3.11+**，不使用 `from __future__ import annotations`
- **所有 import 放在文件顶部**，禁止局部导入 / 延迟导入
- 类型注解：所有公开方法和核心数据结构必须有类型注解
- 命名：PascalCase 类名，snake_case 函数/变量
- `is not None` 做哨兵检查；truthy 检查仅在明确意图时使用
- 避免可变默认参数：用 `None` + `__init__` 内初始化
- async 代码保持显式，不在隐式 helper 中藏 await
- **禁止向后兼容** (除非明确要求)，及时删除遗留代码
- **循环依赖**：遇到时必须重构组件耦合关系，禁止延迟导入等绕行手段
- 不要使用 `rm` 删除文件，用 `mv` 移动到 `trash/` 目录

## 环境变量配置

| 变量 | 说明 | 示例 |
|------|------|------|
| `AGIWO_DEBUG` | 调试模式 | `true` |
| `AGIWO_LOG_LEVEL` | 日志级别 | `INFO` |
| `AGIWO_ROOT_PATH` | 数据根目录 | `.agiwo` |
| `AGIWO_SQLITE_DB_PATH` | SQLite 数据库路径 | `agiwo.db` |
| `AGIWO_MONGO_URI` | MongoDB URI | `mongodb://localhost:27017` |
| `OPENAI_API_KEY` | OpenAI API Key | `sk-...` |
| `DEEPSEEK_API_KEY` | Deepseek API Key | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API Key | `sk-ant-...` |

完整配置列表参见 `agiwo/config/settings.py`
