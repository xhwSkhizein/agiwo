# Agiwo Console

Agiwo Console 是 Agiwo SDK 的控制平面仪表盘，提供 Web UI 和 API 来管理和监控 AI Agent 的运行。

## 架构概述

Console 采用 **FastAPI + Next.js** 前后端分离架构：

- **Backend**: FastAPI (`server/`) — port 8422
- **Frontend**: Next.js + React + Tailwind + Lucide (`web/`) — port 3001
- **数据共享**: 通过 `pip install -e ..` 直接复用 agiwo SDK 的 Storage 类和数据模型

## 目录结构

```
console/
├── server/                   # FastAPI 后端
│   ├── app.py              # FastAPI 入口，lifespan 管理
│   ├── config.py           # ConsoleConfig (pydantic-settings, env prefix: AGIWO_CONSOLE_)
│   ├── dependencies.py     # 全局依赖实例 (StorageManager, AgentRegistry, Scheduler)
│   ├── schemas.py          # API 层 Pydantic 响应模型
│   ├── routers/            # API 路由
│   │   ├── sessions.py     # Sessions/Runs API
│   │   ├── traces.py       # Traces API (含 SSE 实时推送)
│   │   ├── agents.py       # Agent 配置 CRUD
│   │   ├── chat.py         # 实时对话 SSE 流
│   │   ├── scheduler.py    # Scheduler Agent States API
│   │   ├── scheduler_chat.py  # Scheduler Chat SSE + Cancel
│   │   └── feishu.py       # 飞书渠道 Webhook
│   ├── services/           # 业务服务层
│   │   ├── storage_manager.py   # 存储连接管理 (RunStep + Trace + AgentState)
│   │   ├── agent_registry.py    # Agent 配置持久化 (SQLite/MongoDB)
│   │   └── agent_builder.py     # build_model, build_agent, serialize_event
│   ├── channels/           # 渠道集成
│   │   └── feishu.py       # 飞书渠道服务
│   └── tools.py            # Console 专用工具
├── web/                    # Next.js 前端
│   ├── src/
│   │   ├── app/            # Next.js App Router
│   │   │   ├── page.tsx           # 首页 (Agent 列表)
│   │   │   ├── layout.tsx         # 根布局
│   │   │   ├── globals.css        # 全局样式
│   │   │   ├── agents/            # Agent 管理页面
│   │   │   │   ├── [id]/          # Agent 详情
│   │   │   │   │   ├── page.tsx   # Agent 详情页
│   │   │   │   │   ├── scheduler-chat/  # Scheduler 对话
│   │   │   │   │   └── chat/      # 普通对话
│   │   │   │   └── new/           # 新建 Agent
│   │   │   ├── sessions/          # Session 管理
│   │   │   ├── traces/            # Trace 观测
│   │   │   └── scheduler/         # Scheduler 状态监控
│   │   ├── components/     # React 组件
│   │   └── lib/            # 工具函数
│   │       └── api.ts       # API 客户端
│   ├── package.json        # npm 依赖
│   ├── next.config.ts      # Next.js 配置
│   └── tsconfig.json       # TypeScript 配置
├── tests/                  # 后端测试
│   ├── test_scheduler_api.py
│   └── test_scheduler_chat_api.py
├── pyproject.toml          # Python 项目配置
├── .env.example            # 环境变量示例
└── .env.example.full       # 完整环境变量示例
```

## 核心设计

### 1. 配置分层

Console 采用**分层配置**策略：

- **SDK 配置** (`AGIWO_*`): 存储路径、LLM Provider 密钥等核心能力配置
- **Console 配置** (`AGIWO_CONSOLE_*`): 服务器端口、CORS、默认 Agent、飞书等部署/渠道配置

```python
# Console 只定义自己的独特设置
class ConsoleConfig(BaseSettings):
    env_prefix = "AGIWO_CONSOLE_"
    host: str = "0.0.0.0"
    port: int = 8422
    # ...
    
    # 存储路径继承自 SDK
    @property
    def sqlite_db_path(self) -> str:
        return sdk_settings.resolve_path(sdk_settings.sqlite_db_path)
```

### 2. 全局依赖管理

使用模块级变量 + setter 函数管理全局依赖实例：

```python
# dependencies.py
_storage_manager: StorageManager | None = None

def set_storage_manager(manager: StorageManager) -> None:
    global _storage_manager
    _storage_manager = manager

def get_storage_manager() -> StorageManager:
    if _storage_manager is None:
        raise RuntimeError("StorageManager not initialized")
    return _storage_manager
```

在 `app.py` lifespan 中初始化：

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    config = ConsoleConfig()
    set_console_config(config)
    
    storage_manager = StorageManager(config)
    set_storage_manager(storage_manager)
    
    agent_registry = AgentRegistry(config)
    await agent_registry.initialize()
    set_agent_registry(agent_registry)
    
    scheduler = Scheduler(...)  # 全局 Scheduler 单例
    set_scheduler(scheduler)
    
    yield  # 应用运行期间
    
    # 清理
    await scheduler.shutdown()
```

### 3. StorageManager — 统一存储管理

管理三种存储后端：

- **RunStepStorage**: Agent 运行历史 (SQLite/MongoDB)
- **TraceStorage**: 可观测性数据 (SQLite/MongoDB/InMemory)
- **AgentStateStorage**: Scheduler 状态 (SQLite/InMemory)

```python
class StorageManager:
    def __init__(self, config: ConsoleConfig):
        self.config = config
        self._run_step_storage: RunStepStorage | None = None
        self._trace_storage: BaseTraceStorage | None = None
        self._agent_state_storage: AgentStateStorage | None = None
```

### 4. AgentRegistry — Agent 配置持久化

支持 SQLite 和 MongoDB 两种后端，提供：

- `list_agents()` — 列出所有 Agent 配置
- `get_agent(agent_id)` — 获取单个 Agent
- `save_agent(config)` — 保存/更新 Agent
- `delete_agent(agent_id)` — 删除 Agent

Agent 配置包含：

```python
class AgentConfigRecord:
    id: str                    # Agent ID
    name: str                  # 显示名称
    description: str           # 描述
    model_provider: str        # LLM Provider
    model_name: str            # 模型名称
    system_prompt: str         # System Prompt
    options: dict              # AgentOptions (max_steps, timeout等)
    model_params: dict         # 模型参数 (temperature, top_p等)
```

### 5. Scheduler 集成

Console 创建**全局 Scheduler 单例**，与 Agent 深度集成：

- `POST /api/scheduler/chat/{agent_id}` — SSE 流式对话
- `POST /api/scheduler/chat/{agent_id}/cancel` — 取消执行
- `GET /api/scheduler/states` — 列出所有 Agent 状态
- `GET /api/scheduler/states/{id}` — 获取单个状态
- `GET /api/scheduler/states/{id}/children` — 获取子 Agent 列表

Scheduler Chat 采用 **Hooks-based 事件转发**：

```python
# 创建 asyncio.Queue 收集事件
asyncio.Queue()

# 通过 AgentHooks.on_event 转发到队列
hooks = AgentHooks(on_event=lambda e: queue.put_nowait(e))

# 后台任务 submit + wait_for
asyncio.create_task(_run_and_signal(scheduler, agent_id, state, queue))

# SSE 将事件流发送给客户端
async for event in _event_generator(queue, state):
    yield f"data: {json.dumps(event)}\n\n"
```

### 6. 前端架构

**Next.js 15 + React 19 + Tailwind CSS 4** 技术栈：

- **App Router** (`app/`) — 现代 Next.js 路由
- **Server Components** 默认，交互部分用 Client Components
- **Tailwind CSS 4** — 原子化样式
- **Lucide React** — 图标库
- **SSE 流处理** — 实时接收 Agent 输出

关键页面：

- `/` — Agent 列表页，展示所有 Agent 配置
- `/agents/[id]` — Agent 详情，查看/编辑配置
- `/agents/[id]/chat` — 普通对话 (直接 Agent.run)
- `/agents/[id]/scheduler-chat` — Scheduler 编排对话
- `/sessions` — Session 历史查询
- `/traces` — Trace 观测数据
- `/scheduler` — Scheduler 运行状态监控

### 7. 飞书渠道集成

飞书机器人通过 Webhook 接收消息，使用 Scheduler 异步处理：

```python
class FeishuChannelService:
    def __init__(self, config: ConsoleConfig, scheduler: Scheduler):
        self.config = config
        self.scheduler = scheduler
    
    async def handle_message(self, message: FeishuMessage):
        # 1. 去重检查
        # 2. 获取或创建对应 Agent
        # 3. 使用 scheduler.submit_and_subscribe() 异步执行
        # 4. 流式回复飞书消息
```

## 使用方法

### 1. 环境准备

```bash
# 安装后端依赖
cd console
uv pip install -e ".[dev]"

# 安装前端依赖
cd web
npm install
```

### 2. 配置环境变量

复制 `.env.example` 到 `.env` 并配置：

```bash
# SDK 配置
AGIWO_SQLITE_DB_PATH=agiwo.db
DEEPSEEK_API_KEY=your-api-key

# Console 配置
AGIWO_CONSOLE_HOST=0.0.0.0
AGIWO_CONSOLE_PORT=8422
AGIWO_CONSOLE_CORS_ORIGINS=["http://localhost:3001"]
```

### 3. 启动开发服务器

**方式一：同时启动前后端**

```bash
# 后端 (console 目录)
uv run python -m server.app

# 前端 (web 目录，新终端)
npm run dev
```

**方式二：只启动后端 API**

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8422 --reload
```

### 4. 访问控制台

- 前端界面: http://localhost:3001
- API 文档: http://localhost:8422/docs

## API 概览

### Agent 管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/agents` | 列出所有 Agent |
| GET | `/api/agents/{id}` | 获取 Agent 详情 |
| POST | `/api/agents` | 创建 Agent |
| PUT | `/api/agents/{id}` | 更新 Agent |
| DELETE | `/api/agents/{id}` | 删除 Agent |

### 对话

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat/{agent_id}` | 普通对话 SSE |
| POST | `/api/scheduler/chat/{agent_id}` | Scheduler 对话 SSE |
| POST | `/api/scheduler/chat/{agent_id}/cancel` | 取消 Scheduler 执行 |

### Sessions & Traces

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/sessions` | 列出 Sessions |
| GET | `/api/sessions/{id}/runs` | 获取 Session 的 Runs |
| GET | `/api/sessions/{id}/runs/{run_id}/steps` | 获取 Run 的 Steps |
| GET | `/api/traces` | 列出 Traces |
| GET | `/api/traces/stream` | SSE 实时 Trace 流 |

### Scheduler 状态

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/scheduler/states` | 列出所有 Agent 状态 |
| GET | `/api/scheduler/states/{id}` | 获取单个状态 |
| GET | `/api/scheduler/states/{id}/children` | 获取子 Agent |
| GET | `/api/scheduler/stats` | 获取统计信息 |

## 开发指南

### 添加新的 API 路由

```python
# server/routers/my_feature.py
from fastapi import APIRouter, Depends
from server.dependencies import get_storage_manager

router = APIRouter(prefix="/api/my-feature")

@router.get("/items")
async def list_items(
    storage=Depends(get_storage_manager)
):
    return {"items": []}
```

注册路由：

```python
# server/app.py
from server.routers import my_feature

app.include_router(my_feature.router)
```

### 添加新的前端页面

```bash
# 创建页面目录
mkdir -p web/src/app/my-feature

# 创建 page.tsx
touch web/src/app/my-feature/page.tsx
```

```tsx
// web/src/app/my-feature/page.tsx
export default function MyFeaturePage() {
  return (
    <div>
      <h1>My Feature</h1>
    </div>
  );
}
```

### 添加 API 客户端方法

```typescript
// web/src/lib/api.ts
export async function fetchMyFeature(): Promise<MyFeatureResponse> {
  const response = await fetch(`${API_BASE}/api/my-feature/items`);
  if (!response.ok) throw new Error('Failed to fetch');
  return response.json();
}
```

### 运行测试

```bash
# 后端测试
cd console
pytest tests/ -v

# 特定测试
pytest tests/test_scheduler_api.py -v
```

### 代码规范

- **Backend**: 遵循 Agiwo SDK 规范，所有 import 放文件顶部，类型注解完整
- **Frontend**: 
  - 使用 TypeScript 严格模式
  - Server Components 优先，需要交互再用 Client Components
  - Tailwind CSS 原子化样式
  - 图标使用 Lucide React

## 环境变量参考

### SDK 配置 (`AGIWO_*`)

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `AGIWO_DEBUG` | 调试模式 | `false` |
| `AGIWO_LOG_LEVEL` | 日志级别 | `INFO` |
| `AGIWO_SQLITE_DB_PATH` | SQLite 数据库路径 | `agiwo.db` |
| `AGIWO_MONGO_URI` | MongoDB URI | - |
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `DEEPSEEK_API_KEY` | Deepseek API Key | - |
| `ANTHROPIC_API_KEY` | Anthropic API Key | - |

### Console 配置 (`AGIWO_CONSOLE_*`)

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `AGIWO_CONSOLE_HOST` | 服务器绑定地址 | `0.0.0.0` |
| `AGIWO_CONSOLE_PORT` | 服务器端口 | `8422` |
| `AGIWO_CONSOLE_CORS_ORIGINS` | CORS 允许域名 | `["http://localhost:3000", "http://localhost:3001"]` |
| `AGIWO_CONSOLE_RUN_STEP_STORAGE_TYPE` | RunStep 存储类型 | `sqlite` |
| `AGIWO_CONSOLE_TRACE_STORAGE_TYPE` | Trace 存储类型 | `sqlite` |
| `AGIWO_CONSOLE_METADATA_STORAGE_TYPE` | 元数据存储类型 | `sqlite` |
| `AGIWO_CONSOLE_DEFAULT_AGENT_NAME` | 默认 Agent 名称 | `Walaha` |
| `AGIWO_CONSOLE_FEISHU_ENABLED` | 启用飞书渠道 | `false` |
| `AGIWO_CONSOLE_FEISHU_APP_ID` | 飞书 App ID | - |
| `AGIWO_CONSOLE_FEISHU_APP_SECRET` | 飞书 App Secret | - |

完整配置参见 `.env.example.full`

## 部署建议

### 生产环境配置

```bash
# .env
AGIWO_DEBUG=false
AGIWO_LOG_LEVEL=INFO
AGIWO_LOG_JSON=true

AGIWO_CONSOLE_HOST=0.0.0.0
AGIWO_CONSOLE_PORT=8422
AGIWO_CONSOLE_CORS_ORIGINS=["https://your-domain.com"]

# 使用 SQLite 或 MongoDB
AGIWO_CONSOLE_RUN_STEP_STORAGE_TYPE=sqlite
AGIWO_CONSOLE_TRACE_STORAGE_TYPE=sqlite
AGIWO_CONSOLE_METADATA_STORAGE_TYPE=sqlite
```

### Docker 部署 (示例)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -e ".[dev]"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8422"]
```

### 前端构建

```bash
cd web
npm run build
# 输出到 .next/，可部署到 Vercel/Netlify 等平台
```
