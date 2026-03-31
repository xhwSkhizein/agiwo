# Memory System — 核心组件详解

## 1. 配置

Memory 配置通过全局 `AgiwoSettings`（`agiwo/config/settings.py`）加载，而非独立的 `MemoryConfig` dataclass。

关键配置项：

| 配置项 | 默认值 | 说明 |
|---------|--------|------|
| `memory_top_k` | `5` | 默认返回结果数 |
| `memory_chunk_tokens` | `400` | 每块最大 token 数 |
| `memory_chunk_overlap_tokens` | `80` | 相邻块重叠 token 数 |
| `relevant_memory_max_token` | `2048` | 注入 prompt 的最大记忆 token |

构造 `MemoryRetrievalTool` / `WorkspaceMemoryService` 时可通过参数覆盖：

```python
from agiwo.tool.builtin.retrieval_tool.tool import MemoryRetrievalTool

tool = MemoryRetrievalTool(
    top_k=10,
    embedding_provider="openai",
)
```

`embedding_provider = "auto"` 时尝试使用 OpenAI 兼容 API，降级纯 BM25。不设 embedding 时传入 `"disabled"` 即可。

---

## 2. MemoryChunker（文本切割）

位置：`agiwo/memory/chunker.py`

**职责**：将 Markdown 文件切割为多个重叠 chunk。

```python
@dataclass
class MemoryChunk:
    chunk_id: str        # sha256(f"{path}:{start_line}")[:16]
    path: str            # 相对于 workspace 的文件路径
    start_line: int
    end_line: int
    text: str
    content_hash: str    # sha256(text)，用于增量跳过
```

**切割策略**：
- 按 token 数量切割（使用 `tiktoken` 编码，与 OpenAI Embedding 对齐）
- 超出 `chunk_tokens` 限制时分块，复用前一块末尾 N 个 token 保持上下文连续性

---

## 3. Embedding 层

位置：`agiwo/embedding/`

项目通过 `agiwo/embedding/` 包统一管理 Embedding 能力。

```python
from agiwo.embedding import EmbeddingModel, EmbeddingFactory

embedder = EmbeddingFactory.create(
    provider="openai",
    model="text-embedding-3-small",
    dimensions=1536,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# 批量向量化
embeddings = await embedder.embed(["text1", "text2"])
```

**Embedding 缓存**（存储在 SQLite 中）：

```sql
CREATE TABLE embedding_cache (
    content_hash TEXT NOT NULL,
    model_id     TEXT NOT NULL,
    embedding    BLOB NOT NULL,
    updated_at   INTEGER NOT NULL,
    PRIMARY KEY (content_hash, model_id)
);
```

---

## 4. MemoryIndexStore（核心存储 + 索引管理）

位置：`agiwo/memory/index_store.py`

**职责**：持有 SQLite 连接，协调 Chunker，暴露 `sync_files()` 等接口。

### 4.1 Schema

```sql
-- 已索引文件记录
CREATE TABLE files (
    path     TEXT PRIMARY KEY,
    hash     TEXT NOT NULL,
    mtime    INTEGER NOT NULL,
    size     INTEGER NOT NULL
);

-- 文本块主表
CREATE TABLE chunks (
    chunk_id   TEXT PRIMARY KEY,
    path       TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    model_id   TEXT NOT NULL DEFAULT '',
    text       TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX idx_chunks_path ON chunks(path);

-- 向量虚拟表（sqlite-vec 扩展，可选）
CREATE VIRTUAL TABLE chunks_vec USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding FLOAT[{dims}]
);

-- 全文索引虚拟表（FTS5）
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    chunk_id UNINDEXED,
    path UNINDEXED,
    tokenize = "unicode61"
);
```

### 4.2 初始化

MemoryIndexStore 通过构造函数接受 `workspace_dir` 和 `embedder`：

```python
class MemoryIndexStore:
    def __init__(self, workspace_dir: Path, embedder=None):
        self._workspace_dir = workspace_dir
        self._db_path = workspace_dir / "memory.db"
        self._embedder = embedder  # EmbeddingModel 实例或 None
```

### 4.3 增量索引（sync_files）

```
sync_files():
  1. 扫描 MEMORY/*.md
  2. 对每个文件，计算 (mtime, size) 与 files 表比对
  3. 变更文件 → index_file(path)
  4. 已删除文件 → remove_file(path)

index_file(path):
  1. 读取文件内容，计算 sha256 全文 hash
  2. Chunker.chunk_file() → list[MemoryChunk]
  3. 按 content_hash 批量查询 embedding_cache
  4. 对 cache_miss chunks 批量调用 embedder.embed()
  5. 删除该 path 的旧 chunks
  6. 批量 INSERT 新 chunks
```

---

## 5. HybridSearcher（混合检索）

位置：`agiwo/memory/searcher.py`

**职责**：给定 query 字符串，返回 `list[SearchResult]`。

```python
@dataclass
class SearchResult:
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str
    score: float
    vector_score: float
    bm25_score: float
```

**检索流程详见 [data-flow.md](./data-flow.md)。**

---

## 6. WorkspaceMemoryService（业务编排层）

位置：`agiwo/memory/service.py`

`MemoryRetrievalTool` 实际调用的是 `WorkspaceMemoryService`，而非直接使用 `MemoryIndexStore`。

```python
class WorkspaceMemoryService:
    async def search(self, agent_name, agent_id, query, top_k) -> tuple[Workspace | None, list[SearchResult]]:
        # 1. 解析 workspace 目录
        # 2. 初始化 MemoryIndexStore
        # 3. 增量同步 MEMORY/ 目录
        # 4. 混合检索
        # 5. 返回结果
```

---

## 7. MemoryRetrievalTool（对外入口）

位置：`agiwo/tool/builtin/retrieval_tool/tool.py`

内部使用 `WorkspaceMemoryService` 而非 `MemoryIndexStore`。

### 执行流程

```python
async def execute(self, parameters, context, abort_signal):
    if abort_signal and abort_signal.is_aborted():
        return ToolResult.aborted(tool_name=self.name, ...)

    workspace, results = await self._memory_service.search(
        agent_name=context.agent_name,
        agent_id=context.agent_id,
        query=parameters.get("query", ""),
        top_k=parameters.get("top_k", self._top_k),
    )

    return ToolResult.success(tool_name=self.name, content=format_results(results))
```

---

## 8. Prompt 集成

位置：`agiwo/agent/prompt.py`

在 `_render_environment()` 中注入 Memory Recall 指令：

```
## Memory Recall

Before answering anything about prior work, decisions, dates, or preferences:
call `memory_retrieval` to search MEMORY/ files, then use the returned
file paths + line numbers to read precise sections if needed.
```

**双轨机制**：
- **自动注入**：近期记忆直接在 context 中，Agent 无需显式调用工具即可感知
- **按需检索**：历史记忆通过 `memory_retrieval` 工具精准召回，不污染 context
