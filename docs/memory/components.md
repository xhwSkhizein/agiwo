# Memory System — 核心组件详解

## 1. MemoryConfig（配置）

位置：`agiwo/tool/builtin/config.py`

```python
@dataclass
class MemoryConfig:
    # Embedding
    embedding_provider: str = "auto"      # "openai" | "auto" | "disabled"
    embedding_model: str = ""             # 空则使用 provider 默认模型
    embedding_api_base: str = ""          # 空则读取 OPENAI_BASE_URL 环境变量
    embedding_api_key: str = ""           # 空则读取 OPENAI_API_KEY 环境变量
    embedding_dims: int = 1536            # text-embedding-3-small 默认维度

    # Chunking
    chunk_tokens: int = 400               # 每块最大 token 数
    chunk_overlap_tokens: int = 80        # 相邻块重叠 token 数

    # Search
    top_k: int = 5                        # 默认返回结果数
    vector_weight: float = 0.7            # 混合融合向量权重
    bm25_weight: float = 0.3             # 混合融合 BM25 权重

    # Temporal decay（时间衰减）
    temporal_decay_enabled: bool = False
    temporal_decay_half_life_days: float = 30.0

    # MMR 重排
    mmr_enabled: bool = False
    mmr_lambda: float = 0.5              # 相关性 vs 多样性平衡系数
```

**说明**：
- `embedding_provider = "auto"` 时，按顺序探测 OpenAI 兼容 API → 降级纯 BM25
- `embedding_provider = "disabled"` 时，直接使用纯 BM25，不尝试 Embedding
- `MemoryRetrievalTool` 构造时注入 `MemoryConfig`，不依赖全局 singleton

---

## 2. MemoryChunker（文本切割）

位置：`agiwo/tool/builtin/retrieval_tool/chunker.py`

**职责**：将单个 Markdown 文件切割为多个重叠 chunk，输出结构如下：

```python
@dataclass
class MemoryChunk:
    chunk_id: str        # sha256(path + start_line)[:16]
    path: str            # 相对于 workspace 的文件路径
    start_line: int
    end_line: int
    text: str
    content_hash: str    # sha256(text)，用于增量跳过
```

**切割策略**：
- 按 token 数量切割（使用 `tiktoken` 的 `cl100k_base` 编码，与 OpenAI Embedding 对齐）
- 优先在段落边界（空行）切割，避免语义割裂
- 在段落边界找不到合适切点时，回退到句子边界（`. ` / `\n`）
- `chunk_overlap_tokens`：复用前一块末尾 N 个 token，保持上下文连续性

**为何选择 token-based 而非 char-based**：
- Embedding API 有 token 上限（text-embedding-3-small：8192）
- char-based 在中英文混合内容中误差大，token-based 与 API 计费/限制对齐

---

## 3. MemoryEmbedder（向量化）

位置：`agiwo/tool/builtin/retrieval_tool/embedder.py`

**接口**：

```python
class MemoryEmbedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...
    
    @property
    def model_id(self) -> str:
        ...
    
    @property
    def dims(self) -> int:
        ...
```

**唯一实现：`OpenAIEmbedder`**

- 复用项目已有的 HTTP Client 风格（`httpx.AsyncClient`）
- 支持 OpenAI 兼容 API（如 SiliconFlow、DeepSeek、本地 Ollama）
- 批量请求：单次最多 100 个 chunk，超出则分批发送
- 失败时抛出 `EmbeddingError`，由 `MemoryIndexStore` 捕获后降级为纯 BM25

**`EmbedderFactory.create(config)`**：

```
config.embedding_provider == "auto"
  └─ 尝试创建 OpenAIEmbedder（用配置或环境变量中的 api_key）
       ├─ 成功 → 返回 OpenAIEmbedder
       └─ api_key 缺失 → 返回 None（调用方降级 BM25）

config.embedding_provider == "openai"
  └─ 强制创建 OpenAIEmbedder（api_key 缺失则抛出 ConfigError）

config.embedding_provider == "disabled"
  └─ 直接返回 None
```

**Embedding 缓存**（存储在同一 SQLite 中）：

```sql
CREATE TABLE embedding_cache (
    content_hash TEXT NOT NULL,   -- sha256(text)
    model_id     TEXT NOT NULL,
    embedding    BLOB NOT NULL,   -- numpy float32 array，pack 为 bytes
    updated_at   INTEGER NOT NULL,
    PRIMARY KEY (content_hash, model_id)
);
```

- 按 `(content_hash, model_id)` 查询，命中直接返回，不调用 API
- 仅在 chunk 内容变更（`content_hash` 变化）时失效，大幅减少 API 调用

---

## 4. MemoryIndexStore（核心存储 + 索引管理）

位置：`agiwo/tool/builtin/retrieval_tool/store.py`

**职责**：持有 SQLite 连接，协调 Chunker + Embedder，暴露 `search()` 接口。

### 4.1 Schema

```sql
-- 已索引文件记录
CREATE TABLE files (
    path     TEXT PRIMARY KEY,
    hash     TEXT NOT NULL,   -- 文件内容 sha256
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
    model_id   TEXT NOT NULL DEFAULT '',  -- 产生 embedding 的模型，空=BM25-only
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
    start_line UNINDEXED,
    end_line UNINDEXED,
    tokenize = "unicode61"
);

-- Embedding 缓存
CREATE TABLE embedding_cache (
    content_hash TEXT NOT NULL,
    model_id     TEXT NOT NULL,
    embedding    BLOB NOT NULL,
    updated_at   INTEGER NOT NULL,
    PRIMARY KEY (content_hash, model_id)
);
```

### 4.2 初始化与懒加载

```python
class MemoryIndexStore:
    def __init__(self, workspace_dir: Path, config: MemoryConfig):
        self._workspace_dir = workspace_dir
        self._memory_dir = workspace_dir / "MEMORY"
        self._db_path = workspace_dir / "memory.db"
        self._config = config
        self._conn: sqlite3.Connection | None = None
        self._embedder: MemoryEmbedder | None = None
        self._vec_available: bool = False
```

- `_conn` 在第一次 `search()` 时通过 `_ensure_initialized()` 创建，不在构造函数中打开
- 尝试 `conn.load_extension("vec0")` 检测 sqlite-vec 可用性，失败则 `_vec_available = False`
- `_embedder` 通过 `EmbedderFactory.create(config)` 创建，失败则为 `None`（纯 BM25 模式）

### 4.3 增量索引（sync_files）

```
sync_files():
  1. 扫描 MEMORY/*.md（glob，非递归）
  2. 对每个文件，计算 (mtime, size) 与 files 表比对
  3. 变更文件 → index_file(path)
  4. 已删除文件 → remove_file(path)（删除 chunks / fts / vec 中对应行）

index_file(path):
  1. 读取文件内容，计算 sha256 全文 hash
  2. Chunker.chunk(content) → list[MemoryChunk]
  3. 按 content_hash 批量查询 embedding_cache，分离 cache_hit / cache_miss
  4. 对 cache_miss chunks 批量调用 Embedder.embed()，结果写入 embedding_cache
  5. 删除该 path 的旧 chunks（chunks / fts / vec）
  6. 批量 INSERT 新 chunks 到 chunks + fts
  7. 若 _vec_available，批量 INSERT 到 chunks_vec
  8. 更新 files 表（path, hash, mtime, size）
```

**关键设计**：步骤 3-4 是增量核心——同一文本内容无论来自哪个文件、哪次索引，只要 `content_hash` 命中缓存，就跳过 API 调用。对于 Agent 频繁 append 的日志文件，只有新增部分产生 API 费用。

---

## 5. HybridSearcher（混合检索）

位置：`agiwo/tool/builtin/retrieval_tool/searcher.py`

**职责**：给定 query 字符串，返回按相关性排序的 `list[SearchResult]`。

```python
@dataclass
class SearchResult:
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str
    score: float          # 最终融合分数 [0, 1]
    vector_score: float   # 原始向量分数（调试用）
    bm25_score: float     # 原始 BM25 分数（调试用）
```

**检索流程详见 [data-flow.md](./data-flow.md)。**

---

## 6. MemoryRetrievalTool（对外入口）

位置：`agiwo/tool/builtin/retrieval_tool/retrieval.py`

**作为标准 `BaseTool` 实现，对 Agent 暴露两个操作：**

### Tool 参数设计

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Search query to find relevant memories"
    },
    "top_k": {
      "type": "integer",
      "description": "Number of results to return (default: 5)",
      "default": 5
    }
  },
  "required": ["query"]
}
```

**为何去掉 `summarize` 参数**：原 stub 中有 `summarize` 参数（让 LLM 二次总结结果），这增加了不必要的 LLM 调用且职责混乱。Agent 自身可判断是否需要进一步处理检索结果，Tool 只负责检索。

### 执行流程

```python
async def execute(parameters, context, abort_signal):
    # 1. 从 context 解析 workspace_dir
    workspace_dir = resolve_workspace(context, self._config)
    
    # 2. 获取/初始化 MemoryIndexStore（按 workspace_dir 缓存）
    store = await _get_or_create_store(workspace_dir, self._config)
    
    # 3. 增量同步 MEMORY/ 目录
    await store.sync_files()
    
    # 4. 混合检索
    results = await store.search(query, top_k)
    
    # 5. 格式化输出给 LLM
    return ToolResult(content=format_results(results))
```

### Store 缓存

`MemoryRetrievalTool` 内维护 `dict[str, MemoryIndexStore]`，按 `workspace_dir` 缓存 store 实例（避免每次调用重建 SQLite 连接）。由于 Tool 实例在 Agent 生命周期内唯一，此缓存与 Agent 生命周期对齐，无需额外管理。

### 输出格式

```
## Memory Search Results for: "xxx"

### [1] MEMORY/2025-01-15.md (lines 12-25)
Score: 0.87
---
{chunk text}

### [2] MEMORY/project_alpha_2025-01-08.md (lines 3-18)
Score: 0.72
---
{chunk text}
```

包含文件路径和行号，方便 Agent 精确定位后用 `bash_tool` 读取完整文件。

---

## 7. Prompt Runtime 集成

位置：`agiwo/agent/prompt/runtime.py` + `agiwo/agent/prompt/sections.py`

在 `_build_environment_section()` 中注入 Memory Recall 指令：

```
## Memory Recall

Before answering anything about prior work, decisions, dates, or preferences:
call `memory_retrieval` to search MEMORY/ files, then use the returned
file paths + line numbers to read precise sections if needed.
```

同时，`_inject_memories()` 方法在每次 `build()` 时扫描 MEMORY/ 目录，将**最近 N 天**的记忆文件内容直接注入 system prompt 的 `<inject-memories>` 标签中（token 上限由 `AgentOptions.relevant_memory_max_token` 控制）。

**双轨机制**：
- **自动注入**：近期记忆直接在 context 中，Agent 无需显式调用工具即可感知
- **按需检索**：历史记忆通过 `memory_retrieval` 工具精准召回，不污染 context
