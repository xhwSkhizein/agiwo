# Memory System — 架构总览

## 设计目标

为 Agiwo Agent 提供一套**轻量、无外部依赖**的持久化记忆检索系统，让 Agent 能将重要知识写入 MEMORY/ 目录下的 Markdown 文件，并在后续对话中通过混合检索（向量 + BM25）精准召回。

核心约束：
- **零外部服务依赖**：全部基于 SQLite（含 sqlite-vec 扩展），无需 Qdrant / Weaviate 等向量数据库
- **按 Agent 隔离**：每个 `agent_name` 对应独立的 `.agiwo/<agent_name>/` workspace，索引库也独立存储
- **被动索引**：不引入后台守护进程；索引在工具调用时按需触发（增量、懒加载）
- **Embedding 可降级**：若无可用 Embedding Provider，自动退化为纯 BM25 检索，功能不中断
- **与现有架构零耦合**：`MemoryRetrievalTool` 是标准 `BaseTool`，`MemoryIndexStore` 独立于 Agent 生命周期

---

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                      Agent Runtime                       │
│                                                          │
│  SystemPromptBuilder ──► inject <inject-memories> tag   │
│                                                          │
│  ExecutionEngine ────► MemoryRetrievalTool.execute()    │
│                                   │                      │
└───────────────────────────────────┼──────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │         MemoryIndexStore        │
                    │  (per agent_name, lazy init)    │
                    │                                 │
                    │  ┌──────────┐  ┌─────────────┐ │
                    │  │  Indexer │  │   Searcher  │ │
                    │  └────┬─────┘  └──────┬──────┘ │
                    └───────┼───────────────┼─────────┘
                            │               │
                    ┌───────▼───────────────▼─────────┐
                    │          SQLite Database          │
                    │  .agiwo/<agent_name>/memory.db   │
                    │                                   │
                    │  ┌────────┐  ┌───────┐           │
                    │  │ chunks │  │  fts  │ (FTS5)    │
                    │  ├────────┤  ├───────┤           │
                    │  │  vec  │  │ files │           │
                    │  │(vec0) │  │       │           │
                    │  └────────┘  └───────┘           │
                    └───────────────────────────────────┘
                                    ▲
                    ┌───────────────┴───────────────┐
                    │         MEMORY/ Directory       │
                    │  .agiwo/<agent_name>/MEMORY/    │
                    │                                 │
                    │  2025-01-15.md                  │
                    │  code_review_2025-01-10.md      │
                    │  project_alpha_2025-01-08.md    │
                    └─────────────────────────────────┘
```

---

## 与参考实现（OpenClaw）的对比与取舍

| 维度 | OpenClaw | Agiwo（本方案） |
|------|----------|----------------|
| 索引触发 | 文件 Watcher（chokidar）+ 定时器 | 工具调用时按需增量触发 |
| 记忆来源 | memory 文件 + sessions 会话记录 | 仅 memory 文件（KISS） |
| Embedding Provider | openai / gemini / voyage / local | openai 兼容 API / 降级纯 BM25 |
| 向量存储 | sqlite-vec 扩展 | sqlite-vec 扩展（不可用时内存计算） |
| 管理器层次 | 3 层继承（Sync + Embedding + Index） | 单类 `MemoryIndexStore`（组合优于继承） |
| 时间衰减 | 可配置指数衰减 | 可配置指数衰减（保留） |
| MMR 重排 | 可选 | 可选（保留） |
| Query Expansion | FTS-Only 时关键词扩展 | 保留（停用词过滤 + 多词 OR 查询） |
| 配置体系 | resolveMemorySearchConfig() 全局 | `MemoryConfig` dataclass，注入 Tool |

**关键简化**：去掉后台 Watcher 和会话同步，改为在 `memory_retrieval` 工具调用入口做增量索引检查。这使系统完全无状态守护进程，符合 Agiwo 的 "no background daemon" 原则。

---

## 文件组织

```
agiwo/tool/builtin/retrieval_tool/
├── __init__.py
├── store.py          # MemoryIndexStore：SQLite 索引 + 检索核心
├── chunker.py        # Markdown chunk 切割
├── embedder.py       # Embedding Provider 抽象 + OpenAI 实现
├── searcher.py       # HybridSearcher：BM25 + Vector + 融合 + MMR
└── retrieval.py      # MemoryRetrievalTool（BaseTool 实现，对外入口）

agiwo/tool/builtin/config.py
└── MemoryConfig      # 新增配置 dataclass
```

---

## 数据流简图

```
Agent 写记忆
  └─► bash_tool 写入 MEMORY/xxx.md

Agent 查记忆
  └─► memory_retrieval(query="...")
        └─► MemoryIndexStore.search(query)
              ├─ sync_files()          # 增量扫描 MEMORY/，索引变更文件
              ├─ HybridSearcher.search()
              │    ├─ vector_search()  # Embedding + cosine
              │    ├─ bm25_search()    # FTS5
              │    └─ merge()          # 加权融合 + 时间衰减 + MMR
              └─ 返回 top-k chunks，含文件路径 + 行号
```

---

## 子文档索引

- **[components.md](./components.md)** — 核心组件详解（Store / Chunker / Embedder / Searcher）
- **[data-flow.md](./data-flow.md)** — 数据流转、检索流程与关键实现细节
