# Memory System

The Memory system provides hybrid retrieval (BM25 + vector search) over MEMORY/ files, allowing agents to recall prior knowledge across sessions.

## Design Goals

- **Zero external dependencies**: Uses SQLite with optional sqlite-vec extension — no Qdrant or Weaviate needed
- **Per-agent isolation**: Each agent has its own workspace and index
- **Passive indexing**: No background daemons; indexing happens on-demand during retrieval
- **Graceful degradation**: Falls back to BM25-only when no embedding provider is available

## Architecture

```
Agent
  └─► MemoryRetrievalTool.execute()
        └─► WorkspaceMemoryService.search()
              ├── sync_files()      # Incremental file scanning
              └── search(query)
                    ├── vector_search()  # Embedding + cosine similarity
                    ├── bm25_search()    # FTS5
                    └── merge()          # Weighted fusion + temporal decay + MMR
```

## How It Works

### Writing Memories

Agents write memories directly to MEMORY/ files using the `bash` tool:

```
bash: echo "Important note about X..." >> MEMORY/2026-03-18.md
```

### Retrieving Memories

The agent calls `memory_retrieval(query="...")`:

1. **Sync**: Scan MEMORY/ for new/changed/deleted files
2. **Index**: Chunk new files, compute embeddings (cached by content hash)
3. **Search**: Hybrid BM25 + vector search
4. **Return**: Top-k results with file paths and line numbers

### Hybrid Search

| Component | Weight | Description |
|-----------|--------|-------------|
| Vector | 0.7 | Embedding-based cosine similarity |
| BM25 | 0.3 | Full-text search via FTS5 |

When embedding is unavailable, weights auto-compensate to maintain score range [0, 1].

### Features

- **Temporal decay**: Optionally decay scores by file age (half-life configurable)
- **MMR re-ranking**: Maximal Marginal Relevance for diversity
- **Embedding cache**: Content-hash based cache avoids redundant API calls
- **Query expansion**: Keyword extraction for BM25-only mode

## Configuration

Memory settings are loaded from global settings (`AGIWO_*` env vars or `AgiwoSettings`):

| Setting | Default | Description |
|---------|---------|-------------|
| `AGIWO_MEMORY_TOP_K` | `5` | Number of results to return |
| `AGIWO_MEMORY_CHUNK_TOKENS` | `400` | Chunk size in tokens |
| `AGIWO_MEMORY_CHUNK_OVERLAP_TOKENS` | `80` | Overlap between chunks |
| `AGIWO_MEMORY_RELEVANT_MAX_TOKEN` | `2048` | Max tokens for relevant memories |

Per-tool overrides are available via constructor:

```python
from agiwo.tool.builtin.registry import builtin_tool
from agiwo.tool.builtin.retrieval_tool.tool import MemoryRetrievalTool

tool = MemoryRetrievalTool(
    top_k=10,
    embedding_provider="openai",
)
```

## Detailed Design Documents

For the complete design documentation (originally in Chinese):

- [Components](../memory/components.md) — Core component details
- [Data Flow](../memory/data-flow.md) — Retrieval flow and edge cases
- [Overview](../memory/overview.md) — Architecture overview and comparison with alternatives
