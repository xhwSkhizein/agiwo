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
        └─► MemoryIndexStore
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

```python
from agiwo.tool.builtin.config import MemoryConfig

config = MemoryConfig(
    embedding_provider="auto",       # "openai" | "auto" | "disabled"
    embedding_model="text-embedding-3-small",
    chunk_tokens=400,
    chunk_overlap_tokens=80,
    top_k=5,
    vector_weight=0.7,
    bm25_weight=0.3,
    temporal_decay_enabled=False,
    temporal_decay_half_life_days=30.0,
    mmr_enabled=False,
    mmr_lambda=0.5,
)
```

## Detailed Design Documents

For the complete design documentation (originally in Chinese):

- [Components](./memory/components.md) — Core component details
- [Data Flow](./memory/data-flow.md) — Retrieval flow and edge cases
- [Overview](./memory/overview.md) — Architecture overview and comparison with alternatives
