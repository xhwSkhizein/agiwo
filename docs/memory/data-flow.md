# Memory System — 数据流转与检索流程

## 一、写入流程（Agent → MEMORY/）

Memory 写入**不经过任何特殊工具**，Agent 直接使用 `bash_tool` 将内容写入 MEMORY/ 目录：

```
Agent 决定记录某段知识
  └─► bash_tool: echo "..." >> ~/.agiwo/<name>/MEMORY/2025-03-01.md
```

系统不强制记忆格式，但 system prompt 中给出约定：
- 日报式：`<yyyy-mm-dd>.md`
- 分类式：`<category>_<yyyy-mm-dd>.md`（如 `code_review_2025-03-01.md`）

**设计意图**：记忆写入是 Agent 的自主行为，系统只需保证读取侧（检索）的质量。

---

## 二、索引触发时机

索引**在每次 `memory_retrieval` 工具调用时**，于检索前同步触发：

```
memory_retrieval(query="...")
  └─► store.sync_files()     ← 增量扫描 + 索引变更文件
        └─► store.search()   ← 混合检索
```

**增量扫描逻辑**（`sync_files`）：

```
1. glob MEMORY/*.md → current_files: dict[path, (mtime, size)]
2. SELECT path, mtime, size FROM files → indexed_files
3. 对比：
   ├─ new_or_changed = current_files - indexed_files（path 不存在，或 mtime/size 变化）
   └─ deleted = indexed_files - current_files
4. 对 new_or_changed 中每个文件调用 index_file(path)
5. 对 deleted 中每个 path 调用 remove_file(path)
```

**性能说明**：MEMORY/ 通常只有数十个文件，glob + mtime 比对的开销可忽略（< 1ms）。只有实际发生变化的文件才走完整索引流程。

---

## 三、文件索引流程（index_file）

```
index_file(path):

  1. read file content
     file_hash = sha256(content)
     If file_hash == files[path].hash: 内容未变，直接 skip

  2. Chunker.chunk(content, path)
     → list[MemoryChunk]
       每块含：chunk_id, start_line, end_line, text, content_hash

  3. Embedder 可用 → 批量 embed
     a. SELECT content_hash FROM embedding_cache
        WHERE content_hash IN (chunk.content_hash...)
     b. cache_miss chunks → Embedder.embed(texts)
     c. INSERT cache_miss embeddings → embedding_cache

  4. 删除旧索引
     DELETE FROM chunks WHERE path = ?
     DELETE FROM chunks_fts WHERE path = ?
     DELETE FROM chunks_vec WHERE chunk_id IN (旧 chunk_ids)

  5. 批量写入新索引
     INSERT INTO chunks (...)
     INSERT INTO chunks_fts (...)
     若 vec_available: INSERT INTO chunks_vec (chunk_id, embedding)

  6. UPDATE files SET hash=?, mtime=?, size=? WHERE path=?
```

**Embedding 缓存的作用**：同一文本内容（`content_hash` 相同）无论来自哪个文件，只要命中缓存就跳过 API 调用。对于 Agent 频繁 append 的日志文件，只有新增部分产生 API 费用。

---

## 四、混合检索流程（HybridSearcher.search）

### 4.1 完整流程图

```
query: str
  │
  ├──[Embedder 可用]──────────────────────────────────────────┐
  │                                                           │
  │  embed(query) → query_vec                                 │
  │                                                           │
  │  [vec_available] sqlite-vec 扩展存在                      │
  │    SELECT chunk_id, vec_distance_cosine(embedding, ?)     │
  │    FROM chunks_vec ORDER BY distance LIMIT top_k * 5     │
  │    → vector_results: dict[chunk_id, v_score]             │
  │                                                           │
  │  [vec NOT available] 回退内存计算                         │
  │    全量 SELECT chunk_id, embedding FROM chunks           │
  │    内存计算 cosine_similarity(query_vec, row_vec)         │
  │    → vector_results: dict[chunk_id, v_score]             │
  │                                                           │
  └───────────────────────────────────────────────────────────┤
  │                                                           │
  ├──[BM25 / FTS5]────────────────────────────────────────────┤
  │                                                           │
  │  [有 Embedder] 直接 FTS5 MATCH 原始 query                │
  │    SELECT chunk_id, bm25(chunks_fts) AS rank             │
  │    FROM chunks_fts WHERE text MATCH ?                    │
  │    ORDER BY rank LIMIT top_k * 5                        │
  │                                                           │
  │  [无 Embedder] Query Expansion 模式                      │
  │    extract_keywords(query) → keywords list               │
  │    对每个 keyword 分别 FTS MATCH，结果 UNION 去重         │
  │                                                           │
  │  bm25_raw → normalize: score = 1 / (1 + abs(bm25_raw))  │
  │                                                           │
  └───────────────────────────────────────────────────────────┤
                                                              │
  Hybrid Merge ◄──────────────────────────────────────────── ┘

  1. Union chunk_ids from both sides
  2. final_score = vector_weight * v_score + bm25_weight * bm25_score
     缺失某侧时，权重自动补偿到另一侧（保证 score ∈ [0,1]）

  3. [temporal_decay_enabled]
     date = extract_date_from_path(path)   # regex \d{4}-\d{2}-\d{2}
     age_days = (today - date).days
     λ = ln(2) / half_life_days
     final_score *= e^(−λ × age_days)
     无日期文件（如 MEMORY.md）视为常青知识，不参与衰减

  4. Sort by final_score DESC, take top_k * 2

  5. [mmr_enabled] MMR 重排
     selected = []
     while len(selected) < top_k and candidates remain:
       mmr[c] = λ * c.score
                - (1-λ) * max(jaccard(c.text, s.text) for s in selected)
       selected.append(argmax(mmr))

  6. Return top_k SearchResult
```

### 4.2 分数归一化

| 原始分数 | 归一化公式 | 值域 |
|---------|-----------|------|
| 余弦距离（sqlite-vec 返回距离值） | `score = 1 - distance` | [0, 1] |
| BM25 rank（FTS5 bm25()，值为负） | `score = 1 / (1 + abs(rank))` | (0, 1] |

### 4.3 权重补偿规则

```python
v_score  = vector_results.get(chunk_id, 0.0)
bm_score = bm25_results.get(chunk_id, 0.0)

has_vec = v_score > 0
has_bm  = bm_score > 0

if has_vec and has_bm:
    final = config.vector_weight * v_score + config.bm25_weight * bm_score
elif has_vec:
    final = v_score   # 仅向量，满权重
else:
    final = bm_score  # 仅 BM25，满权重
```

这样即便在降级模式下分数范围仍保持 [0, 1]，不需要重新 calibrate。

---

## 五、Query Expansion（FTS-Only 降级模式）

当无 Embedding Provider 时，单一短语 query 的 FTS5 召回率较差。系统对 query 做轻量关键词扩展：

```python
STOP_WORDS_ZH = {"的", "了", "是", "在", "和", "与", "或", "也", "都", "很", ...}
STOP_WORDS_EN = {"the", "a", "an", "is", "are", "was", "were", "in", "of", ...}

def extract_keywords(query: str) -> list[str]:
    # 中文：Unicode range 切字（连续汉字作为一个词保留）
    # 英文：按空格切词
    # 过滤停用词 + 长度 < 2
    # 去重，保持原序
    return keywords
```

多个关键词分别发起 FTS 查询再合并，等价于 OR 语义扩展，大幅提升 BM25-only 模式的召回率。

---

## 六、自动注入（inject-memories）

除了主动调用 `memory_retrieval` 工具，系统还在每次构建 system prompt 时自动注入近期记忆：

### 注入策略

```python
def _inject_memories(self) -> str:
    memory_dir = Path(f"{workspace}/MEMORY")
    if not memory_dir.exists():
        return ""

    # 按 mtime 降序取最新文件，直到 token 总量逼近上限
    files = sorted(memory_dir.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
    chunks, total_tokens = [], 0

    for f in files:
        content = f.read_text()
        tokens = count_tokens(content)
        if total_tokens + tokens > options.relevant_memory_max_token:
            break
        chunks.append(f"### {f.name}\n{content}")
        total_tokens += tokens

    if not chunks:
        return ""
    return "<inject-memories>\n" + "\n\n".join(chunks) + "\n</inject-memories>"
```

### 双轨机制

| 机制 | 触发方式 | 适用场景 |
|------|---------|---------|
| 自动注入 | 每次构建 system prompt | 近期记忆（最近几天），Agent 无需显式操作 |
| 按需检索 | Agent 调用 `memory_retrieval` | 历史记忆（超出 token 窗口），精准语义匹配 |

**Token 预算**：`AgentOptions.relevant_memory_max_token`（默认 2048），控制自动注入的最大上限，防止记忆内容占用过多上下文。

---

## 七、端到端示例

**场景**：Agent 在 2025-01-15 记录了一段关于某项目架构决策的笔记，2025-03-01 用户询问相关决策。

```
Step 1: system_prompt 构建
  _inject_memories():
    MEMORY/ 目录有 30 个文件，最新 5 个在 token 预算内
    2025-01-15.md 较旧，未被注入
  → system prompt 中只有近期记忆

Step 2: 用户提问 "当时为什么选择 SQLite 而不是 PostgreSQL？"
  LLM 在 context 中找不到相关信息

Step 3: LLM 决定调用 memory_retrieval
  memory_retrieval(query="SQLite PostgreSQL 架构决策")

Step 4: sync_files()
  扫描 MEMORY/，发现 2025-01-15.md 已索引，无变化，跳过

Step 5: HybridSearcher.search("SQLite PostgreSQL 架构决策", top_k=5)
  vector_search: embed query → cosine similarity → 找到 2025-01-15.md lines 5-18 (score=0.91)
  bm25_search:   FTS MATCH "SQLite PostgreSQL" → 同一 chunk (score=0.73)
  fusion: 0.7 * 0.91 + 0.3 * 0.73 = 0.856
  temporal decay: age=45 days, λ=0.023, decay=0.35 → final=0.30
  （注：时间衰减默认关闭，此处为演示）

Step 6: 返回结果
  [1] MEMORY/2025-01-15.md (lines 5-18) score=0.856
  --- 选择 SQLite 的理由是... ---

Step 7: LLM 基于检索结果回答用户问题
```

---

## 八、边界情况处理

| 情况 | 处理方式 |
|------|---------|
| MEMORY/ 目录不存在 | `sync_files()` 直接返回，`search()` 返回空列表，Tool 返回 "No memories found" |
| sqlite-vec 扩展不可用 | `_vec_available = False`，回退内存计算余弦相似度 |
| Embedding API 不可达 | 捕获 `EmbeddingError`，降级为纯 BM25 模式，工具正常返回结果 |
| FTS query 包含特殊字符 | `sanitize_fts_query()` 转义 FTS5 运算符（`"`, `*`, `(`, `)` 等） |
| query 在 FTS 中无匹配 | bm25_results 为空，最终用 vector_results（若可用）或返回空列表 |
| chunk embedding 缓存过期 | 缓存按 `(content_hash, model_id)` 索引，换模型时旧缓存自动失效 |
| 文件被删除 | `remove_file()` 清理 chunks / fts / vec 中对应行，files 表删除记录 |
