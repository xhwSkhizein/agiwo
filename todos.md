# Deferred Review TODOs

1. Agent 构造函数：disabled_sdk_tool_names 参数应该在 AgentConfig 中，而不是构造函数上
2. run_child 的 child_id 未被使用？
3. run_identity.py / run_ledger.py 都是只有一个 model 类这些应该放在一起 types.py ？ 或某些子 package 下的 types.py ? AGETNS.md 中统一一下风格？所有的数据模型定义放在哪个文件（types.py ?） 类似的还有 memory_types.py、 compact_types.py； config.py 下也是数据模型定义，直接叫 config.py 歧义非常大！
4. child.py 里放的是 AgentTool ？ 这个命名很奇怪
5. run_loop.py 中包含了 summarize 的 prompt 和逻辑，这里应和 compact 一样单独一个文件甚至一个小 package
6. run_mutations.py 下又仅有几个帮助方法
7. agent/serialization.py 的这个代码感觉更像是下游 console/server 的逻辑污染到了 sdk 层，甚至直接是最核心的 agent 层，这个应该放到 console/server 这种下游具体实现中的东西
8. state_tracking.py 只有几个帮助方法，最终只有一处位置真实在调用
9. step_pipeline.py 只有一个方法，单独一个文件感觉也很奇怪
10. 整体代码组织有些不太合理，python文件 / module 结构非常混乱， 需要重新组织

---

compaction 的 transcripts 目录没有遵循 root_path的逻辑，不是放在 root_path 下的，正确的目录构建逻辑应该是 `root_path / compaction / transcripts / agent / session / xxx.jsonl`