# Deferred Review TODOs


发现了一处额外的不协调点：Web Chat 和 Feishu 在 Console 里对 runtime agent 的持有方式也不完全一致
Owner: @platform-runtime
Tracking: https://github.com/xhwSkhizein/agiwo/issues/45
Target behavior: Both Web Chat and Feishu must obtain and release the runtime agent via a shared RuntimeAgentManager API with consistent lifecycle semantics.
Acceptance criteria:
1. Both integrations use `RuntimeAgentManager.getAgent()` and `RuntimeAgentManager.releaseAgent()`.
2. Unit/integration tests validate identical runtime-agent acquire/rebind/release flows for Web Chat and Feishu.
3. Staging or deployment logs show no duplicate runtime-agent creation for the same session lifecycle.
