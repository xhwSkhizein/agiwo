const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8422";

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

// ── Sessions / Runs ────────────────────────────────────────────────────

export interface SessionSummary {
  session_id: string;
  agent_id: string | null;
  last_user_input: string | null;
  last_response: string | null;
  run_count: number;
  step_count: number;
  created_at: string | null;
  updated_at: string | null;
}

export interface RunResponse {
  id: string;
  agent_id: string;
  session_id: string;
  user_id: string | null;
  user_input: unknown;
  status: string;
  response_content: string | null;
  metrics: Record<string, unknown> | null;
  created_at: string | null;
  updated_at: string | null;
  parent_run_id: string | null;
}

export interface StepResponse {
  id: string;
  session_id: string;
  run_id: string;
  sequence: number;
  role: string;
  agent_id: string | null;
  content: unknown;
  content_for_user: string | null;
  reasoning_content: string | null;
  tool_calls: Record<string, unknown>[] | null;
  tool_call_id: string | null;
  name: string | null;
  metrics: Record<string, unknown> | null;
  created_at: string | null;
  parent_run_id: string | null;
  depth: number;
}

export function listSessions(limit = 20, offset = 0) {
  return fetchJSON<SessionSummary[]>(`/api/sessions?limit=${limit}&offset=${offset}`);
}

export function listRuns(params?: { session_id?: string; limit?: number; offset?: number }) {
  const q = new URLSearchParams();
  if (params?.session_id) q.set("session_id", params.session_id);
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.offset) q.set("offset", String(params.offset));
  return fetchJSON<RunResponse[]>(`/api/runs?${q}`);
}

export function getRun(runId: string) {
  return fetchJSON<RunResponse>(`/api/runs/${runId}`);
}

export function getSessionSteps(sessionId: string) {
  return fetchJSON<StepResponse[]>(`/api/sessions/${sessionId}/steps`);
}

// ── Traces ─────────────────────────────────────────────────────────────

export interface TraceListItem {
  trace_id: string;
  agent_id: string | null;
  session_id: string | null;
  user_id: string | null;
  start_time: string | null;
  duration_ms: number | null;
  status: string;
  total_tokens: number;
  total_llm_calls: number;
  total_tool_calls: number;
  input_query: string | null;
  final_output: string | null;
}

export interface SpanResponse {
  span_id: string;
  trace_id: string;
  parent_span_id: string | null;
  kind: string;
  name: string;
  start_time: string | null;
  end_time: string | null;
  duration_ms: number | null;
  status: string;
  error_message: string | null;
  depth: number;
  attributes: Record<string, unknown>;
  input_preview: string | null;
  output_preview: string | null;
  metrics: Record<string, unknown>;
  llm_details: Record<string, unknown> | null;
  tool_details: Record<string, unknown> | null;
  run_id: string | null;
  step_id: string | null;
}

export interface TraceDetail {
  trace_id: string;
  agent_id: string | null;
  session_id: string | null;
  user_id: string | null;
  start_time: string | null;
  end_time: string | null;
  duration_ms: number | null;
  status: string;
  root_span_id: string | null;
  total_tokens: number;
  total_llm_calls: number;
  total_tool_calls: number;
  total_cache_read_tokens: number;
  total_cache_creation_tokens: number;
  max_depth: number;
  input_query: string | null;
  final_output: string | null;
  spans: SpanResponse[];
}

export function listTraces(params?: { agent_id?: string; session_id?: string; limit?: number; offset?: number }) {
  const q = new URLSearchParams();
  if (params?.agent_id) q.set("agent_id", params.agent_id);
  if (params?.session_id) q.set("session_id", params.session_id);
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.offset) q.set("offset", String(params.offset));
  return fetchJSON<TraceListItem[]>(`/api/traces?${q}`);
}

export function getTrace(traceId: string) {
  return fetchJSON<TraceDetail>(`/api/traces/${traceId}`);
}

// ── Agents ─────────────────────────────────────────────────────────────

export interface AgentConfig {
  id: string;
  name: string;
  description: string;
  model_provider: string;
  model_name: string;
  system_prompt: string;
  tools: string[];
  options: Record<string, unknown>;
  model_params: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface AgentConfigCreate {
  name: string;
  description?: string;
  model_provider: string;
  model_name: string;
  system_prompt?: string;
  tools?: string[];
  options?: Record<string, unknown>;
  model_params?: Record<string, unknown>;
}

export interface AvailableTool {
  name: string;
  description: string;
  type: "builtin" | "agent";
  agent_name?: string;
}

export function listAvailableTools(exclude?: string) {
  const q = exclude ? `?exclude=${encodeURIComponent(exclude)}` : "";
  return fetchJSON<AvailableTool[]>(`/api/agents/tools/available${q}`);
}

export function listAgents() {
  return fetchJSON<AgentConfig[]>("/api/agents");
}

export function getAgent(agentId: string) {
  return fetchJSON<AgentConfig>(`/api/agents/${agentId}`);
}

export function createAgent(data: AgentConfigCreate) {
  return fetchJSON<AgentConfig>("/api/agents", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export function updateAgent(agentId: string, data: Partial<AgentConfigCreate>) {
  return fetchJSON<AgentConfig>(`/api/agents/${agentId}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

export async function deleteAgent(agentId: string) {
  const res = await fetch(`${API_BASE}/api/agents/${agentId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Delete failed: ${res.status}`);
}

// ── Scheduler ─────────────────────────────────────────────────────────

export interface WakeConditionResponse {
  type: string;
  time_value: number | null;
  time_unit: string | null;
  total_children: number;
  completed_children: number;
  wakeup_at: string | null;
}

export interface AgentStateListItem {
  id: string;
  agent_id: string;
  status: string;
  task: string;
  parent_agent_id: string;
  parent_state_id: string | null;
  wake_condition: WakeConditionResponse | null;
  result_summary: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface AgentStateDetail {
  id: string;
  session_id: string;
  agent_id: string;
  parent_agent_id: string;
  parent_state_id: string | null;
  status: string;
  task: string;
  config_overrides: Record<string, unknown>;
  wake_condition: WakeConditionResponse | null;
  result_summary: string | null;
  signal_propagated: boolean;
  created_at: string | null;
  updated_at: string | null;
}

export interface SchedulerStats {
  total: number;
  pending: number;
  running: number;
  sleeping: number;
  completed: number;
  failed: number;
}

export function listAgentStates(params?: { status?: string; limit?: number; offset?: number }) {
  const q = new URLSearchParams();
  if (params?.status) q.set("status", params.status);
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.offset) q.set("offset", String(params.offset));
  return fetchJSON<AgentStateListItem[]>(`/api/scheduler/states?${q}`);
}

export function getAgentState(stateId: string) {
  return fetchJSON<AgentStateDetail>(`/api/scheduler/states/${stateId}`);
}

export function getAgentStateChildren(stateId: string) {
  return fetchJSON<AgentStateListItem[]>(`/api/scheduler/states/${stateId}/children`);
}

export function getSchedulerStats() {
  return fetchJSON<SchedulerStats>(`/api/scheduler/stats`);
}

// ── Chat ───────────────────────────────────────────────────────────────

export function chatStreamUrl(agentId: string) {
  return `${API_BASE}/api/chat/${agentId}`;
}

// ── Scheduler Chat ────────────────────────────────────────────────────

export function schedulerChatStreamUrl(agentId: string) {
  return `${API_BASE}/api/scheduler/chat/${agentId}`;
}

export function cancelSchedulerChat(agentId: string, stateId: string) {
  return fetchJSON<{ ok: boolean; state_id: string }>(
    `/api/scheduler/chat/${agentId}/cancel`,
    {
      method: "POST",
      body: JSON.stringify({ state_id: stateId }),
    }
  );
}
