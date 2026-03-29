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
    let detail = res.statusText;
    try {
      const errorBody = await res.json();
      if (typeof errorBody?.detail === "string") {
        detail = errorBody.detail;
      }
    } catch {
      // Ignore non-JSON error bodies and keep the HTTP status text.
    }
    throw new Error(`API error: ${res.status} ${detail}`);
  }
  return res.json();
}

// ── UserInput Types ────────────────────────────────────────────────────

export interface ContentPartPayload {
  type: string;
  text?: string;
  url?: string;
  mime_type?: string;
  detail?: string;
  metadata?: Record<string, unknown>;
}

export interface ChannelContextPayload {
  source: string;
  metadata?: Record<string, unknown>;
}

export interface UserMessage {
  __type: "user_message";
  content: ContentPartPayload[];
  context?: ChannelContextPayload | null;
}

export interface ContentParts {
  __type: "content_parts";
  parts: ContentPartPayload[];
}

export type UserInput =
  | string
  | ContentPartPayload[]
  | UserMessage
  | ContentParts;

// ── Sessions / Runs ────────────────────────────────────────────────────

export interface RunMetricsSummary {
  run_count: number;
  completed_run_count: number;
  step_count: number;
  tool_calls_count: number;
  duration_ms: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cache_read_tokens: number;
  cache_creation_tokens: number;
  token_cost: number;
}

export type UsageSource = "provider" | "estimated" | "mixed";

export interface RunMetricsPayload {
  duration_ms?: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  total_tokens?: number | null;
  cache_read_tokens?: number | null;
  cache_creation_tokens?: number | null;
  token_cost?: number | null;
  steps_count?: number | null;
  tool_calls_count?: number | null;
}

export interface StepMetricsPayload extends RunMetricsPayload {
  usage_source?: UsageSource | null;
  model_name?: string | null;
  provider?: string | null;
  first_token_latency_ms?: number | null;
}

export interface SpanMetricsPayload {
  "tokens.input"?: number | null;
  "tokens.output"?: number | null;
  "tokens.total"?: number | null;
  "tokens.cache_read"?: number | null;
  "tokens.cache_creation"?: number | null;
  token_cost?: number | null;
  first_token_ms?: number | null;
  duration_ms?: number | null;
  model?: string | null;
  provider?: string | null;
  usage_source?: UsageSource | null;
}

export interface ToolFunctionPayload {
  name: string;
  arguments: string;
}

export interface ToolCallPayload {
  id?: string;
  index?: number;
  type?: string;
  function?: ToolFunctionPayload;
}

export interface SessionSummary {
  session_id: string;
  agent_id: string | null;
  last_user_input: UserInput | null;  // 结构化 UserInput
  last_response: string | null;
  run_count: number;
  step_count: number;
  metrics: RunMetricsSummary;
  created_at: string | null;
  updated_at: string | null;
}

export interface RunResponse {
  id: string;
  agent_id: string;
  session_id: string;
  user_id: string | null;
  user_input: UserInput;
  status: string;
  response_content: string | null;
  metrics: RunMetricsPayload | null;
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
  user_input: UserInput | null;
  tool_calls: ToolCallPayload[] | null;
  tool_call_id: string | null;
  name: string | null;
  metrics: StepMetricsPayload | null;
  created_at: string | null;
  parent_run_id: string | null;
  depth: number;
}

export interface StepDeltaPayload {
  content?: string | null;
  reasoning_content?: string | null;
  tool_calls?: ToolCallPayload[] | null;
  usage?: Record<string, number> | null;
}

export interface StreamEventBase {
  type: string;
  session_id: string;
  run_id: string;
  agent_id: string;
  parent_run_id: string | null;
  depth: number;
  timestamp?: string | null;
}

export interface RunStartedEventPayload extends StreamEventBase {
  type: "run_started";
}

export interface StepDeltaEventPayload extends StreamEventBase {
  type: "step_delta";
  step_id: string;
  delta: StepDeltaPayload;
}

export interface StepCompletedEventPayload extends StreamEventBase {
  type: "step_completed";
  step: StepResponse;
}

export interface RunCompletedEventPayload extends StreamEventBase {
  type: "run_completed";
  response?: string | null;
  metrics?: RunMetricsPayload | null;
  termination_reason?: string | null;
}

export interface RunFailedEventPayload extends StreamEventBase {
  type: "run_failed";
  error: string;
}

export interface SchedulerFailedEventPayload {
  type: "scheduler_failed";
  error: string;
}

export type AgentStreamEventPayload =
  | RunStartedEventPayload
  | StepDeltaEventPayload
  | StepCompletedEventPayload
  | RunCompletedEventPayload
  | RunFailedEventPayload;

export type StreamEventPayload =
  | AgentStreamEventPayload
  | SchedulerFailedEventPayload;

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

export function getSessionSummary(sessionId: string) {
  return fetchJSON<SessionSummary>(`/api/sessions/${sessionId}/summary`);
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
  total_input_tokens: number;
  total_output_tokens: number;
  total_cache_read_tokens: number;
  total_cache_creation_tokens: number;
  total_token_cost: number;
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
  metrics: SpanMetricsPayload;
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
  total_input_tokens: number;
  total_output_tokens: number;
  total_token_cost: number;
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

export interface AgentOptionsPayload {
  config_root: string;
  max_steps: number;
  run_timeout: number;
  max_input_tokens_per_call: number | null;
  max_run_cost: number | null;
  enable_termination_summary: boolean;
  termination_summary_prompt: string;
  enable_skill: boolean;
  skills_dirs: string[] | null;
  relevant_memory_max_token: number;
  stream_cleanup_timeout: number;
  compact_prompt: string;
}

export interface ModelParamsPayload {
  base_url: string | null;
  api_key_env_name: string | null;
  max_output_tokens: number;
  max_context_window: number;
  temperature: number;
  top_p: number;
  frequency_penalty: number;
  presence_penalty: number;
  cache_hit_price: number;
  input_price: number;
  output_price: number;
}

export interface AgentConfig {
  id: string;
  name: string;
  description: string;
  model_provider: string;
  model_name: string;
  system_prompt: string;
  tools: string[];
  options: AgentOptionsPayload;
  model_params: ModelParamsPayload;
  created_at: string;
  updated_at: string;
}

export interface AgentConfigCreate {
  name: string;
  description: string;
  model_provider: string;
  model_name: string;
  system_prompt: string;
  tools: string[];
  options: AgentOptionsPayload;
  model_params: ModelParamsPayload;
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

export function updateAgent(agentId: string, data: AgentConfigCreate) {
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
  wait_for: string[];
  wait_mode: string;
  completed_ids: string[];
  time_value: number | null;
  time_unit: string | null;
  wakeup_at: string | null;
  timeout_at: string | null;
}

export interface AgentStateListItem {
  id: string;
  status: string;
  task: UserInput;
  parent_id: string | null;
  wake_condition: WakeConditionResponse | null;
  result_summary: string | null;
  agent_config_id: string | null;
  is_persistent: boolean;
  depth: number;
  wake_count: number;
  metrics: RunMetricsSummary;
  created_at: string | null;
  updated_at: string | null;
}

export interface AgentStateDetail {
  id: string;
  session_id: string;
  status: string;
  task: UserInput;
  parent_id: string | null;
  pending_input: UserInput | null;
  config_overrides: Record<string, unknown>;
  wake_condition: WakeConditionResponse | null;
  result_summary: string | null;
  signal_propagated: boolean;
  agent_config_id: string | null;
  is_persistent: boolean;
  depth: number;
  wake_count: number;
  metrics: RunMetricsSummary;
  created_at: string | null;
  updated_at: string | null;
}

export interface SchedulerStats {
  total: number;
  pending: number;
  running: number;
  waiting: number;
  idle: number;
  queued: number;
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

// ── Chat Sessions ──────────────────────────────────────────────────────

export interface ChatSessionItem {
  session_id: string;
  run_count: number;
  last_input: string | null;
  last_response: string | null;
  updated_at: string | null;
  current_task_id?: string | null;
  task_message_count?: number;
  source_session_id?: string | null;
  fork_context_summary?: string | null;
}

export interface SessionActionResult {
  session_id: string;
  task_id: string | null;
  source_session_id: string | null;
  previous_session_id?: string | null;
}

export function listAgentChatSessions(agentId: string) {
  return fetchJSON<ChatSessionItem[]>(`/api/chat/${agentId}/sessions`);
}

export function createAgentSession(agentId: string, scopeId: string) {
  return fetchJSON<SessionActionResult>(`/api/chat/${agentId}/sessions/create`, {
    method: "POST",
    body: JSON.stringify({
      chat_context_scope_id: scopeId,
      channel_instance_id: "console-web",
      user_open_id: "console-user",
    }),
  });
}

export function switchAgentSession(
  agentId: string,
  scopeId: string,
  targetSessionId: string,
) {
  return fetchJSON<SessionActionResult>(`/api/chat/${agentId}/sessions/switch`, {
    method: "POST",
    body: JSON.stringify({
      chat_context_scope_id: scopeId,
      target_session_id: targetSessionId,
    }),
  });
}

export function forkAgentSession(
  agentId: string,
  sessionId: string,
  contextSummary: string,
) {
  return fetchJSON<SessionActionResult>(
    `/api/chat/${agentId}/sessions/${sessionId}/fork`,
    {
      method: "POST",
      body: JSON.stringify({ context_summary: contextSummary }),
    },
  );
}

// ── Chat ───────────────────────────────────────────────────────────────

export function chatStreamUrl(agentId: string) {
  return `${API_BASE}/api/chat/${agentId}`;
}

// ── Scheduler Chat ────────────────────────────────────────────────────

export function schedulerChatStreamUrl(agentId: string) {
  return `${API_BASE}/api/scheduler/chat/${agentId}`;
}

export function parseStreamEventPayload(data: string): StreamEventPayload | null {
  try {
    const parsed = JSON.parse(data) as StreamEventPayload;
    if (!parsed || typeof parsed !== "object" || typeof parsed.type !== "string") {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
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

// ── Scheduler Control ─────────────────────────────────────────────────

export interface PendingEventItem {
  id: string;
  target_agent_id: string;
  source_agent_id: string | null;
  event_type: string;
  payload: Record<string, unknown>;
  created_at: string | null;
}

export function steerAgent(id: string, message: string, urgent = false) {
  return fetchJSON<{ ok: boolean }>(`/api/scheduler/states/${id}/steer`, {
    method: "POST",
    body: JSON.stringify({ message, urgent }),
  });
}

export function cancelAgent(id: string, reason?: string) {
  return fetchJSON<{ ok: boolean }>(`/api/scheduler/states/${id}/cancel`, {
    method: "POST",
    body: JSON.stringify({ reason }),
  });
}

export function resumeAgent(id: string, message: string) {
  return fetchJSON<{ ok: boolean }>(`/api/scheduler/states/${id}/resume`, {
    method: "POST",
    body: JSON.stringify({ message }),
  });
}

export function getPendingEvents(id: string) {
  return fetchJSON<PendingEventItem[]>(`/api/scheduler/states/${id}/pending-events`);
}

export function createPersistentAgent(data: { agent_config_id?: string; initial_task?: string; session_id?: string }) {
  return fetchJSON<{ ok: boolean; state_id: string }>(`/api/scheduler/states/create`, {
    method: "POST",
    body: JSON.stringify(data),
  });
}
