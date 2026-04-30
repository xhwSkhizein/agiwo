const API_BASE = process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") ?? "";
const DEFAULT_FETCH_TIMEOUT_MS = 30_000;

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(`API error: ${status} ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = globalThis.setTimeout(() => {
    controller.abort();
  }, DEFAULT_FETCH_TIMEOUT_MS);
  let res: Response;
  try {
    res = await fetch(apiUrl(path), {
      ...init,
      signal: init?.signal ?? controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...init?.headers,
      },
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(`Request timed out after ${DEFAULT_FETCH_TIMEOUT_MS / 1000}s`);
    }
    throw err;
  } finally {
    globalThis.clearTimeout(timeout);
  }
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
    throw new ApiError(res.status, detail);
  }
  return res.json();
}

export interface PageResponse<T> {
  items: T[];
  limit: number;
  offset: number;
  has_more: boolean;
  total: number | null;
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
  last_user_input: UserInput | null;
  last_response: string | null;
  run_count: number;
  step_count: number;
  metrics: RunMetricsSummary;
  created_at: string | null;
  updated_at: string | null;
  chat_context_scope_id: string | null;
  created_by: string | null;
  base_agent_id: string | null;
  root_state_status: string | null;
  source_session_id: string | null;
  fork_context_summary: string | null;
}

export interface SessionRecord {
  id: string;
  chat_context_scope_id: string | null;
  base_agent_id: string;
  created_by: string;
  created_at: string;
  updated_at: string;
  source_session_id: string | null;
  fork_context_summary: string | null;
}

export interface ChatContextRecord {
  scope_id: string;
  channel_instance_id: string;
  chat_id: string;
  chat_type: string;
  user_open_id: string;
  base_agent_id: string;
  current_session_id: string;
  created_at: string;
  updated_at: string;
}

export interface MilestoneItem {
  id: string;
  description: string;
  status: string;
  declared_at_seq: number | null;
  completed_at_seq: number | null;
}

export interface ReviewCheckpoint {
  seq: number;
  milestone_id: string;
  confirmed_at: string;
}

export interface ReviewOutcome {
  aligned: boolean | null;
  experience: string | null;
  step_back_applied: boolean;
  affected_count: number | null;
  trigger_reason: string | null;
  active_milestone: string | null;
  resolved_at: string | null;
}

export interface SessionMilestoneBoard {
  session_id: string;
  run_id: string | null;
  milestones: MilestoneItem[];
  active_milestone_id: string | null;
  latest_checkpoint: ReviewCheckpoint | null;
  latest_review_outcome: ReviewOutcome | null;
  pending_review_reason: string | null;
}

export interface ReviewCycle {
  cycle_id: string;
  run_id: string;
  agent_id: string;
  trigger_reason: string;
  steps_since_last_review: number | null;
  active_milestone: string | null;
  active_milestone_id: string | null;
  hook_advice: string | null;
  aligned: boolean | null;
  experience: string | null;
  step_back_applied: boolean;
  rollback_range: number[] | null;
  affected_count: number | null;
  started_at: string | null;
  resolved_at: string | null;
  raw_notice: string | null;
}

export interface ConversationEvent {
  id: string;
  session_id: string;
  run_id: string | null;
  sequence: number | null;
  kind: string;
  priority: "primary" | "secondary" | "muted" | (string & {});
  title: string;
  summary: string;
  details: Record<string, unknown>;
}

export interface SessionDetail {
  summary: SessionSummary;
  session: SessionRecord | null;
  chat_context: ChatContextRecord | null;
  scheduler_state: AgentStateDetail | null;
  observability: SessionObservability | null;
  milestone_board: SessionMilestoneBoard | null;
  review_cycles: ReviewCycle[];
  conversation_events: ConversationEvent[];
}

export interface RuntimeDecisionEvent {
  kind: "termination" | "compaction" | "step_back" | "rollback" | string;
  sequence: number;
  run_id: string;
  agent_id: string;
  created_at: string;
  summary: string;
  details: Record<string, unknown>;
}

export interface DashboardOverview {
  total_sessions: number;
  total_traces: number;
  total_agents: number;
  total_tokens: number;
  scheduler: SchedulerStats;
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
  condensed_content?: string | null;
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

export interface ContextStepsHiddenEventPayload extends StreamEventBase {
  type: "context_steps_hidden";
  step_ids: string[];
  reason: string;
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

export interface SchedulerAckEventPayload {
  type: "scheduler_ack";
  message?: string | null;
  result_summary?: string | null;
  session_id?: string | null;
  state_id?: string | null;
}

export type AgentStreamEventPayload =
  | RunStartedEventPayload
  | StepDeltaEventPayload
  | StepCompletedEventPayload
  | ContextStepsHiddenEventPayload
  | RunCompletedEventPayload
  | RunFailedEventPayload;

export type StreamEventPayload =
  | AgentStreamEventPayload
  | SchedulerFailedEventPayload
  | SchedulerAckEventPayload;

export function listSessions(limit = 20, offset = 0) {
  return fetchJSON<PageResponse<SessionSummary>>(
    `/api/sessions?limit=${limit}&offset=${offset}`,
  );
}

export function listRuns(params?: { session_id?: string; limit?: number; offset?: number }) {
  const q = new URLSearchParams();
  if (params?.session_id) q.set("session_id", params.session_id);
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.offset) q.set("offset", String(params.offset));
  return fetchJSON<PageResponse<RunResponse>>(`/api/runs?${q}`);
}

export function getRun(runId: string) {
  return fetchJSON<RunResponse>(`/api/runs/${runId}`);
}

export function getSessionSteps(
  sessionId: string,
  params?: {
    start_seq?: number;
    end_seq?: number;
    run_id?: string;
    agent_id?: string;
    limit?: number;
    order?: "asc" | "desc";
  },
) {
  const q = new URLSearchParams();
  if (params?.start_seq) q.set("start_seq", String(params.start_seq));
  if (params?.end_seq) q.set("end_seq", String(params.end_seq));
  if (params?.run_id) q.set("run_id", params.run_id);
  if (params?.agent_id) q.set("agent_id", params.agent_id);
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.order) q.set("order", params.order);
  const query = q.toString();
  return fetchJSON<PageResponse<StepResponse>>(
    `/api/sessions/${sessionId}/steps${query ? `?${query}` : ""}`,
  );
}

export function getSessionSummary(sessionId: string) {
  return fetchJSON<SessionSummary>(`/api/sessions/${sessionId}/summary`);
}

export function getSessionDetail(sessionId: string) {
  return fetchJSON<SessionDetail>(`/api/sessions/${sessionId}`);
}

export function getDashboardOverview() {
  return fetchJSON<DashboardOverview>("/api/overview");
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

export interface SessionObservability {
  recent_traces: TraceListItem[];
  decision_events: RuntimeDecisionEvent[];
}

export interface TraceTimelineEvent {
  kind: string;
  timestamp: string | null;
  sequence: number | null;
  run_id: string | null;
  agent_id: string | null;
  span_id: string | null;
  step_id: string | null;
  title: string;
  summary: string;
  status: string;
  details: Record<string, unknown>;
}

export interface TraceMainlineEvent {
  id: string;
  kind: string;
  title: string;
  summary: string;
  status: string;
  sequence: number | null;
  timestamp: string | null;
  run_id: string | null;
  agent_id: string | null;
  details: Record<string, unknown>;
}

export interface TraceLlmCall {
  span_id: string;
  run_id: string;
  agent_id: string;
  model: string | null;
  provider: string | null;
  finish_reason: string | null;
  duration_ms: number | null;
  first_token_latency_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  total_tokens: number | null;
  message_count: number;
  tool_schema_count: number;
  response_tool_call_count: number;
  output_preview: string | null;
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
  runtime_decisions: RuntimeDecisionEvent[];
  timeline_events: TraceTimelineEvent[];
  mainline_events: TraceMainlineEvent[];
  review_cycles: ReviewCycle[];
  llm_calls: TraceLlmCall[];
}

export function listTraces(params?: {
  agent_id?: string;
  session_id?: string;
  user_id?: string;
  status?: string;
  limit?: number;
  offset?: number;
}) {
  const q = new URLSearchParams();
  if (params?.agent_id) q.set("agent_id", params.agent_id);
  if (params?.session_id) q.set("session_id", params.session_id);
  if (params?.user_id) q.set("user_id", params.user_id);
  if (params?.status) q.set("status", params.status);
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.offset) q.set("offset", String(params.offset));
  return fetchJSON<PageResponse<TraceListItem>>(`/api/traces?${q}`);
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
  relevant_memory_max_token: number;
  stream_cleanup_timeout: number;
  compact_prompt: string;
  enable_context_rollback: boolean;
  enable_goal_directed_review: boolean;
  review_step_interval: number;
  review_on_error: boolean;
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
  is_default: boolean;
  model_provider: string;
  model_name: string;
  system_prompt: string;
  allowed_tools: string[] | null;
  allowed_skills: string[] | null;
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
  allowed_tools: string[] | null;
  allowed_skills: string[] | null;
  options: AgentOptionsPayload;
  model_params: ModelParamsPayload;
}

export interface AvailableTool {
  name: string;
  description: string;
  type: "builtin" | "agent";
  agent_name?: string;
}

export interface AvailableSkill {
  name: string;
  description: string;
}

export interface AgentProviderCapability {
  value: string;
  label: string;
  default_model_name: string | null;
  requires_base_url: boolean;
  requires_api_key_env_name: boolean;
}

export interface AgentCapabilities {
  providers: AgentProviderCapability[];
}

export interface RuntimeDefaultAgentConfig {
  id: string;
  name: string;
  description: string;
  model_provider: string;
  model_name: string;
  system_prompt: string;
  allowed_tools: string[] | null;
  allowed_skills: string[] | null;
  model_params: ModelParamsPayload;
}

export interface RuntimeConfigEditable {
  skills_dirs: string[];
  default_agent: RuntimeDefaultAgentConfig;
}

export interface RuntimeConfigSnapshot {
  editable: RuntimeConfigEditable;
  effective: Record<string, unknown>;
  readonly: Record<string, unknown>;
  runtime_only: boolean;
  restart_required: string[];
}

export function listAvailableTools(exclude?: string) {
  const q = exclude ? `?exclude=${encodeURIComponent(exclude)}` : "";
  return fetchJSON<AvailableTool[]>(`/api/agents/tools/available${q}`);
}

export function listAvailableSkills() {
  return fetchJSON<AvailableSkill[]>("/api/agents/skills/available");
}

export function listAgents() {
  return fetchJSON<AgentConfig[]>("/api/agents");
}

export function getAgentCapabilities() {
  return fetchJSON<AgentCapabilities>("/api/agents/capabilities");
}

export function getRuntimeConfig() {
  return fetchJSON<RuntimeConfigSnapshot>("/api/config/runtime");
}

export function updateRuntimeConfig(data: RuntimeConfigEditable) {
  return fetchJSON<RuntimeConfigSnapshot>("/api/config/runtime", {
    method: "PUT",
    body: JSON.stringify(data),
  });
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
  const res = await fetch(apiUrl(`/api/agents/${agentId}`), { method: "DELETE" });
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

export interface SchedulerRunResult {
  run_id: string | null;
  termination_reason: string;
  summary: string | null;
  error: string | null;
  completed_at: string | null;
}

export interface AgentStateListItem {
  id: string;
  root_state_id: string | null;
  status: string;
  task: UserInput;
  parent_id: string | null;
  wake_condition: WakeConditionResponse | null;
  result_summary: string | null;
  last_run_result: SchedulerRunResult | null;
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
  root_state_id: string | null;
  session_id: string;
  status: string;
  task: UserInput;
  parent_id: string | null;
  pending_input: UserInput | null;
  config_overrides: Record<string, unknown>;
  wake_condition: WakeConditionResponse | null;
  result_summary: string | null;
  last_run_result: SchedulerRunResult | null;
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

export interface SchedulerTreeStats {
  total: number;
  running: number;
  waiting: number;
  queued: number;
  idle: number;
  completed: number;
  failed: number;
  cancelled: number;
}

export interface SchedulerTreeNode {
  state_id: string;
  root_state_id: string;
  parent_state_id: string | null;
  child_ids: string[];
  session_id: string | null;
  agent_id: string;
  task_id: string | null;
  status: string;
  depth: number;
  created_at: string | null;
  updated_at: string | null;
  completed_at: string | null;
  wake_condition: WakeConditionResponse | null;
  pending_event_count: number;
  last_error: string | null;
  result_summary: string | null;
  last_run_result: SchedulerRunResult | null;
}

export interface SchedulerTree {
  root_state_id: string;
  root_session_id: string | null;
  nodes: SchedulerTreeNode[];
  stats: SchedulerTreeStats;
  generated_at: string;
}

export function listAgentStates(params?: { status?: string; limit?: number; offset?: number }) {
  const q = new URLSearchParams();
  if (params?.status) q.set("status", params.status);
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.offset) q.set("offset", String(params.offset));
  return fetchJSON<PageResponse<AgentStateListItem>>(`/api/scheduler/states?${q}`);
}

export function getAgentState(stateId: string) {
  return fetchJSON<AgentStateDetail>(`/api/scheduler/states/${stateId}`);
}

export function getAgentStateChildren(stateId: string) {
  return fetchJSON<AgentStateListItem[]>(`/api/scheduler/states/${stateId}/children`);
}

export function getSchedulerTree(stateId: string) {
  return fetchJSON<SchedulerTree>(`/api/scheduler/states/${stateId}/tree`);
}

export function getSchedulerStats() {
  return fetchJSON<SchedulerStats>(`/api/scheduler/stats`);
}

// ── Chat Sessions ──────────────────────────────────────────────────────

export type ChatSessionItem = SessionSummary;

export interface SessionActionResult {
  session_id: string;
  source_session_id: string | null;
}

export function listAgentSessions(agentId: string) {
  return fetchJSON<PageResponse<ChatSessionItem>>(`/api/agents/${agentId}/sessions`);
}

export function createAgentSession(agentId: string) {
  return fetchJSON<SessionActionResult>(`/api/agents/${agentId}/sessions`, {
    method: "POST",
  });
}

export function forkSession(sessionId: string, contextSummary: string) {
  return fetchJSON<SessionActionResult>(`/api/sessions/${sessionId}/fork`, {
    method: "POST",
    body: JSON.stringify({ context_summary: contextSummary }),
  });
}

export async function deleteSession(sessionId: string) {
  const res = await fetch(apiUrl(`/api/sessions/${sessionId}`), { method: "DELETE" });
  if (!res.ok) throw new Error(`Delete failed: ${res.status}`);
}

// ── Session Input Stream ───────────────────────────────────────────────

export function sessionInputStreamUrl(sessionId: string) {
  return apiUrl(`/api/sessions/${sessionId}/input`);
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

export function cancelSession(sessionId: string, reason?: string) {
  return fetchJSON<{ ok: boolean; session_id: string; state_id: string }>(
    `/api/sessions/${sessionId}/cancel`,
    {
      method: "POST",
      body: JSON.stringify({ reason }),
    },
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
