import type {
  RunMetricsPayload,
  RunMetricsSummary,
  SpanMetricsPayload,
  StepMetricsPayload,
  UsageSource,
} from "@/lib/api";

export type MetricsPayload =
  | Record<string, unknown>
  | RunMetricsPayload
  | SpanMetricsPayload
  | StepMetricsPayload
  | null
  | undefined;

export type ParsedTokenMetrics = {
  durationMs: number;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  cacheReadTokens: number;
  cacheCreationTokens: number;
  toolCallsCount: number;
  stepCount: number;
  tokenCost: number;
  usageSource: UsageSource | null;
};

export const EMPTY_RUN_METRICS: RunMetricsSummary = {
  run_count: 0,
  completed_run_count: 0,
  step_count: 0,
  tool_calls_count: 0,
  duration_ms: 0,
  input_tokens: 0,
  output_tokens: 0,
  total_tokens: 0,
  cache_read_tokens: 0,
  cache_creation_tokens: 0,
  token_cost: 0,
};

function toNumber(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return 0;
}

function readMetric(metrics: MetricsPayload, ...keys: string[]): number {
  if (!metrics) {
    return 0;
  }
  const record = metrics as Record<string, unknown>;
  for (const key of keys) {
    if (key in record) {
      return toNumber(record[key]);
    }
  }
  return 0;
}

function readUsageSource(metrics: MetricsPayload): UsageSource | null {
  if (!metrics) {
    return null;
  }
  const record = metrics as Record<string, unknown>;
  const value = "usage_source" in record ? record.usage_source : null;
  if (value === "provider" || value === "estimated" || value === "mixed") {
    return value;
  }
  return null;
}

export function normalizeRunMetricsSummary(
  metrics: RunMetricsSummary | null | undefined
): RunMetricsSummary {
  if (!metrics) {
    return EMPTY_RUN_METRICS;
  }
  return {
    run_count: toNumber(metrics.run_count),
    completed_run_count: toNumber(metrics.completed_run_count),
    step_count: toNumber(metrics.step_count),
    tool_calls_count: toNumber(metrics.tool_calls_count),
    duration_ms: toNumber(metrics.duration_ms),
    input_tokens: toNumber(metrics.input_tokens),
    output_tokens: toNumber(metrics.output_tokens),
    total_tokens: toNumber(metrics.total_tokens),
    cache_read_tokens: toNumber(metrics.cache_read_tokens),
    cache_creation_tokens: toNumber(metrics.cache_creation_tokens),
    token_cost: toNumber(metrics.token_cost),
  };
}

export function parseGenericMetrics(metrics: MetricsPayload): ParsedTokenMetrics {
  return {
    durationMs: readMetric(metrics, "duration_ms"),
    inputTokens: readMetric(metrics, "input_tokens", "tokens.input"),
    outputTokens: readMetric(metrics, "output_tokens", "tokens.output"),
    totalTokens: readMetric(metrics, "total_tokens", "tokens.total"),
    cacheReadTokens: readMetric(
      metrics,
      "cache_read_tokens",
      "tokens.cache_read"
    ),
    cacheCreationTokens: readMetric(
      metrics,
      "cache_creation_tokens",
      "tokens.cache_creation"
    ),
    toolCallsCount: readMetric(metrics, "tool_calls_count"),
    stepCount: readMetric(metrics, "steps_count", "step_count"),
    tokenCost: readMetric(metrics, "token_cost", "cost.token"),
    usageSource: readUsageSource(metrics),
  };
}

export function formatUsd(amount: number): string {
  return `$${toNumber(amount).toFixed(4)}`;
}

export function formatTokenCount(tokens: number): string {
  return Math.round(toNumber(tokens)).toLocaleString();
}

export function formatDurationMs(durationMs: number): string {
  const value = toNumber(durationMs);
  if (value <= 0) {
    return "-";
  }
  if (value < 1000) {
    return `${Math.round(value)}ms`;
  }
  return `${(value / 1000).toFixed(2)}s`;
}

export function formatUsageSource(value: UsageSource | null): string {
  if (value === "provider") {
    return "usage provider";
  }
  if (value === "estimated") {
    return "usage estimated";
  }
  if (value === "mixed") {
    return "usage mixed";
  }
  return "-";
}
