import type { SchedulerRunResult } from "./api";

export type SchedulerRunResultTone = "success" | "warning" | "danger" | "neutral";

const TERMINATION_REASON_LABELS: Record<string, string> = {
  cancelled: "Cancelled",
  completed: "Completed",
  error: "Errored",
  failed: "Failed",
  max_turns: "Max turns reached",
  no_progress: "No progress",
  sleeping: "Sleeping",
  timeout: "Timed out",
};

export type SchedulerRunResultView = {
  reasonLabel: string | null;
  tone: SchedulerRunResultTone;
  summary: string | null;
  error: string | null;
  message: string | null;
  completedAt: string | null;
  runId: string | null;
};

export function formatSchedulerTerminationReason(reason: string | null | undefined): string {
  if (!reason) {
    return "Unknown";
  }
  const label = TERMINATION_REASON_LABELS[reason];
  if (label) {
    return label;
  }
  return reason
    .split("_")
    .filter(Boolean)
    .map((part) => part[0]?.toUpperCase() + part.slice(1))
    .join(" ");
}

export function getSchedulerRunResultTone(
  result: SchedulerRunResult | null | undefined,
): SchedulerRunResultTone {
  const reason = result?.termination_reason;
  if (reason === "completed") {
    return "success";
  }
  if (reason === "cancelled" || reason === "timeout" || reason === "sleeping") {
    return "warning";
  }
  if (reason) {
    return "danger";
  }
  return "neutral";
}

export function getSchedulerRunResultView(
  result: SchedulerRunResult | null | undefined,
  fallbackSummary?: string | null,
): SchedulerRunResultView | null {
  const summary = result?.summary ?? fallbackSummary ?? null;
  const error = result?.error ?? null;
  const message = error ?? summary;

  if (!result && !message) {
    return null;
  }

  return {
    reasonLabel: result ? formatSchedulerTerminationReason(result.termination_reason) : null,
    tone: getSchedulerRunResultTone(result),
    summary,
    error,
    message,
    completedAt: result?.completed_at ?? null,
    runId: result?.run_id ?? null,
  };
}
