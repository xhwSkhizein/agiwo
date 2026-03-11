import type { WakeConditionResponse } from "@/lib/api";

export type WaitsetProgress = {
  completed: number;
  total: number;
  percent: number;
};

export function getWaitsetProgress(
  wakeCondition: WakeConditionResponse | null | undefined
): WaitsetProgress {
  const total = wakeCondition?.wait_for.length ?? 0;
  const completed = wakeCondition?.completed_ids.length ?? 0;
  const percent =
    total > 0 ? Math.min((completed / total) * 100, 100) : 0;

  return { completed, total, percent };
}

export function formatWakeConditionTimer(
  wakeCondition: WakeConditionResponse
): string {
  if (
    wakeCondition.time_value !== null &&
    wakeCondition.time_value !== undefined &&
    wakeCondition.time_unit
  ) {
    return `${wakeCondition.time_value} ${wakeCondition.time_unit}`;
  }

  if (wakeCondition.wakeup_at) {
    return new Date(wakeCondition.wakeup_at).toLocaleTimeString();
  }

  return wakeCondition.type;
}

export function formatWakeConditionSummary(
  wakeCondition: WakeConditionResponse | null | undefined
): string {
  if (!wakeCondition) {
    return "-";
  }

  if (wakeCondition.type === "waitset") {
    const progress = getWaitsetProgress(wakeCondition);
    return `waitset ${progress.completed}/${progress.total} (${wakeCondition.wait_mode})`;
  }

  if (
    wakeCondition.type === "timer" ||
    wakeCondition.type === "periodic"
  ) {
    return `${wakeCondition.type}: ${formatWakeConditionTimer(wakeCondition)}`;
  }

  if (wakeCondition.type === "task_submitted") {
    return "task_submitted";
  }

  return wakeCondition.type;
}
