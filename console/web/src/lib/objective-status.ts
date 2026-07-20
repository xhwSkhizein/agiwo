import type { ObjectiveView } from "@/lib/api";
import type { PillBadgeVariant } from "@/components/pill-badge";

export function pickActiveObjective(
  objectives: ObjectiveView[],
): ObjectiveView | null {
  if (objectives.length === 0) {
    return null;
  }
  return (
    objectives.find((item) => !item.is_terminal) ??
    objectives[objectives.length - 1]
  );
}

export function objectiveStatusVariant(status: string): PillBadgeVariant {
  const normalized = status.toUpperCase();
  if (normalized === "COMPLETED") return "success";
  if (normalized === "FAILED") return "error";
  if (normalized === "RUNNING" || normalized === "DRAINING") return "running";
  if (
    normalized === "WAITING_USER" ||
    normalized === "BUDGET_PAUSED" ||
    normalized === "USER_PAUSED"
  ) {
    return "warning";
  }
  return "pending";
}

export function objectiveFocusText(objective: ObjectiveView): string {
  if (objective.delivery_report) {
    return objective.delivery_report;
  }
  const waiting = [...objective.timeline]
    .reverse()
    .find(
      (node) =>
        node.kind.includes("Waiting") ||
        node.kind.includes("StatusChanged") ||
        node.kind.includes("Decision"),
    );
  if (waiting?.summary) {
    return waiting.summary;
  }
  const latest = objective.timeline[objective.timeline.length - 1];
  return latest?.summary || objective.status;
}

/** Compact process summary: keep the last few high-signal facts. */
export function compactTimelineNodes(
  objective: ObjectiveView,
  limit = 4,
): ObjectiveView["timeline"] {
  const preferred = new Set([
    "ObjectiveCreated",
    "ObjectiveUserInput",
    "AssignmentCreated",
    "AssignmentExecutionStarted",
    "AssignmentOutcome",
    "DecisionAccepted",
    "ObjectiveStatusChanged",
    "WaitingIntervalStarted",
    "ObjectiveDelivered",
    "SystemFault",
    "DrainStarted",
    "DrainCompleted",
    "BudgetUsageRecorded",
  ]);
  const keyNodes = objective.timeline.filter((node) => preferred.has(node.kind));
  const source = keyNodes.length > 0 ? keyNodes : objective.timeline;
  return source.slice(-limit);
}

export function activeRootRunId(objective: ObjectiveView): string | null {
  const rootRuns = objective.root_runs ?? [];
  const running = rootRuns.find(
    (item) => item.status.toUpperCase() === "RUNNING",
  );
  if (running) {
    return running.run_id;
  }
  const completed = [...rootRuns]
    .reverse()
    .find((item) => item.status.toUpperCase() === "COMPLETED");
  return completed?.run_id ?? rootRuns.at(-1)?.run_id ?? null;
}
