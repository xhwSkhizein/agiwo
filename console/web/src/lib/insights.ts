import type {
  MilestoneItem,
  ReviewCycle,
  SessionMilestoneBoard,
  TraceDetail,
} from "@/lib/api";

export function milestoneStatusClass(status: string): string {
  if (status === "completed") {
    return "border-success/40 bg-success/10 text-success";
  }
  if (status === "active") {
    return "border-accent/70 bg-accent/10 text-foreground";
  }
  if (status === "abandoned") {
    return "border-danger/40 bg-danger/10 text-danger";
  }
  return "border-line bg-panel-muted text-ink-muted";
}

export function reviewSummary(
  cycle: ReviewCycle | null | undefined,
  emptyLabel = "No review cycle yet",
): string {
  if (!cycle) {
    return emptyLabel;
  }
  if (cycle.aligned === true) {
    return "Aligned";
  }
  if (cycle.aligned === false && cycle.step_back_applied) {
    return `${cycle.affected_count ?? 0} steps condensed`;
  }
  if (cycle.aligned === false) {
    return "Drift detected";
  }
  return "Checkpoint recorded";
}

export function activeMilestoneFromBoard(
  board: SessionMilestoneBoard | null | undefined,
): MilestoneItem | null {
  return (
    board?.milestones.find(
      (milestone) => milestone.id === board.active_milestone_id,
    ) ?? null
  );
}

export function latestObjective(trace: TraceDetail): string {
  const latestCycle = trace.review_cycles[trace.review_cycles.length - 1];
  if (latestCycle?.active_milestone) {
    return latestCycle.active_milestone;
  }

  const milestoneEvent = [...trace.mainline_events]
    .reverse()
    .find((event) => event.kind === "milestone_update");
  const rawMilestones = milestoneEvent?.details?.milestones;
  const milestones = Array.isArray(rawMilestones) ? rawMilestones : [];
  const active = milestones.find(
    (item) =>
      item &&
      typeof item === "object" &&
      "status" in item &&
      item.status === "active" &&
      "description" in item,
  );
  if (active && typeof active === "object" && "description" in active) {
    return String(active.description);
  }
  return "No active milestone";
}

export function latestAlignment(trace: TraceDetail): string {
  const latestCycle = trace.review_cycles[trace.review_cycles.length - 1];
  return reviewSummary(latestCycle, "No review cycle");
}
