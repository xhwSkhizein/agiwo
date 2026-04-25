"use client";

import { SectionCard } from "@/components/section-card";
import type { ReviewCycle, SessionMilestoneBoard } from "@/lib/api";
import { formatLocalDateTime } from "@/lib/time";

function milestoneStatusClass(status: string): string {
  if (status === "completed") {
    return "border-emerald-500/40 bg-emerald-500/10 text-emerald-200";
  }
  if (status === "active") {
    return "border-accent bg-panel-strong text-foreground";
  }
  if (status === "abandoned") {
    return "border-red-500/40 bg-red-500/10 text-red-200";
  }
  return "border-line bg-panel-muted text-ink-muted";
}

function latestReviewSummary(board: SessionMilestoneBoard | null): string {
  const outcome = board?.latest_review_outcome;
  if (!outcome) {
    return "No review outcome recorded";
  }
  if (outcome.aligned === true) {
    return "Latest review stayed aligned";
  }
  if (outcome.aligned === false && outcome.step_back_applied) {
    return `Latest review triggered step-back for ${outcome.affected_count ?? 0} steps`;
  }
  if (outcome.aligned === false) {
    return "Latest review detected drift";
  }
  return "Latest checkpoint recorded";
}

function isLatestReviewTarget(
  latestCycle: ReviewCycle | null,
  milestone: SessionMilestoneBoard["milestones"][number],
): boolean {
  if (!latestCycle) {
    return false;
  }
  if (latestCycle.active_milestone_id) {
    return latestCycle.active_milestone_id === milestone.id;
  }
  return latestCycle.active_milestone === milestone.description;
}

export function MilestoneBoard({
  board,
  reviewCycles,
}: {
  board: SessionMilestoneBoard | null;
  reviewCycles: ReviewCycle[];
}) {
  if (!board) {
    return (
      <SectionCard title="Milestone Board" bodyClassName="px-4 py-6">
        <div className="rounded-xl border border-dashed border-line px-4 py-6 text-sm text-ink-muted">
          No milestone board has been declared for this session yet.
        </div>
      </SectionCard>
    );
  }

  const activeMilestone =
    board.milestones.find((milestone) => milestone.id === board.active_milestone_id) ??
    null;
  const latestCycle = reviewCycles[reviewCycles.length - 1] ?? null;

  return (
    <SectionCard title="Milestone Board" bodyClassName="space-y-4 px-4 py-4">
      <div className="grid gap-3 md:grid-cols-3">
        <div className="rounded-xl border border-line bg-panel-muted px-3 py-3">
          <div className="text-[11px] uppercase tracking-wide text-ink-faint">
            Current Focus
          </div>
          <div className="mt-2 text-sm text-foreground">
            {activeMilestone?.description || "No active milestone"}
          </div>
        </div>
        <div className="rounded-xl border border-line bg-panel-muted px-3 py-3">
          <div className="text-[11px] uppercase tracking-wide text-ink-faint">
            Latest Review
          </div>
          <div className="mt-2 text-sm text-foreground">
            {latestReviewSummary(board)}
          </div>
        </div>
        <div className="rounded-xl border border-line bg-panel-muted px-3 py-3">
          <div className="text-[11px] uppercase tracking-wide text-ink-faint">
            Pending Trigger
          </div>
          <div className="mt-2 text-sm text-foreground">
            {board.pending_review_reason || "None"}
          </div>
        </div>
      </div>

      {board.latest_checkpoint ? (
        <div className="rounded-xl border border-line bg-panel px-3 py-3 text-sm text-ink-muted">
          Latest checkpoint on <span className="text-foreground">{board.latest_checkpoint.milestone_id}</span>
          {" · "}
          seq {board.latest_checkpoint.seq}
          {" · "}
          {formatLocalDateTime(board.latest_checkpoint.confirmed_at)}
        </div>
      ) : null}

      <div className="space-y-2">
        {board.milestones.map((milestone) => (
          <div
            key={milestone.id}
            className={`rounded-xl border px-3 py-3 ${
              milestone.id === board.active_milestone_id
                ? "border-accent bg-panel-strong"
                : "border-line bg-panel"
            }`}
          >
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-medium text-foreground">
                {milestone.description}
              </span>
              <span
                className={`rounded-full border px-2 py-0.5 text-[11px] uppercase tracking-wide ${milestoneStatusClass(milestone.status)}`}
              >
                {milestone.status}
              </span>
            </div>
            <div className="mt-1 flex flex-wrap gap-3 text-xs text-ink-muted">
              <span>id {milestone.id}</span>
              <span>declared seq {milestone.declared_at_seq ?? "-"}</span>
              {milestone.completed_at_seq !== null ? (
                <span>completed seq {milestone.completed_at_seq}</span>
              ) : null}
              {isLatestReviewTarget(latestCycle, milestone) ? (
                <span>latest review target</span>
              ) : null}
            </div>
          </div>
        ))}
      </div>
    </SectionCard>
  );
}

export default MilestoneBoard;
